"""Build a post-codegen dependency graph without importing Torch."""

import html
import json
import textwrap
from collections import defaultdict


def output_records(compilation):
    leaves = []

    def visit(value, path):
        if isinstance(value, dict):
            for kind in ("tuple", "list"):
                if kind in value:
                    for index, item in enumerate(value[kind]):
                        visit(item, f"{path}[{index}]")
                    return
            if "dict" in value:
                for key, item in value["dict"]:
                    visit(item, f"{path}[{key!r}]")
                return
            if "node" in value:
                name = value["node"]
                leaves.append(
                    {
                        "name": name,
                        "path": path,
                        "value": compilation.get("values", {}).get(name, {}),
                    }
                )
                return
        kind = "scalar" if isinstance(value, dict) and "symbol" in value else "literal"
        leaves.append(
            {"name": repr(value), "path": path, "value": {"kind": kind, "value": value}}
        )

    for node in compilation.get("graph", {}).get("record", {}).get("nodes", []):
        if node.get("op") == "output":
            args = node.get("args", {}).get("tuple", [])
            if args:
                visit(args[0], "output")
    if "graph_outputs" in compilation:
        values = compilation["graph_outputs"]
        return [
            {
                "name": leaves[index]["name"]
                if len(leaves) == len(values)
                else f"output[{index}]",
                "path": leaves[index]["path"]
                if len(leaves) == len(values)
                else f"output[{index}]",
                "value": value,
                "evidence": "final_graph_outputs",
            }
            for index, value in enumerate(values)
        ], []
    if leaves:
        return [
            {**leaf, "evidence": "post_grad_output_and_lowered_values"}
            for leaf in leaves
        ], [
            "旧记录的输出由 post-grad 返回值及 lowering 映射恢复；未单独记录最终返回时的布局修正，需用 wrapper 核对。"
        ]
    return [], ["未记录图的返回值；不会把无后继 kernel 自动推断为输出。"]


def build_dependencies(compilation):
    calls = compilation.get("calls", [])
    input_nodes = [
        {
            "id": f"input{index}",
            "name": name,
            "value": compilation.get("graph_inputs", {}).get(
                name, compilation.get("values", {}).get(name, {})
            ),
        }
        for index, name in enumerate(compilation.get("input_names", []))
    ]
    inputs = defaultdict(list)
    for node in input_nodes:
        inputs[node["value"].get("buffer", node["name"])].append(node)
    removed = set(compilation.get("removed_buffers", []))
    emitted = {
        member["operation"]
        for call in calls
        for row in compilation.get("call_schedules", {}).get(
            call.get("provenance_key"), []
        )
        for member in row.get("members", [])
    }
    bridges = defaultdict(list)
    for row in compilation.get("scheduler", []):
        operations = [m["operation"] for m in row.get("members", [])]
        if row.get("type") == "NopKernelSchedulerNode" and not emitted.intersection(
            operations
        ):
            for buffer in row.get("writes", []):
                bridges[buffer].append(
                    {"operations": operations, "reads": row.get("reads", [])}
                )
    writers, readers = defaultdict(set), defaultdict(set)
    edges, nodes, unresolved, boundary_edges = {}, [], [], []

    def boundary_edge(source, target, kind, evidence):
        edge = next(
            (
                e
                for e in boundary_edges
                if (e["source"], e["target"], e["kind"]) == (source, target, kind)
            ),
            None,
        )
        if edge is None:
            edge = {"source": source, "target": target, "kind": kind, "evidence": []}
            boundary_edges.append(edge)
        if evidence not in edge["evidence"]:
            edge["evidence"].append(evidence)

    def add_edge(source, target, kind, evidence):
        if source == target:
            return
        edge = edges.setdefault(
            (source, target, kind),
            {"source": source, "target": target, "kind": kind, "evidence": []},
        )
        if evidence not in edge["evidence"]:
            edge["evidence"].append(evidence)

    def resolve(buffer, seen=(), via=()):
        if writers.get(buffer):
            return [
                {"source": source, "source_buffer": buffer, "via_nop": list(via)}
                for source in sorted(writers[buffer])
            ]
        if buffer in seen:
            return [{"unresolved": buffer, "reason": "non_emitting_dependency_cycle"}]
        if buffer in bridges:
            return [
                item
                for bridge in bridges[buffer]
                for parent in bridge["reads"]
                for item in resolve(
                    parent, (*seen, buffer), (*via, *bridge["operations"])
                )
            ] or [{"unresolved": buffer, "reason": "empty_non_emitting_dependency"}]
        if buffer in inputs:
            return [
                {
                    "input": node["name"],
                    "input_id": node["id"],
                    "source_buffer": buffer,
                    "via_nop": list(via),
                }
                for node in inputs[buffer]
            ]
        return [
            {"unresolved": buffer, "reason": "producer_not_recorded_before_consumer"}
        ]

    for index, call in enumerate(calls):
        reads, writes = set(call.get("reads", [])), set(call.get("writes", []))
        node = {
            "id": index,
            "call_id": call.get("id"),
            "name": call["name"],
            "provenance_key": call.get("provenance_key"),
            "reads": sorted(reads),
            "writes": sorted(writes),
            "internal_buffers": [],
            "inputs": [],
            "gaps": [],
        }
        if "reads" not in call or "writes" not in call:
            node["gaps"].append("缺少调用的读写记录；无连线不能解释为独立。")
        for buffer in sorted(reads):
            if (
                buffer in writes
                and buffer not in inputs
                and not writers[buffer]
                and buffer not in bridges
            ):
                node["internal_buffers"].append(buffer)
                continue
            for item in resolve(buffer):
                if "source" in item:
                    add_edge(item["source"], index, "RAW", {"buffer": buffer, **item})
                elif "input" in item:
                    node["inputs"].append(item["input"])
                    boundary_edge(
                        item["input_id"],
                        f"kernel{index}",
                        "INPUT",
                        {"buffer": buffer, **item},
                    )
                else:
                    unresolved.append({"target": index, "buffer": buffer, **item})
                    node["gaps"].append(f"{buffer} 的前驱未解析：{item['reason']}")
        for buffer in sorted(writes - removed):
            for source in sorted(writers[buffer]):
                add_edge(source, index, "WAW", {"buffer": buffer})
            for source in sorted(readers[buffer]):
                add_edge(source, index, "WAR", {"buffer": buffer})
        for buffer in reads - removed - set(node["internal_buffers"]):
            readers[buffer].add(index)
        for buffer in writes:
            if call.get("write_coverage", {}).get(buffer) == "full":
                writers[buffer].clear()
                readers[buffer].clear()
            writers[buffer].add(index)
        node["inputs"] = sorted(set(node["inputs"]))
        nodes.append(node)
    outputs, boundary_gaps = output_records(compilation)
    for index, node in enumerate(outputs):
        node.update({"id": f"output{index}", "index": index, "gaps": []})
        value = node["value"]
        if "buffer" in value:
            for item in resolve(value["buffer"]):
                if "source" in item:
                    boundary_edge(
                        f"kernel{item['source']}",
                        node["id"],
                        "OUTPUT",
                        {"buffer": value["buffer"], **item},
                    )
                elif "input" in item:
                    boundary_edge(
                        item["input_id"],
                        node["id"],
                        "OUTPUT",
                        {"buffer": value["buffer"], **item},
                    )
                else:
                    unresolved.append(
                        {"target": node["id"], "buffer": value["buffer"], **item}
                    )
                    node["gaps"].append(
                        f"输出 {value['buffer']} 的来源未解析：{item['reason']}"
                    )
        elif value.get("kind") != "literal":
            node["gaps"].append(
                "没有可追溯的输出 buffer；标量表达式或不支持的值不补造 kernel 依赖。"
            )
    result = {
        "schema_version": 2,
        "scope": "final_wrapper_calls_and_lowered_buffer_dependencies",
        "nodes": nodes,
        "edges": [edges[key] for key in sorted(edges)],
        "inputs": input_nodes,
        "outputs": outputs,
        "boundary_edges": boundary_edges,
        "unresolved": unresolved,
        "coverage_gaps": [
            "节点是最终 wrapper 调用实例；extern 调用可能对应多个库内部 GPU kernel。",
            "RAW 是降低后逻辑 buffer 的先写后读数据依赖；经过不发射 kernel 的 Nop 节点时保留追溯证据。",
            "WAR/WAW 是同名 buffer 的保守顺序约束；未比较访问区间，不证明适合纵向融合。",
            "不把同流发射先后、共用输入、FX 邻接或 allocator 内存复用当作 RAW 依赖。",
            "未覆盖不同 buffer 名之间的 mutation alias、stream/event 同步、控制流及外部库隐藏副作用；无边不证明可并行或可融合。",
            "输入/输出是本次编译图的边界，可能包含 lifted 参数；不等同于原模型的用户 API。未连线的输入表示没有记录到 buffer 消费，标量的 shape/launch 用途未包含在读写记录中。",
            *boundary_gaps,
        ],
    }
    if compilation.get("mutated_inputs"):
        result["coverage_gaps"].append(
            "模型存在输入 mutation，alias 与额外顺序约束需要另行核对。"
        )
    return result


def dependency_svg(graph, link_prefix=""):
    nodes, edges = graph["nodes"], graph["edges"]
    inputs, outputs = graph.get("inputs", []), graph.get("outputs", [])
    ranks, layers = {}, defaultdict(list)
    for node in nodes:
        index = node["id"]
        ranks[index] = max(
            (ranks[e["source"]] + 1 for e in edges if e["target"] == index),
            default=0,
        )
        layers[ranks[index]].append(index)
    box_width, box_height, step_x, step_y = 340, 134, 390, 224
    left = 350 if inputs else 0
    kernel_width = max(1, max(map(len, layers.values()), default=1)) * step_x + 140
    width = left + kernel_width + (320 if outputs else 0)
    height = max(
        max(1, len(layers)) * step_y + 70, max(len(inputs), len(outputs)) * 110 + 80
    )
    positions = {
        index: (left + 30 + slot * step_x, 50 + rank * step_y)
        for rank, indices in layers.items()
        for slot, index in enumerate(indices)
    }
    colors = {
        "RAW": "#2468ac",
        "WAR": "#b06c12",
        "WAW": "#b06c12",
        "INPUT": "#14805a",
        "OUTPUT": "#8654b3",
    }
    boundary_positions = {
        node["id"]: (24, 50 + index * 110) for index, node in enumerate(inputs)
    }
    output_y = 50
    for node in outputs:
        producers = [
            int(edge["source"].removeprefix("kernel"))
            for edge in graph["boundary_edges"]
            if edge["target"] == node["id"] and edge["source"].startswith("kernel")
        ]
        y = max(output_y, max((positions[p][1] for p in producers), default=50))
        boundary_positions[node["id"]] = (left + kernel_width + 14, y)
        output_y = y + 110
    height = max(height, output_y + 30)
    doc = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img" aria-label="最终 kernel 依赖图">'
        "<title>输入、最终 kernel 调用与输出的 buffer 依赖</title>"
        "<style>text{font-family:system-ui,sans-serif;fill:#23364d}"
        ".kd-node:hover rect{stroke-width:3}.kd-edge:hover path{stroke-width:4}</style>"
        "<defs>"
    )
    for kind, color in colors.items():
        doc += (
            f'<marker id="kd-arrow-{kind}" viewBox="0 0 10 10" refX="9" refY="5" '
            f'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{color}"/></marker>'
        )
    doc += '</defs><rect width="100%" height="100%" fill="#f8fafc"/>'
    for index, edge in enumerate(graph.get("boundary_edges", [])):
        source, target, kind = edge["source"], edge["target"], edge["kind"]
        source_kernel, target_kernel = (
            source.startswith("kernel"),
            target.startswith("kernel"),
        )
        sx, sy = (
            positions[int(source[6:])] if source_kernel else boundary_positions[source]
        )
        tx, ty = (
            positions[int(target[6:])] if target_kernel else boundary_positions[target]
        )
        x1, y1 = sx + (box_width if source_kernel else 266), sy + 47
        x2, y2 = tx, ty + 47
        if kind == "INPUT" and tx > left + 30:
            lane, top = left - 38 + index % 4 * 6, ty - 14 - index % 3 * 8
            x2, y2 = tx + 24 + index % 4 * 12, ty
            path = f"M{x1},{y1} H{lane} V{top} H{x2} V{y2}"
        else:
            middle = (x1 + x2) / 2
            path = f"M{x1},{y1} C{middle},{y1} {middle},{y2} {x2},{y2}"
        title = f"{source} → {target} · {kind}\n" + json.dumps(
            edge["evidence"], ensure_ascii=False, indent=2
        )
        doc += (
            f'<a class="kd-edge" data-kind="{kind}" href="{html.escape(link_prefix)}#kernel-boundary-edge-{index}">'
            f'<title>{html.escape(title)}</title><path d="{path}" fill="none" stroke="{colors[kind]}" '
            f'stroke-width="1.8" marker-end="url(#kd-arrow-{kind})"/></a>'
        )
    for number, edge in enumerate(edges):
        source, target, kind = edge["source"], edge["target"], edge["kind"]
        sx, sy = positions[source]
        tx, ty = positions[target]
        offset = {"RAW": -9, "WAR": 0, "WAW": 9}[kind]
        if ranks[target] == ranks[source] + 1:
            x1, y1, x2, y2 = (
                sx + box_width / 2 + offset,
                sy + box_height,
                tx + box_width / 2 + offset,
                ty,
            )
            middle = (y1 + y2) / 2
            path = f"M{x1},{y1} C{x1},{middle} {x2},{middle} {x2},{y2}"
            lx, ly = (x1 + x2) / 2 + 12, middle + offset
            anchor = "start"
        else:
            x1, y1, x2, y2 = (
                sx + box_width * 0.75,
                sy + box_height,
                tx + box_width * 0.75,
                ty,
            )
            lane = left + kernel_width - 35 - (number % 5) * 13
            path = f"M{x1},{y1} V{y1 + 22} H{lane} V{y2 - 22} H{x2} V{y2}"
            lx, ly = lane - 8, (y1 + y2) / 2
            anchor = "end"
        buffers = sorted({e["buffer"] for e in edge["evidence"]})
        label = ", ".join(buffers[:2]) + (
            f" +{len(buffers) - 2}" if len(buffers) > 2 else ""
        )
        title = f"K{source} → K{target} · {kind}\n" + json.dumps(
            edge["evidence"], ensure_ascii=False, indent=2
        )
        dash = 'stroke-dasharray="6 4"' if kind != "RAW" else ""
        doc += (
            f'<a class="kd-edge" data-kind="{kind}" href="{html.escape(link_prefix)}#kernel-dependency-edge-{number}">'
            f'<title>{html.escape(title)}</title><path d="{path}" fill="none" stroke="{colors[kind]}" '
            f'stroke-width="2" {dash} marker-end="url(#kd-arrow-{kind})"/>'
            f'<text x="{lx}" y="{ly}" text-anchor="{anchor}" font-size="11" paint-order="stroke" stroke="#f8fafc" stroke-width="4">'
            f"{kind} {html.escape(label)}</text></a>"
        )
    for node in nodes:
        index = node["id"]
        x, y = positions[index]
        lines = textwrap.wrap(node["name"], width=42)[:4]
        title = json.dumps(node, ensure_ascii=False, indent=2)
        color = "#b06c12" if node["gaps"] else "#4f6d8b"
        doc += (
            f'<a class="kd-node" href="{html.escape(link_prefix)}#kernel{index}">'
            f"<title>{html.escape(title)}</title>"
            f'<rect x="{x}" y="{y}" width="{box_width}" height="{box_height}" rx="9" fill="white" stroke="{color}"/>'
            f'<text x="{x + 14}" y="{y + 24}" font-size="16" font-weight="650">K{index} · 调用 #{index}</text>'
        )
        for line_number, line in enumerate(lines):
            doc += f'<text x="{x + 14}" y="{y + 45 + line_number * 15}" font-size="12">{html.escape(line)}</text>'
        summary = f"读 {len(node['reads'])} · 写 {len(node['writes'])}"
        if node["gaps"]:
            summary += f" · {len(node['gaps'])} 项证据缺口"
        doc += f'<text x="{x + 14}" y="{y + 120}" font-size="11">{summary}</text></a>'
    for node in inputs + outputs:
        x, y = boundary_positions[node["id"]]
        is_input = node["id"].startswith("input")
        label = (
            f"输入 · {node['name']}"
            if is_input
            else f"输出 [{node['index']}] · {node['name']}"
        )
        value = node["value"]
        detail = value.get("dtype", value.get("kind", "元数据未记录")).removeprefix(
            "torch."
        )
        if "shape" in value:
            detail += " [" + ", ".join(value["shape"]) + "]"
        elif "expression" in value:
            detail += " " + value["expression"]
        elif "value" in value:
            detail += " " + repr(value["value"])
        color = colors["INPUT" if is_input else "OUTPUT"]
        lines = textwrap.wrap(detail, width=34)[:2]
        doc += (
            f'<a class="kd-node kd-boundary" href="{html.escape(link_prefix)}#kernel-boundary-{node["id"]}">'
            f"<title>{html.escape(json.dumps(node, ensure_ascii=False, indent=2))}</title>"
            f'<rect x="{x}" y="{y}" width="266" height="94" rx="9" fill="{("#edf8f2" if is_input else "#f5effa")}" stroke="{color}"/>'
            f'<text x="{x + 12}" y="{y + 24}" font-size="14" font-weight="650">{html.escape(textwrap.shorten(label, 38, placeholder="…"))}</text>'
        )
        for line_number, line in enumerate(lines):
            doc += f'<text x="{x + 12}" y="{y + 46 + line_number * 16}" font-size="12">{html.escape(line)}</text>'
        footer = value.get("buffer", "无 buffer 映射")
        if is_input and not any(
            e["source"] == node["id"] for e in graph["boundary_edges"]
        ):
            footer += " · 未记录消费"
        doc += f'<text x="{x + 12}" y="{y + 81}" font-size="11">{html.escape(footer)}</text></a>'
    if not nodes and not inputs and not outputs:
        doc += '<text x="28" y="55" font-size="16">未捕获最终 wrapper 调用，无法生成依赖图。</text>'
    return doc + "</svg>"


def dependency_html(graph):
    doc = (
        '<h2 id="kernel-dependencies">融合后的 kernel 依赖图</h2>'
        "<p>绿色节点是输入参数，紫色节点是按返回顺序排列的输出，白色 K 节点对应最终调用实例。"
        "输入/输出显示名称、shape 和 dtype，点击查看 stride、offset 等元数据。绿色/紫色边连接输入/输出；蓝色实线表示 RAW 数据依赖；"
        "橙色虚线表示同名 buffer 的 WAR/WAW 保守顺序约束。共用输入和相邻发射不会自动连线。"
        "点击节点跳到形成过程，点击边查看依赖证据；悬停可查看完整名称和 buffer。</p>"
        '<p><a href="kernel_dependencies.svg">独立 SVG</a> · <a href="kernel_dependencies.json">依赖数据 JSON</a></p>'
        '<label>缩放 <input id="kd-zoom" type="range" min="40" max="150" value="100" style="width:160px"></label> '
        '<label><input id="kd-order" type="checkbox" checked style="width:auto">显示 WAR/WAW</label>'
        '<div id="kd-view" style="overflow:auto;max-height:760px;border:1px solid #dce2ea;border-radius:8px;margin:12px 0">'
        + dependency_svg(graph)
        + '</div><script>document.querySelector("#kd-zoom").oninput=function(){const s=document.querySelector("#kd-view svg");'
        's.style.width=(s.viewBox.baseVal.width*this.value/100)+"px";s.style.height=(s.viewBox.baseVal.height*this.value/100)+"px";};'
        'document.querySelector("#kd-order").onchange=function(){document.querySelectorAll("#kd-view .kd-edge[data-kind=WAR],#kd-view .kd-edge[data-kind=WAW]")'
        '.forEach(e=>e.style.display=this.checked?"":"none");};</script>'
    )
    doc += "<details><summary>边的证据与追溯路径</summary><table><tr><th>依赖</th><th>Buffer / Nop 追溯</th></tr>"
    for index, edge in enumerate(graph["edges"]):
        doc += (
            f'<tr id="kernel-dependency-edge-{index}"><td>K{edge["source"]} → K{edge["target"]} · {edge["kind"]}</td>'
            "<td><pre>"
            + html.escape(json.dumps(edge["evidence"], ensure_ascii=False, indent=2))
            + "</pre></td></tr>"
        )
    doc += "</table></details>"
    doc += "<details><summary>输入参数、输出与边界依赖证据</summary><table><tr><th>边界</th><th>名称 / 元数据</th></tr>"
    for node in graph.get("inputs", []) + graph.get("outputs", []):
        label = node["id"] if node["id"].startswith("input") else node["path"]
        doc += (
            f'<tr id="kernel-boundary-{node["id"]}"><td>{html.escape(label)}</td><td><pre>'
            + html.escape(json.dumps(node, ensure_ascii=False, indent=2))
            + "</pre></td></tr>"
        )
    for index, edge in enumerate(graph.get("boundary_edges", [])):
        doc += (
            f'<tr id="kernel-boundary-edge-{index}"><td>{edge["source"]} → {edge["target"]}</td><td><pre>'
            + html.escape(json.dumps(edge["evidence"], ensure_ascii=False, indent=2))
            + "</pre></td></tr>"
        )
    doc += "</table></details>"
    for node in graph["nodes"]:
        for gap in node["gaps"]:
            doc += f'<p class="gap">K{node["id"]}：{html.escape(gap)}</p>'
    for node in graph.get("outputs", []):
        for gap in node["gaps"]:
            doc += f'<p class="gap">{node["id"]}：{html.escape(gap)}</p>'
    doc += "<details><summary>依赖图的范围与限制</summary><ul>"
    doc += "".join(
        "<li>" + html.escape(gap) + "</li>" for gap in graph["coverage_gaps"]
    )
    return doc + "</ul></details>"
