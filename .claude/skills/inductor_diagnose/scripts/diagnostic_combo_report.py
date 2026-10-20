import html
import json
import re


def names(nodes):
    return [n["name"] for n in nodes if isinstance(n, dict) and "name" in n]


def combo_analysis(events, compilation, config=None, log_lines=()):
    calls = {r["seq"]: r for r in events if r["event"] == "call"}
    result = {
        "filters": [],
        "topology": [],
        "chunks": [],
        "accepts": [],
        "benefit": [],
        "branches": [],
        "partition_stages": [],
        "argument_checks": [],
        "nodes": [],
        "configuration": {k: v for k, v in (config or {}).items() if "combo" in k},
        "native_memory_checks": [],
    }
    pattern = re.compile(
        r"ComboKernels memory-aware: (accepted|rejected) (\d+) nodes "
        r"\(peak delta ([+-]?\d+) bytes = ([+-]?[\d.]+)%\)"
    )
    for line_number, line in enumerate(log_lines, 1):
        match = pattern.search(line)
        if match:
            state, count, delta, percent = match.groups()
            result["native_memory_checks"].append(
                {
                    "state": state,
                    "node_count": int(count),
                    "peak_delta_bytes": int(delta),
                    "peak_delta_percent": float(percent),
                    "file": "compiler.log",
                    "line": line_number,
                    "text": line,
                    "certainty": "direct",
                    "gap": "原生日志没有成员名、绝对峰值及逐 buffer 模拟明细；不能只按节点数绑定到某个候选",
                }
            )
    entry_nodes = []
    for row in events:
        fn, event = row["function"], row["event"]
        call = calls.get(row.get("call_id"), row)
        if fn == "Scheduler.create_combo_kernel_nodes" and event == "call":
            entry_nodes = row.get("schedule", [])
        if event == "combo_filter":
            after = set(names(row["after"]))
            condition = row["condition"]
            for node in row["before"]:
                if node["name"] in after:
                    continue
                if "template_nodes" in condition:
                    reason = "template 节点被通用 combo 排除"
                elif "ForeachKernelSchedulerNode" in condition:
                    reason = "已有 foreach 容器被通用 combo 排除"
                elif "has_strict_sum" in condition:
                    reason = "strict sum 筛选排除，保留其既定 reduction block"
                elif "reduction_nodes" in condition:
                    reason = "pointwise-only 配置下排除 reduction"
                elif "indirect_nodes" in condition:
                    reason = "当前 benchmark/调优路径排除间接索引节点"
                else:
                    reason = "节点类型过滤：" + node["type"]
                result["filters"].append(
                    {
                        "node": node,
                        "reason": reason,
                        "event": row["seq"],
                        "condition": condition,
                        "file": row["file"],
                        "line": row["line"],
                        "certainty": "direct",
                    }
                )
        elif fn == "Scheduler._topological_sort_nodes" and event == "return":
            result["topology"].append(
                {"event": row["seq"], "groups": [names(g) for g in row["result"]]}
            )
        elif event == "combo_chunk":
            result["chunks"].append(
                {
                    "event": row["seq"],
                    "input": names(row["context"]["context_nodes"]),
                    "groups": [names(g) for g in row["after"][len(row["before"]) :]],
                    "max_num_nodes": row["context"].get("max_num_nodes"),
                    "contexts": row["context"].get("context_groups"),
                    "condition": row["condition"],
                }
            )
        elif fn.endswith("._register_accept") and event == "return":
            result["accepts"].append(
                {
                    "event": row["seq"],
                    "members": names(call["args"].get("accepted", [])),
                    "container": call["args"].get("combo_node"),
                }
            )
        elif fn == "Scheduler.speedup_by_combo_kernel" and event == "return":
            result["benefit"].append(
                {
                    "event": row["seq"],
                    "call_id": row.get("call_id"),
                    "nodes": names(call["args"].get("nodes", [])),
                    "result": row["result"],
                    "details": row.get("details"),
                    "file": row["file"],
                    "line": row["line"],
                }
            )
        elif event == "combo_branch":
            result["branches"].append(row)
            if fn == "ComboKernel._update_partition":
                ctx = row["context"]
                result["argument_checks"].append(
                    {
                        "event": row["seq"],
                        "condition": row["condition"],
                        "split": row["taken"],
                        "node": ctx.get("node_info"),
                        "current_count": ctx["partition_state"]["cur_count"],
                        "node_rw_count": ctx["node_rw_count"],
                    }
                )
        elif event in ("combo_partition_stage", "combo_yield"):
            result["partition_stages"].append(row)

    for item in result["benefit"]:
        direct_allow = [
            b
            for b in result["branches"]
            if b.get("parent_id") == item["call_id"]
            and b["condition"] == "not config.benchmark_combo_kernel"
            and b["taken"]
        ]
        item["mode"] = (
            "配置关闭 combo benchmark，直接放行"
            if direct_allow
            else "查看返回分支及计时证据"
        )
        item["branch_events"] = [b["seq"] for b in direct_allow]

    for node in entry_nodes:
        node_name = node["name"]
        filters = [r for r in result["filters"] if r["node"]["name"] == node_name]
        accepted = [r for r in result["accepts"] if node_name in r["members"]]
        singletons = [
            r
            for r in result["branches"]
            if r["function"] == "Scheduler.create_combo_kernel_nodes"
            and r["taken"]
            and r["condition"] == "len(members) < 2"
            and node_name in names(r["context"].get("members", []))
        ]
        levels = [
            (g, i)
            for g in result["topology"]
            for i, group in enumerate(g["groups"])
            if node_name in group
        ]
        if accepted:
            state, reason, evidence = (
                "accepted",
                "已建立 foreach 容器",
                [r["event"] for r in accepted],
            )
        elif filters:
            state, reason, evidence = (
                "filtered",
                filters[0]["reason"],
                [r["event"] for r in filters],
            )
        elif singletons:
            state, reason, evidence = (
                "singleton",
                "当前分组筛选后不足两个节点，不建立 combo",
                [r["seq"] for r in singletons],
            )
        else:
            state, reason, evidence = "unknown", "未捕获到最终 combo 处置原因", []
        ops = {m["name"] for m in node["members"]}
        wrappers = [
            c["name"]
            for c in compilation.get("calls", [])
            if ops
            & {
                m["operation"]
                for n in compilation.get("call_schedules", {}).get(
                    c.get("provenance_key"), []
                )
                for m in n["members"]
            }
        ]
        result["nodes"].append(
            {
                "name": node_name,
                "type": node["type"],
                "state": state,
                "reason": reason,
                "events": evidence,
                "topology_levels": [i for _, i in levels],
                "wrappers": wrappers,
            }
        )
    return result


def combo_html(data, event_link):
    out = "<h3>逐节点 combo 选择结果</h3><p>direct 过滤原因来自筛选列表实际变化和执行的源码语句。拓扑层、上下文及切块使用实际运行结果；不同分组不代表理论上绝无其他调度方案。</p>"
    out += "<table><tr><th>节点 / 类型 / 拓扑层</th><th>选择结果</th><th>证据</th></tr>"
    for n in data["nodes"]:
        out += (
            "<tr><td>"
            + html.escape(
                n["name"] + " / " + n["type"] + " / " + str(n["topology_levels"])
            )
            + "</td><td>"
            + html.escape(n["state"] + ": " + n["reason"])
            + "</td><td>"
            + ", ".join(event_link(e) for e in n["events"])
            + "</td></tr>"
        )
    out += "</table><h3>执行过的筛选规则</h3>"
    for r in data["filters"]:
        out += (
            "<p>"
            + html.escape(r["node"]["name"] + ": " + r["reason"])
            + " "
            + event_link(r["event"])
            + "</p><pre>"
            + html.escape(r["condition"])
            + "</pre>"
        )
    out += "<h3>实际拓扑分组、上下文与数量切块</h3>"
    for r in data["topology"]:
        out += (
            "<p>"
            + event_link(r["event"])
            + "</p><pre>"
            + html.escape(json.dumps(r["groups"], ensure_ascii=False, indent=2))
            + "</pre>"
        )
    for r in data["chunks"]:
        out += (
            "<p>上限 "
            + str(r["max_num_nodes"])
            + "，输入 "
            + str(len(r["input"]))
            + " 个节点，切块大小 "
            + str([len(g) for g in r["groups"]])
            + " "
            + event_link(r["event"])
            + "</p><details><summary>上下文与分组证据</summary><pre>"
            + html.escape(json.dumps(r, ensure_ascii=False, indent=2))
            + "</pre></details>"
        )
    out += "<h3>收益检查与实际接受</h3>"
    for r in data["benefit"]:
        out += (
            "<p>"
            + html.escape(str(r["nodes"]) + ": " + str(r["result"]) + "；" + r["mode"])
            + " "
            + event_link(r["event"])
            + " "
            + ", ".join(event_link(e) for e in r["branch_events"])
            + "</p>"
        )
    for r in data["accepts"]:
        out += (
            "<p>实际接受 "
            + str(len(r["members"]))
            + " 个 subkernel："
            + event_link(r["event"])
            + "</p>"
        )
    cfg = data.get("configuration", {})
    out += (
        "<h3>距离窗口与峰值内存门槛</h3><p>配置：<code>"
        + html.escape(
            json.dumps(
                {
                    k: cfg.get(k)
                    for k in (
                        "combo_kernel_max_distance",
                        "combo_kernel_peak_memory_increase_gb",
                        "combo_kernel_peak_memory_pct_threshold",
                    )
                },
                ensure_ascii=False,
            )
        )
        + "</code>。</p>"
    )
    for r in data.get("native_memory_checks", []):
        out += (
            '<p>原生日志 <a href="compiler.log">compiler.log:'
            + str(r["line"])
            + "</a>：</p><pre>"
            + html.escape(r["text"])
            + '</pre><p class="muted">'
            + html.escape(r["gap"])
            + "</p>"
        )
    out += (
        "<h3>Codegen 参数预算检查</h3><p>当前上限为 "
        + str(cfg.get("combo_kernel_max_num_args"))
        + "。这里统计调度节点的读写依赖数，不能当作最终去重后的 kernel ABI 参数个数。</p><table><tr><th>节点</th><th>当前计数 + 新节点读写数</th><th>是否触发预算拆分</th></tr>"
    )
    for r in data["argument_checks"]:
        out += (
            "<tr><td>"
            + html.escape(r["node"]["name"])
            + "</td><td>"
            + str(r["current_count"])
            + " + "
            + str(r["node_rw_count"])
            + "</td><td>"
            + str(r["split"])
            + " "
            + event_link(r["event"])
            + "</td></tr>"
        )
    out += (
        "</table><details><summary>其他执行分支、距离窗口与 tiling 分类</summary><pre>"
        + html.escape(
            json.dumps(
                {
                    "branches": data["branches"],
                    "partition_stages": data["partition_stages"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        + "</pre></details>"
    )
    return out
