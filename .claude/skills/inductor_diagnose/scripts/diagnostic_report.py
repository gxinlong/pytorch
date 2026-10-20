import argparse
import html
import json
from collections import Counter, defaultdict
from pathlib import Path

from diagnostic_combo_report import combo_analysis, combo_html


def read(path, default=None):
    return json.loads(Path(path).read_text()) if Path(path).exists() else default


def members(node):
    return tuple(
        sorted(m.get("name", m.get("operation")) for m in node.get("members", []))
    )


def pair_record(pair, nodes=None):
    return {
        "pair": list(pair),
        "nodes": nodes or [],
        "candidates": [],
        "checks": [],
        "rejections": [],
        "profitability": [],
        "callbacks": [],
        "merges": [],
        "evidence": [],
    }


def classify(record, all_merges):
    if record["merges"]:
        return "merged", "已实际合并", "direct", []
    failed = [r for r in record["callbacks"] if r["result"] is False]
    failed += [
        r
        for r in record["profitability"]
        if isinstance(r["result"], dict) and r["result"].get("should_fuse") is False
    ]
    if failed:
        measured = any(
            "min_ms_fused" in r.get("details", {}) or "ms_fused" in r.get("details", {})
            for r in failed
        )
        return (
            "profitability_rejected",
            (
                "收益判断拒绝，捕获到融合计时"
                if measured
                else "收益判断拒绝，未捕获融合版本实测证据"
            ),
            "direct",
            (
                []
                if measured
                else ["收益预筛、资源限制或其他返回分支需结合源码；不能认定实测变慢"]
            ),
        )
    pending = [
        r
        for r in record["profitability"]
        if isinstance(r["result"], dict) and r["result"].get("pending")
    ]
    if pending:
        own = set(record["pair"])
        competitors = [
            r
            for r in all_merges
            if r["seq"] > pending[-1]["seq"]
            and own & set(r.get("pair") or [])
            and set(r.get("pair") or []) != own
        ]
        if competitors:
            record["competing_merges"] = [r["seq"] for r in competitors]
            return (
                "pending_competition",
                "已提交但未合并；共享节点被其他融合接收",
                "inference",
                [
                    "竞争归因由候选、实际合并及源码推导；没有捕获逐候选移除事件",
                    "未执行收益回调不等于 benchmark 失败",
                ],
            )
        return (
            "pending_unresolved",
            "已提交，未捕获最终接受或拒绝",
            "unknown",
            ["缺少 pending 候选最终处置证据"],
        )
    if record["checks"] and record["checks"][-1]["result"] is False:
        check = record["checks"][-1]
        reasons = [
            r
            for r in record["rejections"]
            if r.get("check_call_id") == check.get("call_id")
        ]
        reason = reasons[-1]["reason"] if reasons else "没有记录该次检查的拒绝原因"
        return (
            "check_rejected",
            reason,
            "direct" if reasons else "unknown",
            ([] if reasons else ["检查返回 false，但未捕获对应 WhyNoFuse 原因"]),
        )
    if record["candidates"]:
        return (
            "candidate_uncommitted",
            "进入候选，但没有实际合并记录",
            "unknown",
            ["候选最终处置未捕获"],
        )
    if record["rejections"]:
        return "check_rejected", record["rejections"][-1]["reason"], "direct", []
    return (
        "not_observed",
        "没有观察到该候选的决策调用",
        "unknown",
        ["类型过滤、候选构造、搜索窗口或状态变化未逐项记录；不能推断为性能失败"],
    )


def analyze_events(events, compilation):
    calls = {r["seq"]: r for r in events if r["event"] == "call"}
    nodes, trees, opportunities = {}, {}, {}
    merge_events, partitions, choices, combos, lowering = [], [], [], [], []
    initial_foreach = []
    for row in events:
        function = row["function"]
        call = calls.get(row.get("call_id"), row)
        args = call.get("args", {})
        pair = tuple(row.get("pair") or call.get("pair") or [])
        if len(pair) == 2:
            record = opportunities.setdefault(pair, pair_record(pair))
            if not record["nodes"] and all(
                isinstance(args.get(k), dict) for k in ("node1", "node2")
            ):
                record["nodes"] = [args["node1"], args["node2"]]
        else:
            record = None
        if (
            function == "WhyNoFuse.__call__"
            and row["event"] == "call"
            and record is not None
        ):
            parent = calls.get(row.get("parent_id"), {})
            while parent and parent["function"] != "Scheduler.can_fuse":
                parent = calls.get(parent.get("parent_id"), {})
            reason, arguments = args.get("reason"), args.get("args", [])
            if arguments:
                try:
                    reason = reason % tuple(arguments)
                except (TypeError, ValueError):
                    pass
            record["rejections"].append(
                {
                    "seq": row["seq"],
                    "reason": reason,
                    "arguments": arguments,
                    "check_call_id": parent.get("seq"),
                }
            )
            record["evidence"].append(row["seq"])
        if row["event"] != "return":
            continue
        result = row.get("result")
        if function == "GraphLowering.run_node":
            lowering.append({"seq": row["seq"], "fx": args.get("n"), "value": result})
        if (
            function == "Scheduler.create_scheduler_node"
            and isinstance(result, dict)
            and "members" in result
        ):
            key = members(result)
            if key not in trees:
                trees[key] = {
                    "stage": "initial_scheduler",
                    "event": row["seq"],
                    "node": result,
                    "evidence_gap": "成员在进入 Scheduler 前已合并；未记录完整 lowering 内联与物化决策",
                }
            for member in result["members"]:
                nodes.setdefault(member["name"], result)
        if function == "Scheduler.create_foreach_nodes":
            for node in row.get("schedule", []):
                if not node.get("foreach"):
                    continue
                key = members(node)
                tree = {
                    "stage": "initial_foreach",
                    "event": row["seq"],
                    "node": node,
                    "members": list(key),
                    "children": [
                        trees.get(
                            (name,),
                            {"stage": "unobserved_formation", "members": [name]},
                        )
                        for name in key
                    ],
                }
                trees[key] = tree
                initial_foreach.append(tree)
        if function == "Scheduler.get_possible_fusions" and isinstance(result, list):
            for value in result:
                if (
                    isinstance(value, list)
                    and len(value) == 2
                    and all(isinstance(n, dict) and "name" in n for n in value)
                ):
                    key = tuple(n["name"] for n in value)
                    item = opportunities.setdefault(key, pair_record(key, value))
                    item["candidates"].append(row["seq"])
                    item["evidence"].append(row["seq"])
        if record is not None:
            item = {
                "seq": row["seq"],
                "call_id": row.get("call_id"),
                "result": result,
                "details": row.get("details", {}),
                "file": row["file"],
                "line": row["line"],
            }
            if function == "Scheduler.can_fuse":
                record["checks"].append(item)
                record["evidence"].append(row["seq"])
            elif function == "Scheduler.speedup_by_fusion":
                record["profitability"].append(item)
                record["evidence"].append(row["seq"])
            elif function.endswith(".benchmark_when_ready"):
                record["callbacks"].append(item)
                record["evidence"].append(row["seq"])
        if (
            function == "Scheduler.fuse_two_nodes"
            and isinstance(result, dict)
            and "members" in result
        ):
            merge_events.append(row)
            if record is not None:
                record["merges"].append(row["seq"])
                record["evidence"].append(row["seq"])
            left, right = args.get("node1", {}), args.get("node2", {})
            stage = (
                "template_epilogue"
                if left.get("template")
                else (
                    "template_prologue" if right.get("template") else "ordinary_fusion"
                )
            )
            trees[members(result)] = {
                "stage": stage,
                "event": row["seq"],
                "node": result,
                "children": [
                    trees.get(members(n), {"stage": "unobserved_formation", "node": n})
                    for n in (left, right)
                ],
            }
        if function.endswith("._register_accept"):
            accepted = args.get("accepted", [])
            if isinstance(accepted, list):
                key = tuple(
                    sorted(
                        m for n in accepted if isinstance(n, dict) for m in members(n)
                    )
                )
                if key:
                    tree = {
                        "stage": "generic_combo",
                        "event": row["seq"],
                        "members": list(key),
                        "children": [
                            trees.get(
                                members(n), {"stage": "unobserved_formation", "node": n}
                            )
                            for n in accepted
                        ],
                    }
                    trees[key] = tree
                    combos.append(tree)
        if function == "Scheduler.create_combo_kernel_nodes":
            for node in row.get("schedule", []):
                tree = trees.get(members(node))
                if node.get("foreach") and tree and tree["stage"] == "generic_combo":
                    tree["node"] = node
                    tree["node_event"] = row["seq"]
        if function == "ComboKernel.horizontal_partition":
            partitions.append(
                {
                    "seq": row["seq"],
                    "groups": [
                        [m for n in group for m in members(n)] for group in result
                    ]
                    if isinstance(result, list)
                    else [],
                    "raw_result": result,
                }
            )
        if function.endswith(
            (
                "get_min_choice",
                "choice_timings",
                "finalize_as_triton_caller",
                "_has_layout_conflict_for_template",
                "_replace_node",
            )
        ):
            choices.append(
                {
                    "seq": row["seq"],
                    "function": function,
                    "buffer": call.get("buffer"),
                    "pair": list(pair),
                    "args": args,
                    "result": result,
                    "details": row.get("details", {}),
                }
            )

    initial = list(nodes.values())
    initial = list({n["name"]: n for n in initial}.values())
    for left in initial:
        for right in initial:
            if left["name"] == right["name"]:
                continue
            if set(left.get("write_buffers", [])) & set(right.get("read_buffers", [])):
                pair = (left["name"], right["name"])
                opportunities.setdefault(pair, pair_record(pair, [left, right]))[
                    "discovery"
                ] = "initial_IR_data_dependency"

    for record in opportunities.values():
        status, reason, certainty, gaps = classify(record, merge_events)
        record.update(
            status=status, reason=reason, certainty=certainty, evidence_gaps=gaps
        )
        timed = any(
            "min_ms_fused" in r.get("details", {}) or "ms_fused" in r.get("details", {})
            for r in record["callbacks"] + record["profitability"]
        )
        direct_allow = any(
            isinstance(r["result"], dict) and r["result"].get("should_fuse") is True
            for r in record["profitability"]
        )
        record["benefit_evidence"] = (
            "captured_fusion_timing"
            if timed
            else "direct_allow_without_timing"
            if direct_allow
            else "submitted_without_timing_callback"
            if record["profitability"]
            else "no_profitability_call"
        )
        record["evidence"] = sorted(set(record["evidence"]))
        first, second = record["nodes"] if len(record["nodes"]) == 2 else ({}, {})
        record["route"] = (
            "template_epilogue"
            if first.get("template") and not second.get("template")
            else (
                "template_prologue"
                if second.get("template") and not first.get("template")
                else (
                    "template_pair"
                    if first.get("template") and second.get("template")
                    else "ordinary_fusion"
                )
            )
        )

    filters = []
    for row in events:
        if row["event"] != "return" or not row["function"].endswith(
            ".combinable_nodes"
        ):
            continue
        before = calls.get(row.get("call_id"), {}).get("args", {}).get("nodes", [])
        if not isinstance(before, list) or not isinstance(row["result"], list):
            continue
        after = {m for n in row["result"] for m in members(n)}
        for node in before:
            if not set(members(node)) <= after:
                filters.append(
                    {
                        "seq": row["seq"],
                        "node": node,
                        "status": "combo_filtered",
                        "reason": "combinable_nodes 的实际返回值排除了该节点",
                        "evidence_gap": "函数内部每个过滤条件没有独立记录；节点类型与源码可辅助解释",
                    }
                )

    kernels = []
    for call in compilation.get("calls", []):
        schedule = compilation.get("call_schedules", {}).get(
            call.get("provenance_key"), []
        )
        ops = tuple(sorted({m["operation"] for n in schedule for m in n["members"]}))
        tree = trees.get(ops)
        related_partitions = [
            r for r in partitions if any(set(g) == set(ops) for g in r["groups"])
        ]
        parents = [
            group
            for group in combos + initial_foreach
            if ops and set(ops) < set(group["members"])
        ]
        if tree is None and parents and related_partitions:
            parent = min(parents, key=lambda g: len(g["members"]))
            tree = {
                "stage": "codegen_partition",
                "event": related_partitions[-1]["seq"],
                "members": list(ops),
                "children": [parent],
            }
        if tree is None:
            tree = {
                "stage": "final_codegen_members",
                "members": list(ops),
                "children": [
                    trees.get((n,), {"stage": "unobserved_formation", "members": [n]})
                    for n in ops
                ],
            }
        buffers = {
            b for n in schedule for m in n["members"] for b in m.get("buffers", [])
        }
        related_choices = [
            r
            for r in choices
            if r.get("buffer") in buffers
            or set(members(r.get("args", {}).get("node", {}))) & set(ops)
            or set(r.get("pair", [])) & set(ops)
        ]
        origins = sorted(
            {o for n in schedule for m in n["members"] for o in m["origins"]}
        )
        kernel = {
            "call": call,
            "members": list(ops),
            "fx_origins": origins,
            "formation": tree,
            "choice_events": related_choices,
            "partitions": related_partitions,
            "generated_code": compilation.get("kernels", {}).get(call["name"]),
            "evidence_gaps": [],
        }
        if not ops:
            kernel["evidence_gaps"].append("缺少 call 到调度节点的映射")
        if not kernel["generated_code"]:
            kernel["evidence_gaps"].append(
                "extern 调用：库内部 GPU kernel 的拆分、融合及算法过程不由 Inductor 记录"
            )
        kernel["opportunity_pairs"] = [
            r["pair"]
            for r in opportunities.values()
            if any(set(members(n)) & set(ops) for n in r["nodes"])
        ]
        kernels.append(kernel)
    for record in opportunities.values():
        left, right = record["nodes"] if len(record["nodes"]) == 2 else ({}, {})
        both = set(members(left)) | set(members(right))
        record["final_wrapper_indices"] = [
            i for i, k in enumerate(kernels) if both & set(k["members"])
        ]
        record["final_colocation"] = [
            i for i, k in enumerate(kernels) if both and both <= set(k["members"])
        ]
    return {
        "initial_nodes": initial,
        "kernels": kernels,
        "opportunities": list(opportunities.values()),
        "combo_filters": filters,
        "partitions": partitions,
        "choice_events": choices,
        "lowering_results": lowering,
        "merge_count": len(merge_events),
        "combo_count": len(combos),
        "initial_foreach_count": len(initial_foreach),
    }


def parity(current, historical):
    if not historical or not current:
        return {"available": False}
    current_identity = current["graph"]["identity"]
    historical_identity = historical["graph"]["identity"]
    if isinstance(current_identity, dict):
        current_identity = current_identity.get("sha256")
    if isinstance(historical_identity, dict):
        historical_identity = historical_identity.get("sha256")
    comparison = {
        "graph_identity": current_identity is not None
        and current_identity == historical_identity,
        "scheduler": current["scheduler"] == historical["scheduler"],
        "calls": current["calls"] == historical["calls"],
        "call_schedules": current["call_schedules"] == historical["call_schedules"],
        "kernel_sources": {
            k: v.get("normalized_sha256") for k, v in current["kernels"].items()
        }
        == {k: v.get("normalized_sha256") for k, v in historical["kernels"].items()},
    }
    return {
        "available": True,
        "equal": comparison,
        "strict_match": all(comparison.values()),
        "historical_wrapper_calls": len(historical["calls"]),
        "current_wrapper_calls": len(current["calls"]),
        "historical_groups": [
            {
                "name": c["name"],
                "members": [
                    m["operation"]
                    for n in historical["call_schedules"].get(
                        c.get("provenance_key"), []
                    )
                    for m in n["members"]
                ],
            }
            for c in historical["calls"]
        ],
        "current_groups": [
            {
                "name": c["name"],
                "members": [
                    m["operation"]
                    for n in current["call_schedules"].get(c.get("provenance_key"), [])
                    for m in n["members"]
                ],
            }
            for c in current["calls"]
        ],
    }


def event_link(seq):
    return f'<a href="events/{(seq - 1) // 500:04d}.html#e{seq}">事件 {seq}</a>'


def tree_html(tree):
    labels = {
        "initial_scheduler": "进入 Scheduler 时已有的表达式",
        "ordinary_fusion": "普通融合",
        "template_epilogue": "Template epilogue",
        "template_prologue": "Template prologue",
        "generic_combo": "通用 combo",
        "initial_foreach": "普通融合前建立的 foreach 容器",
        "final_codegen_members": "最终 codegen 成员",
        "codegen_partition": "Codegen 从 foreach/combo 容器中划出该分区",
        "unobserved_formation": "缺少形成事件",
    }
    node = tree.get("node", {})
    text = (
        labels.get(tree["stage"], tree["stage"])
        + "："
        + (node.get("name") or ", ".join(tree.get("members", [])))
    )
    if node.get("type"):
        text += " [" + node["type"] + "]"
    detail = ""
    if tree.get("event"):
        detail += " " + event_link(tree["event"])
    if tree.get("node_event"):
        detail += (
            "<div>组合后的实际 Scheduler 容器："
            + event_link(tree["node_event"])
            + "</div>"
        )
    if tree["stage"] == "generic_combo":
        detail += '<div class="muted">横向 combo 保留各 subkernel；每个 subkernel 内部可以已有普通融合。实际生成几个 kernel 还需看 codegen 分区。</div>'
    if tree["stage"] == "initial_foreach":
        detail += '<div class="muted">这是 create_foreach_nodes 阶段实际建立的容器，早于普通融合和通用 combo；不能算作后续 combo 收益判断的接受结果。</div>'
    if node.get("members"):
        detail += (
            '<div class="muted">FX: '
            + html.escape(
                ", ".join(sorted({o for n in node["members"] for o in n["origins"]}))
            )
            + "</div>"
        )
    if tree.get("evidence_gap"):
        detail += '<div class="gap">' + html.escape(tree["evidence_gap"]) + "</div>"
    children = "".join(
        "<li>" + tree_html(child) + "</li>" for child in tree.get("children", [])
    )
    return (
        html.escape(text) + detail + ("<ul>" + children + "</ul>" if children else "")
    )


STYLE = """body{font:15px/1.65 system-ui,sans-serif;color:#202c3b;max-width:1400px;margin:32px auto;padding:0 24px}h1{font-size:27px}h2{font-size:22px;margin-top:32px}h3{font-size:17px}a{color:#185db1}table{border-collapse:collapse;width:100%;margin:16px 0}td,th{border:1px solid #dce2ea;padding:9px;text-align:left;vertical-align:top}th{background:#eef3f8}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f6f9;padding:16px;font-size:12px}code{overflow-wrap:anywhere}.gap{color:#8b5100;background:#fff5df;padding:5px 8px;margin:6px 0}.muted{color:#596b7e;font-size:13px}.card{border:1px solid #dce2ea;border-radius:8px;padding:18px;margin:16px 0}input{padding:10px;width:70%;font-size:15px}summary{cursor:pointer}nav a{margin-right:18px}ul{padding-left:24px}.ok{color:#176642}.direct{color:#176642}.inference{color:#815900}.unknown{color:#a23b2c}"""


def build_report(run):
    run = Path(run)
    if not (run / "run_manifest.json").is_file():
        raise ValueError("Not a diagnostic run: missing run_manifest.json")
    if not any(
        (run / name).exists()
        for name in (
            "compile_result.json",
            "failure.json",
            "process_failure.json",
            "observations/decisions.jsonl",
        )
    ):
        raise ValueError("No compilation evidence or failure record in this run")
    comp = read(run / "diagnostics/compilation.json", {})
    event_path = run / "observations/decisions.jsonl"
    events = (
        [json.loads(line) for line in event_path.open()] if event_path.exists() else []
    )
    result = analyze_events(events, comp)
    effective = read(run / "effective_config.json", {})
    log_lines = (
        (run / "compiler.log").read_text(errors="replace").splitlines()
        if (run / "compiler.log").exists()
        else []
    )
    result["combo_decisions"] = combo_analysis(events, comp, effective, log_lines)
    for item in result["combo_filters"]:
        rules = [
            r
            for r in result["combo_decisions"]["filters"]
            if r["node"]["name"] == item["node"]["name"]
        ]
        if rules:
            item.update(
                reason=rules[0]["reason"],
                rule_events=[r["event"] for r in rules],
                evidence_gap="",
            )
    manifest = read(run / "run_manifest.json", {})
    old = run / "historical_compilation.json"
    if not old.exists():
        old = Path(manifest.get("historical_diagnostics", "/nonexistent"))
    result["parity"] = parity(comp, read(old))
    result["previous_diagnostic_parity"] = parity(
        comp, read(run / "comparison_compilation.json")
    )
    result["identity"] = read(run / "identity.json", {})
    result["observer"] = read(run / "observations/observer_status.json", {})
    result["validation"] = read(run / "validation.json", {})
    result["compile_result"] = read(run / "compile_result.json", {})
    result["run_status"] = read(run / "status.json", {})
    result["process_failure"] = read(run / "process_failure.json", {})
    result["tlparse_status"] = read(run / "tlparse_result.json", {})
    result["native_artifacts"] = {
        "graph_transforms": [
            str(p.relative_to(run))
            for p in sorted((run / "graph_transforms").glob("*.dot"))
        ],
        "debug_files": [
            str(p.relative_to(run))
            for p in sorted((run / "debug").rglob("*"))
            if p.is_file() and p.suffix in (".txt", ".log", ".py", ".cpp", ".json")
        ],
        "trace_files": [
            str(p.relative_to(run)) for p in sorted((run / "trace").glob("*.log"))
        ],
    }
    result["native_log_counts"] = {
        name: sum(marker in line for line in log_lines)
        for name, marker in {
            "fusion": "[__fusion]",
            "rejections": "cannot fuse",
            "merge_intents": "fusing op",
            "layout_conflicts": "Layout conflict detected",
        }.items()
    }
    result["coverage_gaps"] = [
        "未执行路径和未进入候选的精确原因不能由缺少日志反推。",
        "观察器仅覆盖主编译线程；后台编译和计时内部流程不完整。",
        "GraphTransformObserver 仅记录已接入 pass 的可观察节点变化，不覆盖所有 pattern 匹配失败或纯属性改写。",
        "没有为每个 lowering 内联、realize、布局物化决定记录原因。",
        "融合检查复合条件只记录实际返回位置，未对每个子条件重新求值。",
        "extern 调用可能对应多个 GPU kernel；本报告不声称记录了库内部形成过程。",
        "诊断记录有开销，编译器调优计时不是正式子图性能结果。",
        "机会表覆盖观察到的节点对和初始 IR 数据依赖，不穷举全部理论上的横向组合。check_rejected 也可能是 can_fuse 中的启发式预筛，并不都属于语义合法性失败。",
    ]
    if not read(run / "tlparse_result.json", {}).get("available"):
        result["coverage_gaps"].append(
            "tlparse 尚未执行或不可用；原始结构化 trace 保留在 trace/。"
        )
    for target in result["observer"].get("missing_combo_targets", []):
        result["coverage_gaps"].append("当前源码缺少已适配的 combo 观察目标：" + target)
    for key in ("unfinished_calls", "combo_pending_statements"):
        if result["observer"].get(key):
            result["coverage_gaps"].append(
                f"观察未完整结束：{key}={result['observer'][key]}"
            )
    for item in result["tlparse_status"].get("traces", []):
        if item["returncode"]:
            result["coverage_gaps"].append(
                f"tlparse 解析失败，退出码 {item['returncode']}；查看 tlparse 日志。"
            )
        index = Path(item["index"])
        local_index = run / "tlparse" / index.parent.name / "index.html"
        if not local_index.is_file():
            result["coverage_gaps"].append(
                "tlparse HTML 入口不存在：" + str(local_index.relative_to(run))
            )
    if result["tlparse_status"].get("available") and not result["tlparse_status"].get(
        "traces"
    ):
        result["coverage_gaps"].append("tlparse 可执行文件存在，但没有已解析的 trace。")
    if not result["kernels"]:
        result["coverage_gaps"].append(
            "没有取得最终 wrapper 调用映射；不能声称已覆盖每个 kernel。"
        )
    (run / "analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    batches = defaultdict(list)
    for event in events:
        batches[(event["seq"] - 1) // 500].append(event)
    (run / "events").mkdir(exist_ok=True)
    for batch, rows in batches.items():
        body = '<a href="../report.html">返回报告</a><h1>决策事件</h1>'
        for event in rows:
            seq = event["seq"]
            source = "../source_snapshot/" + event["file"]
            body += f'<details id="e{seq}"><summary>#{seq} {html.escape(event["event"])} {html.escape(event["function"])} {html.escape(str(event.get("pair") or ""))}</summary><a href="{html.escape(source)}">源码 {html.escape(event["file"])}:{event["line"]}</a><pre>{html.escape(json.dumps(event, ensure_ascii=False, indent=2))}</pre></details>'
        body += "<script>if(location.hash){const el=document.getElementById(location.hash.slice(1));if(el)el.open=true;}</script>"
        (run / "events" / f"{batch:04d}.html").write_text(
            '<!doctype html><meta charset="utf-8"><style>' + STYLE + "</style>" + body
        )

    counts = Counter(r["status"] for r in result["opportunities"])
    doc = (
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Inductor 编译决策报告</title><style>'
        + STYLE
        + "</style><body>"
    )
    doc += '<h1>Inductor：kernel 形成过程与融合停止位置</h1><nav><a href="#identity">身份与复现</a><a href="#key">关键决策</a><a href="#kernels">逐 kernel</a><a href="#lowering">Lowering</a><a href="#opportunities">融合机会</a><a href="#combo">Combo</a><a href="#native">原生工具</a><a href="#gaps">证据缺口</a></nav>'
    doc += (
        "<p>对象："
        + html.escape(manifest.get("model_name", "ExportedProgram"))
        + "。每个 kernel 项目对应一个 Inductor wrapper 调用，extern 库内部 GPU kernel 未展开。输入、动态约束和配置见运行清单。</p>"
    )
    doc += (
        "<p>运行状态：<code>"
        + html.escape(result["run_status"].get("status", "unknown"))
        + "</code>；正确性：<code>"
        + html.escape(str(result["validation"].get("passed", "unknown")))
        + "</code>。</p>"
    )
    if result["process_failure"]:
        doc += (
            '<p class="gap">worker 进程失败，以下内容是可用的部分记录：</p><pre>'
            + html.escape(
                json.dumps(result["process_failure"], ensure_ascii=False, indent=2)
            )
            + "</pre>"
        )
    if manifest.get("config_overrides"):
        doc += (
            "<p>本次显式参数修改：<code>"
            + html.escape(json.dumps(manifest["config_overrides"]))
            + "</code>。</p>"
        )
    doc += f"<p><b>{len(result['kernels'])} 个 wrapper 调用 · {result['merge_count']} 次实际融合返回 · {result['combo_count']} 个已接受 combo · {len(events)} 条补充事件</b></p>"
    doc += '<p>证据标签：<span class="direct">direct＝实际调用/返回或原生日志</span>；<span class="inference">inference＝结合状态和源码推导</span>；<span class="unknown">unknown＝缺少决策证据</span>。</p>'
    doc += '<h2 id="identity">身份、正确性与历史结果核对</h2>'
    if result["parity"].get("available") and not result["parity"].get("strict_match"):
        doc += '<p class="gap">本次产物没有通过历史产物的全部严格一致性检查。本报告解释本次编译；不能把它的决策当作历史编译的决策重放。</p>'
    doc += '<p>源码身份和允许差异见 <a href="run_manifest.json">运行清单</a>、<a href="identity.json">实际环境</a>；正确性见 <a href="validation.json">校验结果</a>。</p>'
    doc += (
        "<pre>"
        + html.escape(
            json.dumps(
                {
                    "source_commit": result["identity"].get("commit"),
                    "registered_source_changes": manifest.get(
                        "registered_source_changes"
                    ),
                    "parity": result["parity"].get("equal"),
                    "correctness_passed": result["validation"].get("passed"),
                    "observer_errors": result["observer"].get("errors"),
                    "stops": dict(counts),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        + "</pre>"
    )
    doc += (
        "<details><summary>历史与本次完整分组对比</summary><pre>"
        + html.escape(json.dumps(result["parity"], ensure_ascii=False, indent=2))
        + "</pre></details>"
    )
    if result["previous_diagnostic_parity"].get("available"):
        doc += (
            "<details><summary>上一次诊断编译与本次分组对比</summary><pre>"
            + html.escape(
                json.dumps(
                    result["previous_diagnostic_parity"], ensure_ascii=False, indent=2
                )
            )
            + "</pre></details>"
        )
        previous = result["previous_diagnostic_parity"]
        doc += (
            "<p>上次 → 本次 wrapper 调用数："
            + str(previous["historical_wrapper_calls"])
            + " → "
            + str(previous["current_wrapper_calls"])
            + "。下表按本次成员集合对照上次分组，kernel 名称可能随重新生成而变化。</p><table><tr><th>本次调用 / 成员</th><th>上次相交的调用 / 成员</th></tr>"
        )
        for current in previous["current_groups"]:
            old_groups = [
                g
                for g in previous["historical_groups"]
                if set(g["members"]) & set(current["members"])
            ]
            doc += (
                "<tr><td><code>"
                + html.escape(current["name"])
                + "</code><br>"
                + html.escape(str(current["members"]))
                + "</td><td>"
            )
            doc += (
                "<br>".join(
                    html.escape(g["name"] + ": " + str(g["members"]))
                    for g in old_groups
                )
                + "</td></tr>"
            )
        doc += '</table><p class="gap">数量分组变化与重新调优产生的 template 选择变化需分别看待；调用数减少不能直接证明端到端加速，也不能全部归因于节点上限。</p>'
    doc += '<h2 id="key">关键决策：候选、检查、收益与落实</h2><p>下表优先列出实际进入收益判断的 template 机会，以及跨越最终调用边界的初始依赖。完整表还保留各轮被预筛的节点对。候选曾被拒绝不等于最终成员仍然分离；同一 combo 内也不等于子 kernel 间已做纵向融合。</p><table><tr><th>机会</th><th>本次结论</th><th>实测记录（编译器内部，毫秒）</th></tr>'
    for record in result["opportunities"]:
        important = record["profitability"] and record["route"] != "ordinary_fusion"
        boundary = (
            record.get("discovery")
            and len(record["final_wrapper_indices"]) > 1
            and not record["final_colocation"]
        )
        if not (important or boundary):
            continue
        detail = (record["callbacks"] or record["profitability"] or [{}])[-1].get(
            "details", {}
        )
        timing = {
            k: v
            for k, v in detail.items()
            if k in ("ms1", "ms2", "min_ms_fused", "ms_fused_choice")
        }
        doc += (
            "<tr><td>"
            + html.escape(" → ".join(record["pair"]))
            + "<br>"
            + html.escape(record["route"])
            + "</td><td>"
            + html.escape(record["reason"])
            + "<br>"
            + ", ".join(event_link(n) for n in record["evidence"][-3:])
            + "</td><td><code>"
            + html.escape(json.dumps(timing, ensure_ascii=False))
            + "</code></td></tr>"
        )
    doc += "</table><p>收益值是本次调优样本，不能当作正式端到端加速结果。pending 仅证明已提交，只有实际融合返回和最终 codegen 成员才能确认落实。</p>"
    doc += (
        "<p>本次配置：<code>"
        + html.escape(
            json.dumps(
                {
                    k: effective.get(k)
                    for k in (
                        "epilogue_fusion",
                        "prologue_fusion",
                        "benchmark_fusion",
                        "combo_kernels",
                        "benchmark_combo_kernel",
                        "combo_kernel_max_num_nodes",
                        "max_epilogue_benchmarked_choices",
                    )
                },
                ensure_ascii=False,
            )
        )
        + "</code>。收益判断方式以实际分支和计时证据为准，不能只根据返回 true 认定做过 benchmark。</p>"
    )
    doc += "<details><summary>实际执行的阶段入口</summary><ul>"
    phase_names = {
        "Scheduler.create_foreach_nodes",
        "Scheduler.fuse_nodes",
        "Scheduler.finalize_multi_template_buffers",
        "Scheduler.create_combo_kernel_nodes",
    }
    doc += "".join(
        "<li>" + html.escape(e["function"]) + " " + event_link(e["seq"]) + "</li>"
        for e in events
        if e["event"] == "call" and e["function"] in phase_names
    )
    doc += '</ul><p>这里仅展示已观察的阶段入口；未单独覆盖的 MM template combo 或其他新路径需要适配当前源码，不能由缺少事件推断路径关闭。见 <a href="source_snapshot/torch/_inductor/scheduler.py">scheduler.py 源码快照</a>。</p></details>'
    doc += '<h2 id="kernels">每个 kernel 的形成过程</h2>'
    for index, kernel in enumerate(result["kernels"]):
        call = kernel["call"]
        doc += f'<article class="card" id="kernel{index}"><h3>K{index} · {html.escape(call["name"])}</h3>'
        doc += (
            "<p>读："
            + html.escape(", ".join(call.get("reads", [])))
            + "；写："
            + html.escape(", ".join(call.get("writes", [])))
            + "</p>"
        )
        doc += "<ul><li>" + tree_html(kernel["formation"]) + "</li></ul>"
        doc += (
            "<details><summary>ABI、读写 buffer、形状和 stride</summary><pre>"
            + html.escape(json.dumps(call, ensure_ascii=False, indent=2))
            + "</pre></details>"
        )
        local_lowering = [
            r
            for r in result["lowering_results"]
            if isinstance(r.get("fx"), dict) and r["fx"]["name"] in kernel["fx_origins"]
        ]
        doc += (
            "<details><summary>对应 FX 的 lowering 返回（"
            + str(len(local_lowering))
            + " 条）</summary><ul>"
        )
        doc += (
            "".join(
                "<li>"
                + html.escape(r["fx"]["name"])
                + " "
                + event_link(r["seq"])
                + "<code>"
                + html.escape(json.dumps(r["value"], ensure_ascii=False))
                + "</code></li>"
                for r in local_lowering
            )
            + "</ul></details>"
        )
        if kernel["generated_code"]:
            code_path = "diagnostics/kernels/" + call["name"] + ".py"
            doc += f'<p><a href="{html.escape(code_path)}">实际生成代码</a>；<code>{html.escape(call.get("provenance_key", ""))}</code></p>'
        for choice in kernel["choice_events"]:
            doc += (
                "<details><summary>相关候选及实现选择（未必最终采用）"
                + event_link(choice["seq"])
                + "</summary><pre>"
                + html.escape(json.dumps(choice, ensure_ascii=False, indent=2))
                + "</pre></details>"
            )
        for partition in kernel["partitions"]:
            doc += (
                "<p>codegen 分区："
                + html.escape(str(partition["groups"]))
                + " "
                + event_link(partition["seq"])
                + "</p>"
            )
        for gap in kernel["evidence_gaps"]:
            doc += '<p class="gap">' + html.escape(gap) + "</p>"
        doc += "</article>"
    doc += (
        '<h2 id="lowering">进入 Scheduler 之前</h2><p>每个 FX 的 lowering 返回与初始 Scheduler origins 都已记录。例如同一初始节点包含多个 FX origins，说明合并已发生在 Scheduler 普通融合之前；origin 归属本身不证明每项操作都产生了独立算术，也不能还原所有内联与物化原因。</p><details><summary>全部 lowering 结果（'
        + str(len(result["lowering_results"]))
        + " 条）</summary><pre>"
        + html.escape(
            json.dumps(result["lowering_results"], ensure_ascii=False, indent=2)
        )
        + "</pre></details>"
    )
    doc += '<h2 id="opportunities">具体融合机会停在哪一步</h2><input id="filter" placeholder="筛选节点、路径、拒绝原因或证据标签"><table id="opportunity-table"><thead><tr><th>节点 / 路径</th><th>停止位置 / 证据</th><th>事件及缺口</th></tr></thead><tbody>'
    ordered = sorted(
        result["opportunities"],
        key=lambda r: (
            r["route"] == "ordinary_fusion",
            r["status"] == "merged",
            r["pair"],
        ),
    )
    for record in ordered:
        doc += (
            "<tr><td>"
            + html.escape(" → ".join(record["pair"]))
            + "<br>"
            + html.escape(record["route"])
            + "</td><td><b>"
            + html.escape(record["status"])
            + "</b><br>"
            + html.escape(record["reason"])
            + '<br><span class="'
            + record["certainty"]
            + '">'
            + record["certainty"]
            + "</span></td><td>"
        )
        evidence = record["evidence"]
        doc += ", ".join(event_link(n) for n in evidence[-5:])
        if len(evidence) > 5:
            doc += (
                "<details><summary>全部事件</summary>"
                + ", ".join(event_link(n) for n in evidence)
                + "</details>"
            )
        doc += "".join(
            '<div class="gap">' + html.escape(g) + "</div>"
            for g in record["evidence_gaps"]
        )
        if record["final_colocation"]:
            doc += (
                "<p>最终同属 wrapper："
                + ", ".join(
                    f'<a href="#kernel{i}">K{i}</a>' for i in record["final_colocation"]
                )
                + "；是否纵向融合需查看形成树。</p>"
            )
        doc += (
            "<details><summary>候选、检查、收益判断及最终落实</summary><pre>"
            + html.escape(json.dumps(record, ensure_ascii=False, indent=2))
            + "</pre></details></td></tr>"
        )
    doc += '</tbody></table><script>document.querySelector("#filter").oninput=function(){const q=this.value.toLowerCase();document.querySelectorAll("#opportunity-table tbody tr").forEach(r=>r.hidden=!r.textContent.toLowerCase().includes(q));};</script>'
    doc += '<h2 id="combo">Combo 过滤与最终分区</h2>'
    if result["combo_decisions"]["branches"] or result["combo_decisions"]["filters"]:
        doc += combo_html(result["combo_decisions"], event_link)
    else:
        for item in result["combo_filters"]:
            doc += (
                "<p>"
                + html.escape(item["node"]["name"] + ": " + item["reason"])
                + " "
                + event_link(item["seq"])
                + '<span class="gap">'
                + html.escape(item["evidence_gap"])
                + "</span></p>"
            )
    for item in result["partitions"]:
        doc += (
            "<p>"
            + html.escape(str(item["groups"]))
            + " "
            + event_link(item["seq"])
            + "</p>"
        )
    doc += '<h2 id="native">原生记录与工具报告</h2><p><a href="compiler.log">完整原生编译日志</a> · <a href="observations/decisions.jsonl">补充调用事件 JSONL</a> · <a href="analysis.json">完整分析 JSON</a> · <a href="command.json">运行命令</a> · <a href="README.txt">使用说明</a> · <a href="postprocessing.json">报告后处理版本</a></p>'
    for parse in read(run / "tlparse_result.json", {}).get("traces", []):
        index_path = Path(parse["index"])
        try:
            relative = str(index_path.relative_to(run))
        except ValueError:
            relative = "tlparse/" + index_path.parent.name + "/index.html"
        if (run / relative).is_file():
            doc += (
                '<p><a href="'
                + html.escape(relative)
                + '">tlparse 原生 HTML 报告</a>，解析退出码 '
                + str(parse["returncode"])
                + "</p>"
            )
        else:
            doc += (
                '<p class="gap">tlparse 未生成 HTML 入口；解析退出码 '
                + str(parse["returncode"])
                + "。查看 tlparse_result.json 和对应解析日志。</p>"
            )
        if not (run / Path(relative).parent / "collectives_parity.json").exists():
            doc += '<p class="muted">tlparse 首页的 Collectives Parity 是通用入口，本次没有生成对应文件；FX、IR、生成代码和 combo_decisions 文件均单独保留。</p>'
    for group, files in result["native_artifacts"].items():
        doc += (
            "<details><summary>"
            + html.escape(group)
            + " ("
            + str(len(files))
            + ")</summary><ul>"
        )
        doc += (
            "".join(
                '<li><a href="' + html.escape(p) + '">' + html.escape(p) + "</a></li>"
                for p in files
            )
            + "</ul></details>"
        )
    doc += (
        '<h2 id="gaps">明确未捕获的过程</h2><ul>'
        + "".join("<li>" + html.escape(g) + "</li>" for g in result["coverage_gaps"])
        + "</ul>"
    )
    if result["observer"].get("errors"):
        doc += (
            '<p class="gap">观察器有错误，相关决策覆盖不完整：</p><pre>'
            + html.escape(
                json.dumps(result["observer"]["errors"], ensure_ascii=False, indent=2)
            )
            + "</pre>"
        )
    if (run / "failure.json").exists():
        doc += (
            "<h2>运行失败</h2><pre>"
            + html.escape((run / "failure.json").read_text())
            + "</pre>"
        )
    (run / "report.html").write_text(doc + "</body></html>")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    build_report(parser.parse_args().run_dir)
