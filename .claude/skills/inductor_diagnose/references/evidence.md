# 决策证据如何形成

## 三路采集

1. 原生 `TORCH_LOGS`、`TORCH_COMPILE_DEBUG`、`TORCH_TRACE`、`INDUCTOR_PROVENANCE`：保存 IR、生成代码、调优、WhyNoFuse、图改写及来源映射。tlparse 用结构化 trace 生成原生 HTML。
2. `compilation_recorder.py`：临时包装 debug/provenance、GraphLowering 和 C++ wrapper 入口，保留原调用，退出恢复。建立 FX origins、调度节点成员、buffer/view、wrapper 调用及代码的映射。
3. `diagnostic_observer.py`：主线程 `sys.setprofile` 观察指定函数的调用/返回、参数、局部状态和父调用；`diagnostic_combo.py` 用当前源码 AST 和 `sys.settrace` 捕获 combo 指定语句的执行分支和列表前后变化。不会为了补记录重跑 `can_fuse()` 或额外 benchmark。

函数调用、语句事件共用 seq。使用 call_id/parent_id、融合轮次、节点原始成员关联原因。AST 只定位观察点，过滤条件由编译器原本执行；复合条件内短路子项不重新求值。

## 形成树与停止位置

`diagnostic_report.py` 根据实际 `fuse_two_nodes` 返回构造普通/template 融合父节点；根据 `_register_accept` 和最终 schedule 记录真实 ForeachKernelSchedulerNode；用 `horizontal_partition` 和 provenance 映射到实际 wrapper。初始节点已有多个 FX origins 时，说明合并发生在 Scheduler 之前，但不证明所有 FX 都单独产生算术。

`create_foreach_nodes` 阶段已建立的 foreach 容器单独展示，不计为后续通用 combo 的接受结果；其 codegen 分区仍关联回该容器。

机会按候选、检查、收益、异步回调、实际合并分类。pending 不等于接受；共享 epilogue 被别的融合占用可以由事件推导，但未捕获逐候选移除时仍是 inference。之前拒绝、后来完成融合的候选按实际最终合并解释。

`diagnostic_combo_report.py` 提供每节点的过滤、接受、单例组或 unknown：

- 过滤原因关联实际 `filtered_nodes` 变化与源码条件。
- 拓扑、device/stream/mempool、数量切块、距离窗口使用实际输出。
- combo 收益函数直接放行需匹配对应调用内的实际分支，不能看到 true 就声称 benchmark 获胜。
- 峰值内存检查优先引用原生日志；日志没有成员名或绝对峰值时，不按数量强行绑定到某候选。
- codegen 参数预算是读写依赖计数，不是最终去重的 kernel ABI 参数数目。

## 覆盖边界

`kernel_dependencies.py` 将最终 wrapper 调用及其 `call_schedules` 与最终 Scheduler 的非发射节点读写关联，生成 kernel 依赖图。比如 concat 的 Nop 节点不会画成额外 kernel，而是追溯其输入 buffer 的生产者，连接到真正的消费者。图中保留全部已记录 RAW 边，不因存在间接路径就删除直接数据依赖。

输入/输出是本次编译图的边界，可能含 lifted 参数。采集器保存最终 `graph_inputs` / `graph_outputs`，报告连接输入到消费 kernel、生产 kernel 到实际返回值；保留重复输出、view 元数据和直接返回输入。旧记录只使用 post-grad 图的返回结构及 lowering 值映射恢复边界，并明确缺少最终返回布局修正的独立记录。FX 内部邻接不参与 kernel 连线。没有 buffer 的标量输出、缺失输出来源或未记录消费的输入均显式标注，不推断为可删除。

每个调用实例单独编号，即使复用同一个 kernel 名称也不合并。融合内部的 buffer 不形成自环。对同名 buffer 的多次写入，仅在写覆盖为 full 时替换旧生产者；partial/atomic/opaque 保守保留。WAR/WAW 使用虚线，表示 buffer 粒度顺序约束，未证明访问区间重叠；这类边不能作为纵向融合的直接依据。未记录的 alias、内存复用、控制流与 stream/event 同步不靠 FX 边或发射顺序补造。

仅记录实际发生的主线程调用和已适配语句，不覆盖所有线程、所有 pass、所有分支子项或外部库内部 GPU kernel。仅记录 return false 且没有 WhyNoFuse 原因时，不将其改写为确定的布局或性能问题。

`observer_status.json` 的 errors、unfinished_calls、combo_pending_statements、missing_combo_targets 应检查。缺少事件不是路径关闭的证据。即使错误为零，也不表示所有决策路径都已适配。

编译器版本变化可能改变函数、变量或源码结构；保留当前源码快照并适配观察点，不将旧函数名称的含义套用到新版本。观察器开销可能影响调优时序，所以诊断包不能自动视为历史包的重放，也不用于正式性能计时。

## 从已完成报告继续工作

先看身份/图与历史分组对照，再看 kernel 形成树、具体融合机会和 combo。用事件编号回到原始记录与源码行。
raw trace 和 decisions.jsonl 保持不变；后处理可独立迭代并记录版本。tlparse 中的补充 combo_decisions artifact 是生成时的分析快照，后续离线报告变化不会自动回写原始 trace。
