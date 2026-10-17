# Kernel-name FX subgraph tool

从同一次 AOT 编译的完整 kernel name 反查所有调用，提取 post-grad FX 节点并集，采集真实 AOT 边界，再独立编译和测量。名称必须完全匹配；数字后缀不能省略。同名多调用全部纳入。

工具源码位于 Git 管理的 `tools/subgraph/`，入口为 `python -m tools.subgraph`。旧 `agent_space/inductor_test/subgraph/` 副本暂时保留，用于历史实验生成的入口；后续开发和新实验使用本目录。模型、快照、日志继续放在忽略的 `agent_space/inductor_test/` 下，生成源码、AOT 包和缓存放在 `build/inductor_test/` 下，不纳入 Git。

当前适配环境为 redpredict2 的 PyTorch 2.14、源码 `2b3ec34829036a65cd9d1398ea72a0167dc37470`、现有 TorchFourier 和物理 GPU3。工具检查源码、运行库和 GPU UUID，不修改 Inductor 默认优化规则。TorchFourier 的 Triton 版本检查在工具进程内适配到现有运行库，记录原要求及实际版本。

默认编译器基线是官方 `v2.14.0`，即上述提交。允许在此基础上修改或提交 `tools/subgraph/`，但该目录之外的提交差异、暂存/未暂存改动及未跟踪文件均会阻断运行。每次 prepare/capture/verify/独立重放记录实际 checkout 提交、编译器参考提交和工具 Python 源码哈希；工具分支的提交号不会被误报为基线提交。

## 使用

在远端仓库根目录，使用现有环境运行：

```bash
PY=/opt/miniconda3/envs/torchfourier-offline-torch214/bin/python
TOOL=tools.subgraph
$PY -m $TOOL prepare --exported-program /absolute/path/concat.pt2 --run-dir /absolute/path/new-run --build-dir /absolute/path/build/inductor_test/new-run
$PY -m $TOOL extract --bundle /absolute/path/new-run/bundle.json --kernel FULL_NAME_1 --kernel FULL_NAME_2
$PY -m $TOOL capture --case /absolute/path/new-run/cases/CASE/case.json
$PY -m $TOOL verify --case /absolute/path/new-run/cases/CASE/case.json --benchmark
```

`extract` 打印新 case 路径。`capture` 默认分别采集 B=80、1、256，也可重复传入 `--batch`。所有输出均采用独占创建；失败产物保留，重试时重新 extract 创建新 case。不能用历史 AOT 包冒充 prepare 的新基线。

采集成功后，`verify --case … --attempt NAME` 可复用真实边界数据，在新的 `verify_NAME/` 目录重新独立编译和验证，保留原验证记录。

采集副本复用基线的算法测速、`.kernel_perf` 融合测速和 `.best_config` 记录，并保存来源文件哈希。图缓存关闭，实际重新执行 lowering/codegen 和采集插桩。独立子图使用新的空缓存，不复用这些测速结果。

`prepare` 的加载器复用 TorchFourier `const2placeholder`、必要算子注册、CUDA 转换、优化上下文、captured-b80 及三个计数字段规范化。B=1/256 由同一原始样本按既有规则裁剪或重复，每个 batch 记录实际输入哈希。首版加载器面向带 `aotconfig.json` 的这个 TorchFourier 模型目录。

## 产物与判定

| 文件 | 内容 |
|---|---|
| `bundle.json`, `inputs.json`, `compilation.json` | 源码与运行库身份、配置、输入哈希、Provenance、FX 到 IR 到 wrapper 的映射 |
| build 下 `input_graph/`, `diagnostics/graph_0/` | 可恢复 GraphModule 结构、常量、来源、结构校验及现有 FX repro 导出 |
| build 下 `debug/`, `diagnostics/wrapper_1.cpp` | 融合前后 Scheduler IR、最终 wrapper、编译源码 |
| `mapping.json`, `boundary.json` | 选中名称、全部调用点、拓扑节点、额外关联 kernel、逻辑边界和共享输入 |
| `capture/compiler_parity.json` | 基线与采集副本的图、分组、参数、配置/代码指纹、值映射核对 |
| `capture/b*/` | 原始调用处快照、实际 scalar/SymInt、恢复后的共享 storage/view、完整模型输出核对及来源 |
| `verify/fusion.json` | 节点对应、融合分组、读写关系、新增/拆分节点、物化与模板配置差异 |
| `verify/b*.json`, `verify/verification.json` | eager、独立 AOT 与原 AOT 区域参考比较，ordinary/CUDA Graph 原始样本及中位数 |

现有 repro 中随机输入生成代码只作源码诊断，绝不用于边界数据或数值验收。实际恢复入口是 `core.load_graph` 和 `core.load_tensors`。

验证后生成 build 下的 `verify/run_experiment.py`。它直接加载独立包及真实边界，无需加载或运行完整模型：

```bash
$PY /absolute/path/build/inductor_test/RUN/cases/CASE/verify/run_experiment.py \
  --batch 80 --mode aot --benchmark \
  --run-dir /absolute/path/new-replay \
  --build-dir /absolute/path/build/inductor_test/new-replay
```

`--mode eager` 可重跑提取图。实验入口仍检查物理 GPU3 和源码身份。生成名称在定义、metadata 和 `async_compile` 注册处统一规范化；从旧格式记录恢复源码时先核对原始 SHA-256。

数值使用 FP16 `rtol=1e-3, atol=1e-3`、FP32 `rtol=1e-4, atol=1e-5`，记录最大绝对/相对误差与逐元素相等数量，不自动放宽阈值。独立运行、数值、融合分组、实现配置分别报告。融合分组通过但模板/BLOCK 不同，只能称为融合分组复现。未知或不能对应的 kernel 指纹不视为相同。

融合比较以实际调用和 Scheduler 的 `exact_origin` 区分计算成员与下游继承的来源信息；后者保留在 `inherited_external_provenance` 中。相同表达式的多个节点按输出位置及操作数路径对应，输出 buffer 的共享关系变化另列为物化差异。`200*B` 等单符号正仿射 scalar 边界从 tensor 动态维重建，不将采集时的 batch 固化为常数。

benchmark 使用预驻留 GPU 数据和无采集的独立包，30 次预热、20 个样本、每样本 10 次调用。ordinary 和 CUDA Graph 分开保存 CUDA event 时间、墙钟时间，并核对 graph 输出。采集、编译和磁盘操作不进入计时；不据此计算相对历史 Nsight 时间片的加速比。

## 支持范围

支持单 GPU、无副作用、可恢复 tensor/view/scalar 边界和单符号正仿射动态维。相同 IR buffer、相同布局的输入合并为一个参数；不同 view 由共同 backing storage 恢复。数值相同不构成别名，内存池地址复用也不构成别名。

对 mutation、跨区域写入、无法证明完整的部分写入、循环分区、未物化边界、缺少映射、不可恢复动态关系或多 post-grad 图，给出阻断原因。第一版不自动扩大选择范围，不额外 realize，也不修改融合规则。中间快照来自本工具生成的可信 AOT 包，不接受来历不明的 pickle 文件。

明确支持记录为 FixedLayout/ReinterpretView 的边界。尚未记录 backing-storage 映射的 NonOwningLayout，以及未解析的 BaseView，明确拒绝；不把它们当独立 storage 恢复。不同布局的 view 参数保留各自的逻辑值并共享恢复的 storage；同布局同 buffer 的重复参数合并。

## 测试

```bash
CUDA_VISIBLE_DEVICES=GPU-693e3eac-1d0e-2b9f-c31b-1a8efbcce283 \
  /opt/miniconda3/envs/torchfourier-offline-torch214/bin/python \
  -m tools.subgraph.tests.test_core -v
/opt/miniconda3/envs/torchfourier-offline-torch214/bin/python \
  -m tools.subgraph.tests.test_source -v
```

设备数值测试使用 device type tests；其他测试使用仓库 TestCase、run_tests 和 parametrize。执行前确认 GPU3 空闲。真实 AOT 小图和 Tianjiao 实验报告放在独立 task/run 下，不纳入 Git。
