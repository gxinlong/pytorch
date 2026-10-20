---
name: inductor_diagnose
description: 对给定 PT2 中的 ExportedProgram/FX 图执行一次 Inductor 诊断编译，保存原生 trace、kernel 来源、fusion/combo 决策和中文 HTML 报告。适用于分析 kernel 如何形成、融合机会停在哪一步；编译配置和正确性对照可显式指定。
---

# Inductor Diagnose

使用本目录 `scripts/diagnose.py`，不要依赖某次实验目录中的脚本。保留用户指定的模型、配置、输入 ABI 和动态约束。完整参数、输入格式及配置说明见 [references/usage.md](references/usage.md)；解释事件和证据缺口时读 [references/evidence.md](references/evidence.md)。

## 输入与环境

用户只需给出 PT2 路径；从现有会话确定执行主机、Python、源码和已授权 GPU。先用 `diagnose.py PT2 --inspect` 检查包类型，该操作不导入 Torch。

- 优先使用包中的 ExportedProgram 和 example inputs。多个程序必须明确选择 `--model-name`。
- 仅有 AOT 二进制时，不能从二进制或不完整 graph JSON 恢复模型。寻找对应原始 EP，或完整的序列化 sidecars，并显式传入 `--graph-source`；记录其身份。缺少图、权重或合法输入时明确说明缺少什么。
- 外部输入使用 `torch.save((args, kwargs), path)`。禁止逐 tensor `.contiguous()`、随机替换或改变 batch 来绕过问题；保持 shape、stride、offset、storage alias 和标量契约。
- 若项目已指定编译 profile，沿用它并显式传入；有 default/safe 合约时默认 default，safe 仅在用户要求时启用。额外参数覆盖 profile，原生 Torch 默认值不会被本 skill 改成某个历史模型的设置。
- CUDA 必须沿用已确定的 GPU，传 `--gpu` 或单 GPU 的 `CUDA_VISIBLE_DEVICES`。占用时停止此次启动，不切卡、不杀进程。脚本不迁移图的 CPU/CUDA device。
- 依赖能加载该 EP 的 Torch、对应 Inductor 工具链；CUDA 使用兼容 Triton。tlparse 用现有可执行文件或 `--tlparse` 指定。缺少 tlparse 时保留原始 trace，明确原生 HTML 未生成；不要把整体报告宣称为完整。

## 执行

在用户指定的工作目录创建新运行目录。此仓库默认放在 `agent_space/model_subgraph/{模型名}/diagnose_{时间戳}/`；不覆盖原始包或已有报告。源码、缓存、日志、测量数据不写入 skill 目录。

```bash
python -B <skill目录>/scripts/diagnose.py /绝对路径/model.pt2 \
  --output-dir /绝对路径/新的运行目录 \
  --source-root /绝对路径/pytorch \
  --gpu <已确定的GPU索引或UUID>
```

`--source-root` 校验实际导入源码；省略时记录当前环境的源码身份。编译由新进程执行，环境在 Torch 导入前设置。脚本快照、源码哈希、输入哈希、配置和 dirty diff 都随运行保存。

一轮只进行一次 `aoti_compile_and_package`。成功后加载包，并默认与原 ExportedProgram 对同一组独立克隆、保持 alias 的输入比较。额外历史 AOT/参考输出须显式传入。失败保留目录与报告；查明原因后在新目录修复重试，不自动换配置、精度或输入。模型自定义算子和优化上下文可通过用户提供的 `--setup` 扩展，不为通过编译暗中插入模型专用改写。

需要修改报告时离线运行 `scripts/diagnostic_report.py RUN_DIR`，不要重新编译；保持原始事件和 trace。若报告逻辑已更改，保存后处理脚本版本及哈希。

## 报告与交付

核心产物是 `report.html`、`analysis.json`、`observations/decisions.jsonl`、`diagnostics/compilation.json`、`subgraph.pt2`、`validation.json` 和可用时的 `tlparse/trace_*/index.html`。

先报告编译/包加载/正确性/诊断覆盖各自状态，再给本地报告和执行主机 AOT 的绝对路径。解释每个 kernel 的形成树，明确普通融合、template、foreach/combo 和 codegen 分区的关系。对具体融合机会使用实际候选、检查、收益、合并和分区证据。

保留 `direct / inference / unknown` 区分；pending 不等于接受，没有 benchmark 记录不等于性能失败，候选没有出现不等于合法性失败。一组 combo 不一定是一个最终 kernel。观察器有开销且仅覆盖部分路径；图一致不保证与历史编译的分组或调优选择一致。编译调优时间、kernel 数量和正确性通过均不能代替正式端到端性能结论。

适配器目前以此仓库 Torch 2.14 的内部接口为依据；换版本时检查 `observer_status.json`、缺失的观察目标和映射数量。未知路径标明未覆盖，不编造原因。运行 `scripts/test_model_io.py`、`scripts/test_diagnostic_combo.py` 和 `scripts/test_diagnostic_report.py` 验证改动；测试临时文件也置于运行目录的 `TMPDIR`。
