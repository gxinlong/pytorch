# 输入、配置与命令

下文的 `SKILL`、`RUN` 和 `PYTHON` 是实际绝对路径；命令应由调用者按现有环境填入。脚本只需 Python 标准库即可执行 `--help`、`--inspect`；编译和 Torch 测试必须使用完整 Torch 环境。

## PT2 格式

- 原生 `torch.export.save` 的现代 PT2：读取 `models/<name>.json` 和对应权重、常量、sample inputs。只加载所选 EP，不为寻找图加载包内 AOT 二进制。当前适配使用 Torch 的 `_load_exported_programs`。
- legacy EP ZIP：通过 `torch.export.load` 加载。
- 只有 AOT 的 PT2：原始文件作为请求身份保留；使用 `--graph-source /original/export.pt2`，或包含以下四个文件的目录：
  `serialized_exported_program.json`、`serialized_state_dict.pt`、`serialized_constants.pt`、`serialized_example_inputs.pt`。
  不自动选择邻近 JSON，也不使用 Netron 专用改写文件。
- `--model-name` 在多 EP 包中选择一个模型。没有唯一选择时停止，避免编译错图。
- `--inputs /inputs.pt` 覆盖 example inputs，文件必须是 `(args, kwargs)`。Torch 保存和加载整体对象以保留存储关系；脚本会对 shape/stride/offset/alias 元数据做往返核对。

输入和 EP 的 device 必须匹配。CUDA 图在指定物理 GPU 的隔离可见环境中运行，通常图设备为 `cuda:0`；带其他设备索引、多 GPU 或自定义设备迁移要求的图需另行适配，不能静默改图。

## 配置

`--config` 接受纯 Inductor 配置字典，或以下结构：

```json
{
  "inductor_configs": {
    "combo_kernels": true,
    "combo_kernel_max_num_nodes": 30
  },
  "torch_backends": {
    "cuda.matmul.allow_tf32": false,
    "cudnn.allow_tf32": false
  },
  "environment": {
    "TORCHINDUCTOR_MEMORY_PLANNING": null
  }
}
```

`environment` 中 null 表示删除变量，`<work_dir>` 替换成此次运行目录。所有环境在新 worker 导入 Torch 前生效。

支持项目的 `{"default_profile":"default","profiles":{...}}`：

```bash
"$PYTHON" -B "$SKILL/scripts/diagnose.py" "$PT2" \
  --output-dir "$RUN" --gpu "$GPU" \
  --profile-file "$PROFILE_FILE" --profile default \
  --set combo_kernel_max_num_nodes=30 --tf32 off
```

优先级：当前环境/原生默认值 → 指定 profile → config JSON → `--set KEY=JSON` 和 `--tf32`。为取得完整编译过程，诊断日志与独立缓存路径由脚本固定，FX graph cache 关闭；这些诊断覆盖值也写入清单。不会默认开启 CatViewPass、combo 或关闭 epilogue。

## 项目初始化扩展

需要注册自定义算子、加载项目模块或提供优化上下文时，传入 `--setup /path/setup.py`。脚本须定义 `compilation_context(plan)`，返回上下文管理器；其进入发生在反序列化 EP 之前，退出发生在验证完成后。扩展文件会计入输入哈希。

```python
from contextlib import contextmanager

@contextmanager
def compilation_context(plan):
    import project_custom_ops
    with project_custom_ops.export_context():
        yield
```

导入路径由执行环境提供。此接口只是显式加载项目逻辑，不授予修改模型、部署或发送外部消息的权限。

## 正确性与历史对照

- 默认与 `ep.module()` 比较同一组保存输入。默认 `rtol=1e-4, atol=1e-5`，可显式调整。NaN/Inf、dtype、shape 和输出 pytree 一起检查。
- `--reference-outputs`：`torch.save(outputs, path)` 的原参考输出。
- `--reference-aot`：额外对照历史编译包，使用独立但 alias 一致的输入副本。
- `--baseline-compilation`：历史 `diagnostics/compilation.json`，对照 FX 身份、调度分组、调用与生成代码。独立编译改变调优结果时单列变化。
- 会修改输入或 buffer 的 EP 不自动套用纯推理正确性比较；需要对应验证方案。`--skip-validation` 是显式跳过，报告中的 passed 为 null，不得称通过。
- 此入口默认只验证选定的一组输入。多 batch、多种子和正式性能实验需独立安排，不自动随机造输入。

## 离线报告与文件

```bash
"$PYTHON" -B "$RUN/scripts_snapshot/diagnostic_report.py" "$RUN"
```

`execution.log` 保存 worker 输出；`compiler.log` 是原生文本日志；`trace/` 保留结构化 trace。
`source_snapshot/` 保存实际命中的源码，`identity.json` 保存完整 Inductor 源文件哈希和 native extension 身份。
`input_fx.json`、`input_fx.py`、`export_contract.json` 记录编译前图和 ABI；`diagnostics/graph_0/` 是 post-grad 图。
`inputs.pt` 是实际输入副本；`scripts_snapshot/` 是实际运行的 skill 脚本。
`status.json` 分别显示 complete、complete_with_gaps 或 failed。缺少 tlparse、观察错误、未完成调用或缺少最终 kernel 映射均需检查。

输入未决时可用 `--inspect` 检查，不创建编译目录：

```bash
"$PYTHON" -B "$SKILL/scripts/diagnose.py" "$PT2" --inspect
```

该检查只识别容器和图入口，不反序列化 tensor payload，也不证明包可编译；实际加载、输入契约及编译兼容性由 worker 检查。

脚本不会安装依赖、checkout、提交、推送或清理其他任务。
