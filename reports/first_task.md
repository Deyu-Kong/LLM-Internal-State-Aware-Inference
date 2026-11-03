# 第 14 节交付报告

## 范围与环境

完整阅读研究计划后，仅实施 repository scaffold 与 reproducibility infrastructure。环境为 WSL2、Python 3.10.12、uv 0.13.0；初始目录只有研究计划，无 Git 仓库。未连接 a800/node32，未运行 GPU 训练，也未实现三个 WP 的方法。

## 修改文件

- `pyproject.toml`、`uv.lock`、`.gitignore`：可安装 src 包、CLI、固定依赖和生成物排除。
- `src/state_inference/__init__.py`、`__main__.py`、`config.py`、`manifest.py`、`cli.py`：包入口、YAML 校验、版本追踪、manifest、CPU smoke 与 JSONL 日志。
- `configs/smoke.yaml`：独立的默认示例配置。
- `tests/unit/test_config.py`、`tests/integration/test_smoke.py`：配置错误、CLI、产物、哈希、日志、种子复现与防覆盖测试。
- `README.md`、`docs/*.md`、`scripts/README.md`：安装执行、实验数据规范、接口边界及后续工作。
- `configs/`、`data/`、`src/state_inference/` 和 `tests/` 内的 `.gitkeep`：研究计划目录占位。
- `results/examples/first-task-smoke/`：真实运行的 config、manifest、dataset manifest、events 与 result。

## 验证

`uv sync --locked --group dev` 完成安装；`uv run --locked pytest -q` 共 11 项通过；`uv lock --check` 通过；已确认包可导入、版本为 0.1.0。CLI smoke 成功生成完整产物。示例配置快照存于示例输出目录，默认配置使用另一运行 ID，避免示例产物阻碍首次运行。

以上仅为基础设施验证，不是模型实验结论，不支持任何 WP 的研究假设。

## 已知限制与下一步

本任务无阻塞。目录没有 Git，因此 commit/dirty 记录为 null；建议开始研究迭代前初始化版本管理。未安装或检查 GPU 研究栈，CUDA/GPU 与模型数据字段明确为空；正式实验前须补齐。

下一任务建议限定为状态采集：先只读检查优先节点 `ssh a800`（备选 `ssh node32`）的硬件与现有环境，再固定兼容依赖并确定小型 Causal LM 的模型/tokenizer revision。在 CPU 或短 GPU forward 中验证 hidden states、logits、KV 的层/位置与形状及主生成不变性，暂不训练 Probe。OTV 论文与官方代码研究随后单独推进。
