# LLM Internal-State-Aware Inference

毕业论文研究共享实验平台。已完成研究计划第 14 节的项目骨架与可复现基础设施，并增加微型模型内部状态采集验证。增加了依据论文的单 Probe forward/backward 检查（[研究说明](docs/otv_research.md)），尚未训练或完成官方 OTV 复现。WP1 多视角 KV Probe、WP2 预算感知验证、WP3 状态感知投机解码尚未实现。smoke 不加载模型、不下载数据、不调用 GPU。

## 安装与运行

需要 Python 3.10+ 与 uv。在项目根目录执行：

```bash
uv sync --locked --group dev
uv run --locked pytest -q
uv run --locked state-inference smoke --config configs/smoke.yaml
```

也可通过 `uv run --locked python -m state_inference smoke --config configs/smoke.yaml` 运行。

仓库附带一次真实输出：`results/examples/first-task-smoke/`。默认配置写入独立的 `results/runs/local-smoke-001/`，首次运行可直接执行上面的命令。运行目录不可覆盖，后续每次运行请编辑 `configs/smoke.yaml`，使用新的 `experiment_id`。输出路径相对于运行命令时的工作目录。

安装仅包含 PyYAML 和开发测试依赖，`uv.lock` 固定传递依赖。尚未选定的 PyTorch、Transformers、PEFT、Datasets、vLLM/CUDA 组合不在本次安装范围，后续应针对实际远程硬件单独固定。

## 配置与输出

配置严格要求 `experiment_id`、`seed`、`output_dir` 三项。seed 为 32 位非负整数，ID 不允许路径分隔符；未知字段会报错。

每个运行目录包含：

- `config.yaml`：解析后的完整配置。
- `manifest.json`：配置 SHA256、Git SHA/dirty、Python/依赖版本、平台、种子、UTC 时间与实际运行耗时。
- `dataset_manifest.json`：本次无数据集，明确标注未使用。
- `events.jsonl`：结构化开始/完成日志，含版本、种子和运行时间。
- `result.json`：固定种子的基础设施校验输出，**不是研究性能指标**。

未使用的模型、tokenizer、adapter、数据集版本与推理参数为 null/空；未检查 GPU/CUDA，不将“未检查”解释为“不可用”。没有 Git 仓库时 SHA/dirty 为 null。启用 Git 后会自动采集。smoke 的 seed 仅用于 Python RNG，不代表已经配置 GPU 训练确定性。

复现示例结果时，将输出中的 `config.yaml` 复制到一个空目录，在已安装环境中执行 `state-inference smoke --config config.yaml`。随机校验值应一致，时间和主机信息自然不同。

## 目录

`src/state_inference/` 为可安装包；`configs/` 为配置；`tests/` 为单元与集成测试；`data/` 保存数据及清单；`scripts/` 保留后续脚本；`results/` 保存实验产物；`reports/` 为技术报告；`docs/` 为规范。除只读状态快照外，研究方法模块目前只有目录占位。

## 计算节点与后续范围

真正的计算实验优先使用 `ssh a800`，备选 `ssh node32`。已在 a800 建立项目独立环境，状态采集检查详见 [运行文档](docs/state_capture.md)。仅允许物理 GPU 0、1；每次检查占用后按 UUID 限制可见设备。不得使用 GPU 2–7、终止他人进程或修改共享环境。长 GPU 作业前必须估计时间、显存与输出存储。

详见 [实验规范](docs/experiment_protocol.md)、[数据规范](docs/dataset_protocol.md)、[本次报告](reports/first_task.md) 与 [完整研究计划](llm_internal_state_research_implementation_plan.md)。

状态采集的实测结果见 [报告](reports/state_capture.md)，归档产物见 `results/examples/state-capture/`。

单 Probe 的 CPU/GPU 验证结果、初始化修正与已知缺口见 [报告](reports/single_probe.md)。

数据准备已支持固定版本 GSM8K 参考解答和 1,000 条受控合成算术轨迹，切分、字段与精确运行命令见 [数据规范](docs/dataset_protocol.md)。此阶段仅使用 CPU，不生成模型轨迹或训练。

本轮数据统计与跨机器验证见 [数据准备报告](reports/data_preparation.md)。

PRM800K 小子集训练架构、标签过滤、资源边界与运行命令见 [训练设计](docs/prm800k_pilot.md)。

单 Probe 的多随机种子、数据偏差及预训练/随机骨干对照见 [诊断报告](reports/prm800k_diagnostics.md)。

扩大原题、同前缀配对及独立校准/测试实验的预设协议见 [扩展实验方案](docs/prm800k_expanded.md)。
