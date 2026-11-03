# Git 建仓与内部状态采集验证

## 完成范围

建立本地 main 分支并推送至 https://github.com/Deyu-Kong/LLM-Internal-State-Aware-Inference 。fetch 使用用户指定 HTTPS URL，push 使用已认证的 GitHub SSH；没有更改全局 Git 配置。初始提交为 f600b84，状态采集代码为 0762360，实际检查基于干净提交 1c42433f7458e4f6a79db9e1d5273e35ace8814d。

a800 项目位于 `/root/workspace/llm-state-aware-inference`，在该目录建立独立 `.venv`，只读复用 qwen 环境的依赖。没有修改共享 conda 环境。安装 pytest 时默认镜像返回 403，改为命令行指定官方 PyPI 后成功，没有修改系统 pip 配置。节点系统盘余量约 48GB，因此不复制大型依赖或下载权重。

新增只读状态快照 API、CPU/GPU 检查脚本、配置、依赖记录、测试和运行文档。AGENTS.md 固定 GPU 使用限制。没有实现 Probe、WP2、WP3 或 GPU 训练。

## 资源限制与验证结果

预估单次检查不超过 2 分钟、显存不超过 1GB、输出不超过 1MB。实际仅在空闲物理 GPU 0 上执行一次微型随机 Llama 验证；通过 UUID 限制可见设备。GPU 1 未用于计算，GPU 2–7 未用于计算；没有终止其他进程或改变 GPU 设置。结束后物理 GPU 0、1 均为 0 MiB/0%。

- 本地：`uv run --locked pytest -q` → 18 passed，1 skipped（整个模型测试模块因无 torch 跳过）。
- a800：`CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 .venv/bin/python -m pytest -q` → 25 passed in 3.98s。
- CPU 状态检查成功，验证循环耗时 0.0200 秒。
- GPU 0 状态检查成功，验证循环耗时 0.9636 秒；PyTorch 峰值 allocated 为 8,622,592 字节（约 8.2 MiB），不是包含 CUDA context 的进程总显存。
- 四步 greedy cached decode 的 token、每步 logits、最终 KV 在有/无采集的独立分支中精确一致。
- 状态形状：logits `[1, 1, 32]`，hidden `[1, 1, 32]`，选中 KV `[1, 2, 2, 8]`。
- 测试验证快照修改不会影响原始状态，并拒绝越界索引、未授权 GPU 和已占用 GPU。

耗时不含 Python 导入、模型构造等准备工作，无 warmup，不能解释为吞吐 benchmark。随机模型结果不能代表预训练模型质量或任一 WP 研究假设。

## 产物与复现

完整配置、模型配置、数据清单、manifest、事件日志与结果已归档到 `results/examples/state-capture/`，并附环境包版本快照。每次运行使用新的 experiment_id，不覆盖历史产物。主要依赖：Python 3.12.13、torch 2.11.0、transformers 5.8.0、CUDA runtime 13.0、驱动 595.58.03；精确命令见 `docs/state_capture.md`。

```bash
OMP_NUM_THREADS=2 .venv/bin/python scripts/check_state_capture.py --config configs/state_capture.yaml
OMP_NUM_THREADS=2 .venv/bin/python scripts/check_state_capture.py --config configs/state_capture_gpu.yaml --gpu 0
```

## 限制与下一任务

当前仅验证普通 DynamicCache、微型 Llama 和 eager attention；未验证滑窗/量化缓存、真实预训练模型、随机采样或多卡。快照会复制到 CPU，存在同步成本；后续必须实测采集开销。复用依赖环境仍可能被外部更改，正式实验前应建立完整独立锁定环境。空闲检测与 CUDA 启动之间存在时间窗口，共享节点上仍需遵守资源分配约定。

建议下一任务单独研究 OTV 官方论文和实现，明确 probe token、KV 旁路、attention、LoRA、标签、loss 与开销，再提出 Single Probe 的最小复现计划。实际复现先验证短 forward/backward，不直接启动训练；GPU 限制继续严格执行。
