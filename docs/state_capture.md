# 最小状态采集检查

本里程碑只验证状态读取机制，不实现 Probe 或训练。使用随机初始化的两层 Llama（32 hidden size，32 token vocabulary），固定输入 token IDs，不需要 tokenizer 或权重下载。不能据此宣称真实模型的质量、吞吐或研究假设得到支持。

## 接口

`capture_states(output, hidden_layers=..., kv_layers=..., positions=..., kv_positions=...)` 读取已有 forward 的 hidden states、logits 和 KV；forward 必须开启 `output_hidden_states=True` 与 `use_cache=True`。hidden layer 0 是 embedding，KV layer 0 是第一个 transformer block。支持负索引；位置保留 batch 维。`positions` 是本次 forward 的位置，`kv_positions` 是完整 cache 中的位置，默认沿用 positions。

快照 detach 后复制到 CPU，无共享 tensor 存储，也不回写生成 cache。CPU 拷贝会同步数据，当前仅为正确性原型，未声称采集开销低。验证范围是普通 DynamicCache 和 legacy tuple；未验证滑窗、量化缓存或其他模型架构。

## a800 运行

远程使用项目目录 `/root/workspace/llm-state-aware-inference`。只读复用已有 qwen 环境的 torch/transformers；在项目目录创建独立 venv，测试工具仅安装到该 venv：

```bash
/root/miniconda3/envs/qwen/bin/python -m venv --system-site-packages .venv
.venv/bin/python -m pip install --index-url https://pypi.org/simple -r requirements/state-capture.txt
.venv/bin/python -m pip install --index-url https://pypi.org/simple --no-deps -e .
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 .venv/bin/python -m pytest -q
OMP_NUM_THREADS=2 .venv/bin/python scripts/check_state_capture.py --config configs/state_capture.yaml
```

提供的 CPU 与 GPU 配置已使用不同 experiment_id。GPU 检查命令：

```bash
OMP_NUM_THREADS=2 .venv/bin/python scripts/check_state_capture.py --config configs/state_capture_gpu.yaml --gpu 0
```

`--gpu` 只接受物理编号 0 或 1。脚本在导入 torch 前查询所选 GPU 的显存、利用率及计算 PID，全部为空闲才设置 `CUDA_VISIBLE_DEVICES=<UUID>`；否则退出。可见逻辑设备 0 映射到该 UUID，不代表任意物理 GPU 0。查询与启动存在时间窗口，共享节点上仍需遵守资源分配约定，不运行常驻或长时间任务，不结束其他任务。

四步 greedy cached decode 对比不采集/采集两条独立分支：要求每步 logits、token 和最终 KV 精确一致。测试另检查快照修改不影响原始状态，以及非法索引拒绝。输出包含配置、完整模型配置、数据清单、版本/commit/设备 manifest、日志与结果；GPU 耗时同步测量，但无 warmup，不作性能基准。
