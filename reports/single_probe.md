# OTV 研究与单 Probe 最小计算路径验证

## 结论与范围

完成论文方法审阅和独立单 Probe 原型，CPU/GPU 上均完成一次有效 forward/backward。通过测试验证冻结参数、有限梯度、主 KV 隔离、后续生成不变以及历史/自身 attention。未运行 optimizer 更新、训练或真实数据评测；没有实现多 Probe、WP2、WP3。

**官方代码审计仍存在缺口**：论文给出的 GitHub 及公开 API 在 2026-11-03 返回 404，未取得源码 commit 或 checkpoint。因此本交付不能称为官方 OTV 复现。来源版本、PDF 哈希和访问状态记录在 `data/manifests/otv_sources.json`；论文确认细节与本地实现选择分列在 `docs/otv_research.md`。

## 主要修改

- `src/state_inference/kv_probe/single_probe.py`、`__init__.py`：冻结 Llama，验证上下文启用 Q/K/V LoRA，独立可学习 Probe embedding、三层 sigmoid 回归 head，MSE 伪目标辅助函数，严格限定普通 DynamicCache 的分支拷贝。
- `src/state_inference/runtime/devices.py`：复用原 GPU 空闲检查与 UUID 隔离逻辑；原状态采集脚本改用共享 helper。
- `scripts/check_single_probe.py`、`configs/wp1_probe/single_probe_{cpu,gpu}.yaml`：一次 forward/backward、零优化器更新、梯度/参数/主缓存/续写检查及原始计时样本。
- `tests/integration/test_single_probe.py`：新增 14 项模型相关测试。
- `docs/otv_research.md`、`data/manifests/otv_sources.json`、README：文献、复现边界、独立实现取舍和运行方法。
- `results/examples/single-probe/`：保留两轮 CPU/GPU 原始产物、环境版本快照及本次初始化修正说明。

## 初始化修正与可追溯性

首轮 `*-001` 对应干净提交 `7c1e195275f946501ec0b2908f7c406a8d0ed5f2`。模型先迁到目标设备，再初始化 LoRA/Probe/head；CPU 与 CUDA 的随机数流不同，同一 seed 产生了不同 Probe 参数，故这轮不适合 CPU/GPU 数值对照。两条路径各自的隔离与梯度检查均通过，首轮原始结果保留。

发现原因后，提交 `02c65dc54f8efb12c2522fd74331997ad58d855f` 将所有参数统一在 CPU 初始化后再迁移。重新使用 `*-002` ID 运行，不覆盖首轮。这两次运行均基于干净工作区，CPU 与 GPU 的 score/loss 在 rtol=1e-5、atol=1e-6 下匹配；此次具体数值也恰好相等，但不要求一般情况下跨设备位级一致。

## 验证结果

- 本地轻量：`uv run --locked pytest -q`，18 passed、2 skipped（两个模型测试模块因无 torch 跳过）。
- a800 CPU 全量：`CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 .venv/bin/python -m pytest -q`，39 passed in 4.11s。
- 测试将 LoRA-B 设为非零后，普通生成与未包装模型连续三步 logits/KV 精确一致，证明隔离不只依靠初始零增量。
- 零增量旁路的 hidden/attention 与整段 prefix+Probe 重算参考匹配；attention 覆盖四个 prefix token 和 Probe 自身。
- inference-mode KV 可安全 clone 为普通 detached tensor 后反传；异常退出恢复 gate；空 cache、重复包装及非法标签被拒绝。
- 本轮合成检查有 2,817 个可训练参数。可训练项均有有限梯度；LoRA-B、Probe embedding 有非零梯度。因 LoRA-B 初始为零，LoRA-A 首次梯度为零；额外测试用非零 B 验证了 A 的非零梯度路径。
- 原始 generator KV 内容/长度不变，后续三步生成 logits/KV 不变；全部参数保持不变，optimizer steps=0。

最终 `*-002` 诊断测量：

| 项目 | CPU | GPU 0 |
|---|---:|---:|
| autograd forward，单次 ms | 3.4683 | 14.9150 |
| backward，单次 ms | 114.8091 | 61.2587 |
| KV fork，均值 ms | 0.0923 | 0.1935 |
| 私有 Probe forward，均值 ms | 1.2971 | 2.7268 |
| 含 fork 的端到端 Probe，均值 ms | 1.4030 | 2.9022 |
| 整个验证循环，秒 | 0.1701 | 0.5490 |

均值来自 3 次 warmup 后的 10 次 no-grad 样本；forward/backward 单次包含冷启动效应，不是训练速度 benchmark。GPU 计时有 synchronize，微型模型上的 CPU/GPU 速度差异不能外推。峰值 PyTorch allocated 为 18,331,136 字节（约 17.5 MiB），包含整个诊断的模型/校验副本等，不是 Probe 增量显存，也不是 CUDA 进程总显存。四组原始结果及环境共约 40KB。

合成 outcome 是任意梯度夹具；target=0.75 和 loss 只能证明计算成立，不是可信度校准、错误检测质量或 WP1 增量收益。此阶段不触发 WP1 科学 Go/No-Go。

## 资源与环境

a800 的项目独立 `.venv`，复用只读依赖；Python 3.12.13、torch 2.11.0、transformers 5.8.0、CUDA runtime 13.0、驱动 595.58.03。无新增 GPU 依赖、模型权重下载或共享环境改动。

两次短 GPU 检查均仅使用物理 GPU 0，启动前经占用检查并按 UUID 限制可见设备。GPU 1 未用于计算，GPU 2–7 未用于计算，没有终止进程、修改 GPU 设置或留下作业。结束后 GPU 0、1 均为 0 MiB、0%。单次运行小于预估的 2 分钟/1GB/1MB。

## 复现与下一步

从仓库根目录执行 `docs/otv_research.md` 中的命令；修改为新的 experiment_id。每组产物自带 config、model/probe config、数据与文献清单、manifest、日志、result；原始提交可 checkout，不能用当前默认配置直接覆盖旧记录。

当前缺口是官方源码、真实预训练模型和正式数据验证。缓存完整拷贝具有线性成本，当前 API 只支持顺序、无 padding 的 full-attention Llama，不具备生产服务或真实模型复现资格。

下一里程碑建议按计划 Day 4 准备合成算术与 GSM8K 数据：记录来源/版本，按原题分组切分，测试扰动泄漏，严格区分最终答案、启发式伪目标和真实步骤标签。并行补齐官方代码访问后的审计；未补齐前不宣称官方 baseline 已复现。先冻结数据协议与预训练模型 revision，再另行评估训练成本。
