# OTV：论文审阅与单 Probe 验证边界

## 来源与已确认方法

依据 [EMNLP 2026 论文](https://aclanthology.org/2026.emnlp-main.519.pdf) §3.1–3.4、§4.1、附录 A/B/C；版本和 PDF SHA256 见 `data/manifests/otv_sources.json`。

| 项目 | 论文描述 |
|---|---|
| Probe | 验证专用 `[ToT]`，输出连续分数，不解码“正确/错误”词 |
| LoRA | 主模型冻结，Q/K/V 增量仅在验证模式启用 |
| KV / attention | 单 token 读取各层前缀 KV，并可 attention 到自身；不重算前缀 |
| Head | 最终 hidden state 经三层 perceptron 得到标量 |
| 监督 | 最终结果标签 y 构造启发式 `c_t = 0.5 + (y - 0.5)t/T` |
| Loss | response token 的 MSE；prompt/query 不计入 |
| 推理 | 从指定前缀发起一次验证；可选聚合分数进行筛选 |
| 并行训练 | 各 Probe 只能看对应历史和自身，不能互看 |

这些伪标签不是真实步骤标签，也不等于经验证的概率校准。论文开销不能直接套用于本项目。

## 官方代码可用性

2026-11-03，论文给出的 [GitHub 仓库](https://github.com/Multimedia-Analytics-Laboratory/OTV) 及其公开 API 均返回 404，`git ls-remote` 也未获取到版本。可能未公开、已移动或需要权限，不能据此断言仓库永久不存在。本次没有获得源码、commit、训练配置或 checkpoint，**不宣称完成官方代码研究或 OTV 结果复现**。

待官方代码可访问后逐项核对：特殊 token 的初始化/保存、embedding 是否训练、head 隐层宽度与激活、LoRA rank/缩放/初始化、位置编码、padding、精度、并行 mask、缓存隔离、数据生成和评测聚合。不得把下述本地选择写成作者官方实现。

## 本项目独立实现选择

实现目标是验证一个合法旁路的可微分计算路径，尚无任务学习能力。本地选择如下：

- 延用随机微型 Llama，2 层、hidden 32、4 个 query head、2 个 KV head；使用 eager attention 和 float32。
- rank=2，Q/K/V LoRA 的 A 使用 Kaiming 初始化、B 为零，增量缩放为 1，无 adapter dropout。首次 backward 中 A 梯度为零是零初始化 B 的预期结果；测试另外设非零 B 检查 A 梯度和生成隔离。
- Probe 向量从冻结 embedding 表的 token 1 拷贝后独立训练，通过 inputs_embeds 输入，不扩展 tokenizer/vocabulary。它不是自然语言 token 的语义实现。
- Head 选择 `32 → 32 → 32 → 1`，中间 SiLU，最终 sigmoid。只训练 LoRA、独立 Probe 向量和 head。
- 为使 CPU/GPU 诊断共享初始化，所有参数统一在 CPU 初始化再迁移设备；同一 seed 不保证不同设备随机数流相同，也不要求跨设备位级一致。
- 主模型始终 eval；上下文门控在异常退出时也恢复。生成路径不启用 LoRA；仅支持顺序调用、无 padding 的等长 batch，不提供 serving 并发保证。
- 每次 Probe 对普通 DynamicCache 的所有 K/V 在原设备 detach+clone，Probe 新 KV 只追加到副本。该策略复制成本为 O(前缀长度)，不是目标高效部署方案。拒绝 static、滑窗或 offload cache。
- 本轮只发起一次单位置 MSE backward，不运行 optimizer、不生成 checkpoint、不开展数据集训练或评估。合成 outcome 仅作梯度夹具，不能解释为真实数学正确性。

## 验收与测量

CPU 完整测试通过后，才运行受 GPU guard 保护的 GPU 检查。预估单次短检查小于 2 分钟、显存小于 1GB、产物小于 1MB；只允许空闲的物理 GPU 0 或 1。源模型参数必须不变，主 KV 长度/内容不变，后续三步生成的 logits/KV 不变。独立测试还以整段 prefix+Probe 重算作为零增量参考，核对 position 和 attention 到历史及自身。

记录 forward/backward 时间；3 次 warmup 后采样 10 次：缓存拷贝、私有分支 forward、包含拷贝的端到端 Probe。GPU 使用同步计时。微型合成诊断不代表真实模型延迟、吞吐降幅或 WP1 有效性。

## 运行

使用已有项目 `.venv`（依赖与 `docs/state_capture.md` 相同，无新增 GPU 包）：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 .venv/bin/python -m pytest -q
OMP_NUM_THREADS=2 .venv/bin/python scripts/check_single_probe.py --config configs/wp1_probe/single_probe_cpu.yaml
OMP_NUM_THREADS=2 .venv/bin/python scripts/check_single_probe.py --config configs/wp1_probe/single_probe_gpu.yaml --gpu 0
```

每次换新的 experiment_id，保留旧结果；脚本从仓库根目录运行。配置、完整模型与 Probe 配置、数据及来源清单、日志、manifest 和原始计时样本存入运行目录。

## 后续决策

本轮只判断计算路径、梯度和隔离是否成立，不触发 WP1 科学假设的 Go/No-Go。先补齐官方实现审计；在此之前可独立准备按原题分组的合成算术/GSM8K 数据协议，严格区分 outcome-derived pseudo-confidence 与真实过程标签。小型预训练模型与正式训练另设里程碑，不直接放大本原型。
