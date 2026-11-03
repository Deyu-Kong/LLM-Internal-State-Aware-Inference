# PRM800K 扩展训练与独立校准/测试

## 固定方案

在 phase1_train 可用候选中排除先前 pilot 选过的全部 250 道原题，以及与 ProcessBench 四个分区规范化题目精确相同的题目。按 seed 42 的原题哈希排序，固定 100 道测试、100 道校准，其余用于训练。所有三组原题互不重合；不使用官方 PRM800K test 或 PRMBench。

每个原题最多 8 个候选。完整 tokenization 后整条排除超过 512 token 的样本；在剩余候选中，每题优先选择同一前缀下的正负配对，按固定哈希排序，余额再从其他候选填充。配对规则使用标注标签构造队列，因此测试分布不是自然频率分布；它用于受控配对评价。测试标签不进入优化、校准、阈值选择或模型选择。保留根级分组、全部长度排除清单及样本 hash，输入仍仅包含题目、历史和当前候选。

模型沿用固定版本 SmolLM2-135M-Instruct，冻结主干，单 Probe、rank-2 LoRA、logits Head；三个模型种子 42/43/44，数据 seed 始终 42。单 GPU batch 1、梯度累积 4、AdamW 3e-4、至多 300 次更新，训练部分最多 1200 秒。类别权重只从训练集计算。各次运行均校验原始数据/模型哈希、冻结权重和主 KV、正常生成 logits 不变，并验证 checkpoint 重载。

训练完后，使用独立校准集拟合正斜率 Platt 变换，并仅在校准集选择 F1 阈值；然后一次性在独立测试集报告 AUROC、average precision、Brier、固定及校准阈值 F1。该概率仅针对配对优先的选择分布，不能直接当作自然部署分布下校准概率。同一测试集还报告只用训练集拟合的长度/位置基线和同前缀配对排序准确率。

三个种子均按预先固定规则评估，不根据测试指标选择 checkpoint 或继续调参；三次使用同一测试原题，因此不是三个独立测试集。结果仅限外部轨迹重放与该 135M 模型，不是官方 OTV 复现，也不证明自然生成过程检测能力。ProcessBench 检查只能排除规范化精确重复，无法排除改写或预训练污染；PRMBench 未进行重合审计。

## 命令与资源

估计每种子 5–10 分钟、单卡显存低于 4GB、输出低于 100MB。实际以 manifest 为准。需先准备固定哈希 PRM800K/模型资产以及 `configs/datasets/processbench_audit.yaml` 指定的四个 ProcessBench JSON；网络下载与复制只用于 CPU。数据和模型保存在 `.gitignore` 覆盖的 `data/raw/`。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 .venv/bin/python -m pytest -q
OMP_NUM_THREADS=2 .venv/bin/python scripts/train_probe_expanded.py --config configs/wp1_probe/prm800k_expanded.yaml --preflight
OMP_NUM_THREADS=2 .venv/bin/python scripts/train_probe_expanded.py --config configs/wp1_probe/prm800k_expanded.yaml --gpu 0
OMP_NUM_THREADS=2 .venv/bin/python scripts/train_probe_expanded.py --config configs/wp1_probe/prm800k_expanded_seed43.yaml --gpu 0
OMP_NUM_THREADS=2 .venv/bin/python scripts/train_probe_expanded.py --config configs/wp1_probe/prm800k_expanded_seed44.yaml --gpu 0
```

训练仅在检查指定物理 GPU 0/1 空闲后启动，并在导入 torch 前通过 UUID 限制可见设备。每次独立检查，若被占用则退出，不等待、不抢占。命令必须从仓库根目录运行；结果目录不可覆盖，复现需新 experiment_id。

每次运行保存完整配置、版本、数据来源和哈希、原题分组、长度排除列表、模型/tokenizer 版本、训练日志、校准参数、测试预测、指标、仅 Probe 参数的 checkpoint 与 manifest。含题目的样本和 checkpoint 留在被忽略的运行目录；归档只保存不含题目文本的审计材料。
