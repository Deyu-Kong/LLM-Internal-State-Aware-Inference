# PRM800K 小子集训练设计

## 目标与范围

首轮只验证真实人工步骤标签能否驱动单 Probe 学习，并与未训练、常数先验、长度/位置基线比较。使用官方 PRM800K 的 phase1_train，不使用官方 test、ProcessBench 或 PRMBench 的评测标签。跨数据集原题重合尚未审计，后续报告这些基准成绩前必须先去重并重新确定训练集；本试验不声称独立基准性能。

数据来源：[openai/prm800k](https://github.com/openai/prm800k)，固定 `7ecc794703b2877f63226f2477a49b34f9b25163`。原始 phase1_train 的 LFS SHA256 和长度固定在代码、训练配置中，非 LFS 指针。约 7.9MB、949 条记录。

## 数据适配

每个样本是“题目 + 实际选中/人工修正历史 + 一个候选步骤”，目标为该候选人工标签。+1 为正，-1 为负，0 不作为监督。0 步骤可留在已选历史中，但不作为正标签。排除 QC/screening、give-up/bad-problem、flagged/未知标签和空文本；同一输入存在正负冲突时去除。

候选同属于一个前缀，不能把兄弟候选拼接成历史。遇到被选中的负标签/不确定步骤后停止；遇到 unflagged 的人工修正对象可延伸，但 corrected_rating=-1 不延伸。人工修正只用于前缀，不把未知 rating 转为训练正标签。也不使用未来 pre_generated_steps 或 reference answer。

审计后按规范化原题分组，固定 seed 选 200 道 train、50 道 validation，每题最多 4 个候选。先分组再 tokenization；超过 512 token 的整个样本排除，不截断待验证步骤。输出排除清单和实际保留原题数。两类都必须出现在两个分区。步骤样本数量、类别比、长度和来源 ID 全部记录。

模型输入通过 problem/prefix_steps/candidate_step 显式白名单构造。label、source metadata、ground-truth answer、未来步骤均不可作为输入。

## 模型与优化

选择 [SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)，固定 revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`；同一 revision 的 tokenizer。此模型很小，数学能力有限，不是研究计划中的最终强 baseline。权重的官方 LFS SHA256、其他文件的 Git blob ID 和长度固定到模型配置，离线加载前重新校验。

冻结 fp32 的 Llama 主模型，保持 eval。Q/K/V rank-2 LoRA、独立 Probe embedding、三层 Head 可训练。正常前缀计算关闭 LoRA；Probe 在独立 detached KV 克隆上验证一个位置，返回 logits。以人工标签执行 BCEWithLogitsLoss；训练用仅由 train 类别比计算的 pos_weight，评估报告不加权 BCE。保持原有 sigmoid/MSE 机制检查接口作为默认兼容模式。

AdamW，lr=3e-4，weight_decay=0，梯度裁剪 1.0，100 optimizer steps，每步累计 4 个不填充的单样本，effective batch=4。固定 seed=42，按本地 RNG 打乱循环；不是正式多种子实验。所有随机新增参数在 CPU 初始化后迁移设备。最多 900 秒的运行预算，达到时停止更新并报告实际步数。无自动超参数搜索或根据 validation 选 checkpoint。

在固定训练子集（最多 96 个样本）和全部保留 validation 上记录训练前/后 AUROC、average precision、F1@0.5、accuracy、Brier、BCE。AP 为阈值分组的 average precision，不将它冒充梯形积分 AUPRC。常数先验仅由 train 估计，长度/位置 logistic baseline 的均值方差及拟合也只使用 train。

## 正确性与产物

先运行 CPU 单元/集成测试及预训练模型 preflight（一次 backward、零 optimizer step）。正式 GPU 运行再次检查空闲，只接受物理 GPU 0/1，按 UUID 设置 CUDA_VISIBLE_DEVICES 后才导入 torch，不启用多卡/device_map=auto。

训练前后校验全部冻结参数 SHA256、保存的 generator KV 内容，以及相同前缀下正常解码 logits 精确不变。可训练参数必须实际改变。保存仅包含 LoRA/Probe/Head 的 checkpoint，不重复保存主模型；用 weights_only=True 重载并验证首个 validation 分数。

run 目录包含配置、模型/原始文件 SHA、数据清单、selected IDs、长度排除记录、实际 train/validation 样本、逐步日志、训练前后预测、checkpoint、结果与 manifest。样本与 checkpoint 不提交 Git，归档只保存无原题文本的清单、统计、预测与日志。

预估：单卡 GPU 0，不超过 4GB，训练评估约 5–15 分钟，产物小于 100MB；模型约 269MB，原始数据约 8MB。实际以记录为准，不将下载时间算成训练时间，不将 peak allocated 称为进程总显存。

## 运行

在可联网主机获取公开固定资产（可重复执行，会校验已有文件，不覆盖损坏缓存）：

```bash
uv run --locked python scripts/fetch_pilot_assets.py
```

若 a800 不能直连 Hugging Face，在本地主机获取后将同一 data/raw 路径复制到项目目录，训练仍逐文件校验。只使用项目独立环境：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 .venv/bin/python -m pytest -q
OMP_NUM_THREADS=2 .venv/bin/python scripts/train_probe.py --config configs/wp1_probe/prm800k_pilot.yaml --preflight
OMP_NUM_THREADS=2 .venv/bin/python scripts/train_probe.py --config configs/wp1_probe/prm800k_pilot.yaml --gpu 0
```

运行 ID 不可覆盖；preflight 使用独立的 `-preflight` 后缀。复现实验使用新 ID。短训练结果只能说明这个配置的局部验证能力；外部文本重放得到的是目标模型的新 KV，不是原始生成模型 KV，也不是 on-policy 成功证据。
