# CPU 数据准备与防泄漏验证

## 完成范围

新增数据配置、固定来源下载、GSM8K 参考解答解析、合成算术轨迹生成、分组切分、独立标签回放和跨机器哈希验证。没有训练、模型生成或 GPU 计算。基于干净代码提交 `1a10e3a38df60f1563cba4b5f24753ad4e3e9777` 在本地与 a800 分别生成。

这一阶段完成研究计划 Day 4 的基础数据管线和 1,000 条可复现样本；尚未收集真实 LLM reasoning traces，不能把参考解答或合成数据当成真实模型的 OTV 校准数据。

## 机制验证为何不是训练

前一里程碑的 Single Probe 验证仅执行 forward → MSE → backward：backward 计算梯度，不直接改变参数。没有 optimizer.step，全部参数逐项检查保持相等。39 项测试检验可微分性、冻结参数、KV 分支隔离、续写不变、attention/position 正确。2,817 个参数具备可训练计算路径，不意味着已经学会验证。随机分数约 0.508、单合成 MSE 及微型模型延迟均不是任务质量结论。若之后执行优化器更新并反复拟合数据，才进入训练阶段。

## 数据与分区

GSM8K 固定官方原始仓库 commit `3101c7d5072418e28b9008a6636bde82a006892c`，train/test/LICENSE 均检查 SHA256。原始 7,473 条 train 与 1,319 条 test 全部解析成功；官方 train 按原题划分 90/10，官方 test 原样保留为 test_id。实际未发现跨官方 train/test 的规范化题目重合，隔离排除数为 0。

合成部分为 200 道 ID 原题 + 50 道 OOD 原题，每题 4 变体，合计 1,000 条；四种变体各 250 条。最终正确 500 条、错误 500 条；有历史局部错误 750 条（其中显式纠正后最终正确 250 条），无错误 250 条。

| 数据集 | train | validation | test_id | test_ood | 总计 |
|---|---:|---:|---:|---:|---:|
| synthetic | 640 | 80 | 80 | 200 | 1,000 |
| GSM8K reference | 6,726 | 747 | 1,319 | 0 | 8,792 |
| 合计 | 7,366 | 827 | 1,399 | 200 | 9,792 |

全局审计有 9,042 道原题/规范化问题，9,792 个唯一 sample_id，同题变体没有跨分区。OOD 仅为合成数据的更长运算链和不同数值范围，不能外推为真实语言/任务 OOD。

## 标签语义

合成数据分别校验当前算术、规定依赖、局部合法性、无错参考数值、最终答案和 first-error。错误后的局部运算可以正确但继续传播错误；纠错后的最终正确不抹掉历史错误。所有错误目前固定在第二步，存在明显结构捷径，只适合接口与受控分析，正式训练前需设计更丰富的错误位置与类型。

GSM8K 是参考解答，按非空行作启发式步骤划分，不是生成轨迹或人工过程正确性标注。所有步骤过程标签为 null、first_error_status=unknown；尚无模型预测，final_answer_correct 为 null。没有 tokenizer，所以不伪造逐 token OTV pseudo-confidence。训练输入适配器尚未实现，后续必须防止把 variant/label/reference answer/first-error 等监督元数据输入模型。

## 代码与测试

- `src/state_inference/data/config.py`：数据任务独立的严格 YAML 配置。
- `sources.py`：不可变 revision/hash、有限大小下载、原子写入和损坏缓存拒绝覆盖。
- `splits.py`：规范化问题 fingerprint、按 root 分组切分和泄漏审计。
- `gsm8k.py`：Decimal 答案解析、参考步骤保存、官方 test 保留及重合训练题隔离。
- `synthetic.py`：四类受控轨迹、局部/依赖/最终标签及独立结构化回放。
- `preparation.py`、`scripts/prepare_data.py`：配置、清单、日志、结果、源许可与失败状态记录。
- `tests/unit/test_data.py`、`tests/integration/test_data_preparation.py`：新增 24 项 CPU 测试，涵盖重排稳定性、同题变体泄漏、官方 test 保护、损坏标签/缓存拒绝、离线准备、防覆盖和复现哈希。

实际验证：

```text
本地 uv run --locked pytest -q: 42 passed, 2 skipped（缺少 torch 的两个模型模块）
a800 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 .venv/bin/python -m pytest -q:
63 passed in 4.27s
```

本地 Python 3.10.12 与 a800 Python 3.12.13 的整个 dataset manifest 一致，包括 8 个分区文件、所有原始文件和分组/排除清单的哈希。本地处理约 1.027 秒，a800 约 0.581 秒（使用已验证原始缓存，不含首次网络下载）；这些是数据准备耗时，不是模型实验性能。

## 产物与复现

两台机器上都保留：`data/raw/gsm8k/<revision>/`、`data/processed/data-preparation-001/`、`results/data-preparation/data-preparation-001/`。原始文件约 4.8MiB，处理后数据及分组清单约 23MiB，不提交全量数据到 Git。归档的配置、manifest、日志、结果、MIT 许可和合成示例位于 `results/examples/data-preparation/{local,a800}/`；跨机器检查结果为 `cross_host_verification.json`。

新环境从仓库根目录执行：

```bash
uv sync --locked --group dev
uv run --locked python scripts/prepare_data.py --config configs/datasets/prepare.yaml --download
```

已有运行目录时先用新 experiment_id，禁止覆盖。a800 可用已有 `.venv/bin/python` 替代 uv；数据任务不导入 torch，本轮没有使用 GPU 0/1 或任何其他 GPU。处理与原始缓存目录遵守 .gitignore，不涉及共享环境修改。

## 局限与下一步

规范化去重不能发现语义近重复或预训练污染；步骤边界是行分割，不是完整推理结构。合成标签是精确受控执行，不代表自然模型错误分布。真实模型/tokenizer revisions、真实生成、任务质量评估、训练和高效 KV 分支仍未完成；官方 OTV 代码访问缺口仍在。

建议下一里程碑先选择兼容且可访问的小型预训练模型，固定模型/tokenizer revision，增加严格排除监督字段的输入适配器，然后在 train/validation 小子集收集少量真实轨迹，核验最终答案和来源。参考解答不冒充模型输出，测试集保持锁定。开始任何训练前再确定标签目标、估算 GPU0/1 的时间/显存/存储，并单独制定实验范围。
