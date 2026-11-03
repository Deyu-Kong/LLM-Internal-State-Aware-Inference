# 数据准备与标签规范（v1）

## 范围与来源

本里程碑仅在 CPU 准备数据，不生成 LLM 轨迹、不训练模型。GSM8K 来自 [官方原始仓库](https://github.com/openai/grade-school-math)，固定 commit `3101c7d5072418e28b9008a6636bde82a006892c`；原始 train/test 为 JSONL，最终数值答案在 `####` 之后。[原始说明](https://github.com/openai/grade-school-math/blob/3101c7d5072418e28b9008a6636bde82a006892c/README.md)

原始数据及 [MIT 许可](https://github.com/openai/grade-school-math/blob/3101c7d5072418e28b9008a6636bde82a006892c/LICENSE) 均固定 SHA256，见 `configs/datasets/prepare.yaml`。缓存已有但哈希不符时拒绝覆盖；缺失时仅在指定 `--download` 后获取固定 URL。单文件限制 8MiB，下载校验通过后原子发布。不会用 main/master 作为运行时版本。

## 切分与泄漏检测

所有记录都有 `sample_id`、`root_id`、`split`、来源与标签 provenance。

- 原题 fingerprint：题目 Unicode NFKC、空白折叠、casefold 后 SHA256。合成原题另外以结构化问题规范计算 ID。
- 分组分配：先对唯一原题按 `SHA256(seed:root_id)` 排序，按比例取边界。固定原题集合、协议版本与 seed 时，打乱输入/增加同题变体不会改变切分。同题所有变体留在同一分区。
- 合成 ID 原题按 80/10/10 分到 train/validation/test_id；50 道独立 OOD 原题全部 test_ood。
- GSM8K 官方 train 只按原题分 train/validation（90/10），官方 test 全部保留在 test_id。GSM8K 不伪造 OOD 分区，test_ood 文件为空。
- 若官方 train 存在规范化后与官方 test 相同的问题，隔离对应 train 记录并输出排除清单，保留官方 test；不静默跨分区保留重复题。
- 输出完整 `split_assignments.jsonl` 并固定哈希。新增原题可能改变排序边界，必须作为新版本/新运行保存，不把旧 split 随意扩展。
- 全局审计重复 sample_id、跨分区 root_id 和规范化相同题目，以及官方 test 是否进入训练。该审计不能检测语义近重复、模型预训练污染或未提供 parent ID 的外部改写。后续扰动必须显式继承 root_id。

## 合成算术与精确标签

默认 200 道 ID 原题 + 50 道 OOD 原题，每题 4 条轨迹，共 1,000 条。ID 是 3 步 add/mul/sub，初值 10–50；OOD 是 5 步 add/mul/sub/mul/add，初值 100–200。OOD 同时改变长度与数值尺度，是受控分布偏移，不代表自然语言泛化。

四类变体：正确；第二步算术错误并向后传播；第二步引用过期的 x0 造成依赖错误；第二步出错后立即显式纠正，再继续正确推导。纠正会显式标注 `corrects_step`，允许重新赋值同一变量；其他步骤不得跳过或重复。固定错误位置是本初版的局限，不能作为真实错误定位训练分布。

每一步单独记录：

| 字段 | 含义 |
|---|---|
| arithmetic_valid | 按实际声明的输入值，当前运算与结果是否相等 |
| dependency_valid | 引用变量、operation、operand 是否符合题目规定 |
| local_process_valid | 前两者均成立；仅表示当前这一步的局部合法性 |
| reference_value_match | 当前输出数值是否与无错误参考执行一致 |
| corrects_step | 显式纠正的先前步骤序号，否则 null |

索引从 1 开始。`first_error_step` 是最早局部算术/依赖错误；`first_error_status` 区分 known_error、known_no_error、unknown。最终答案与参考结果精确比较得到 `final_answer_correct`。

错误后的步骤可能算术与引用均合法，但继续传播错误中间值，故不能把 local_process_valid 当作“整个前缀无错”。显式纠错轨迹可最终正确但历史含错，不能用最终标签覆盖所有步骤。独立结构化回放校验每步标签、纠正契约、最终结果及最早错误，无 eval/代码执行。

这些样本是规则控制的结构化算术，不是模型生成、不是真实过程标注集或 OTV 训练复现；没有 tokenizer 时不构造逐 token 伪置信度。保留 `pseudo_confidence_targets: null`。

## GSM8K 参考答案

保留原始 question、reference_solution 和来源行号。以非空行启发式划分参考步骤，同时保留原文和去除 `<<...>>` 注释后的文本；不执行其中表达式。答案用 Decimal 规范化，支持整数、带逗号数值和普通小数，拒绝额外单位、表达式与模糊格式。

`trace_origin=dataset_reference_solution`；参考步骤不等同于已经验证的局部标签。arithmetic_valid/dependency_valid/local_process_valid/reference_value_match 全为 null，first_error_status=unknown，final_answer/final_answer_correct 为 null，因为还没有待评模型的输出。reference_answer 是数据集提供的答案，不能冒充模型预测。

GSM8K 不能只凭最终答案制作真实逐步标签。后续模型轨迹需额外保存模型/tokenizer revision、采样参数、原题 ID、最终预测与验证方法；OTV outcome-derived 伪置信度与真实步骤标签必须分字段保存。

## 运行与产物

从仓库根目录执行：

```bash
uv sync --locked --group dev
uv run --locked pytest -q
uv run --locked python scripts/prepare_data.py --config configs/datasets/prepare.yaml --download
```

已有校验通过的缓存后可省略 `--download`。默认运行 ID 为 data-preparation-001，每次再运行需使用新的 experiment_id，禁止覆盖已有 processed/result 目录。`gsm8k: null` 可用于离线合成-only 准备（仍须使用数据配置完整字段）。

- `data/raw/gsm8k/<revision>/`：原始文件及 LICENSE；不提交全量原始数据。
- `data/processed/<experiment_id>/<dataset>/<split>.jsonl`：统一字段的样本与标签。
- `data/processed/<experiment_id>/split_assignments.jsonl` 和 `excluded_training_overlaps.jsonl`：分组与排除记录。
- `results/data-preparation/<experiment_id>/`：完整配置、manifest、数据清单、版本/seed/耗时日志、result、合成示例、许可副本。

manifest 固定原始 revision/hash、处理参数、分区计数、所有数据文件 hash/bytes、split assignment hash、标签政策和 Git 版本。失败也记录 failed manifest；不能把部分输出作为成功数据。归档的示例输出见 `results/examples/data-preparation/`。

后续训练输入只允许明确选择 question/trace 文本等观察字段；严禁把 variant、label、reference answer、first_error、split 等监督/元数据整体序列化进模型输入。必须在单独的训练数据适配器中检查此项，本轮不提供训练适配器。
