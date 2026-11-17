# node32 主动 KV Probe 重训 pilot

状态：2026-11-17，三个对照均完成训练、独立测试、checkpoint 重载验证和结果留档。

用户目标：检验 KV Cache 是否包含能被主动 Probe 读出的最终正确性预测信息。不是复现论文全部数值，也不验证过程错误定位或提前终止。

## 结果与结论

供 Git 版本审阅的轻量原始统计：[pilot-results.json](pilot-results.json)。完整轨迹与权重在下文列出的本地/远程产物目录，不纳入 Git。

**在本轮 1.5B pilot 中，主动 Probe 能读出与最终正确性相关的信号，但没有显示出优于参数匹配 hidden-state 读出的优势；早期检测能力仍不稳定。** AUROC 不是分类准确率，随机排序对应约 0.5。

GSM8K 独立测试：32 道 Probe 训练未见题目、128 条轨迹，其中正确 59 条、错误 69 条。

| 方法 | 可训练参数 | 固定 64 响应 token AUROC | 响应末端单 Probe AUROC |
|---|---:|---:|---:|
| 未训练 OTV | — | 0.460 | 0.543 |
| 完整 OTV | 1,697,537 | 0.644 | 0.721 |
| 无 LoRA：query embedding + head | 206,593 | 0.649 | 0.664 |
| 参数匹配 hidden-state head | 1,697,189 | 0.689 | 0.727 |

OTV 的题级 bootstrap 95% CI：固定 64 token 为 [0.519, 0.750]，末端为 [0.627, 0.812]。末端 OTV−hidden 的**配对** AUROC 差值 CI 为 [-0.109, 0.099]；固定 64 token 为 [-0.137, 0.047]。均跨 0，不能宣称主动 Probe 更好，也不能宣称两者等效。

完整 OTV 验证 MSE：0.08703 → 0.08210 → 0.07961。三组均按最低验证 MSE 选择第 3 个 epoch，没有按测试分数调整训练设置。

### 题目难度、长度与缓存内容

- 固定 64 token 时长度相同，但不保证答案尚未出现。
- GSM8K 的 22 道混合正负结果题，OTV 题内平均 AUROC：固定 64 token 为 0.667，95% CI [0.496, 0.822]；末端为 0.697，CI [0.549, 0.833]。这支持晚期存在轨迹相关读出信号，早期仍不确定。
- 每题固定选一正一负、同为 64 token：22 对中 15 对给正确轨迹更高分。清零响应部分 K/V 后同题差异为 0。它确认分数使用响应缓存内容；清零是分布外干预，不是语义错误识别的因果证明。
- 事后数值字面泄漏诊断：排除前 64 token 已出现标准或模型最终答案数值的轨迹，剩余 120 条；OTV AUROC 0.644，hidden 0.701。可能漏掉数字词或语义等价表达，也可能过度排除题目已给数字或步骤编号，不是严格的“答案首次出现前”证明。

### MATH 域内测试

13 道题、39 条轨迹：OTV 固定 64 token AUROC 为 0.497，末端为 0.781；hidden 对应 0.538、0.743。只有 4 道题同时有正负轨迹，OTV 末端题内 AUROC 为 0.396，样本不足，不给题内 CI。较高 pooled 分数可能混有题目难度信号，不能外推为稳定的早期轨迹检测能力。

## 已完成的实现检查

- node32 / 主机名 amax，使用物理 GPU 1，按 UUID 限制可见设备。
- 官方测试和 pilot 标签/指标测试共 23 项通过。
- Qwen3-4B-Instruct-2507 的真实模型检查通过：冻结基础参数与原 KV 不变，验证路径有梯度，并行/独立 Probe 在 BF16 容差内一致。
- 上述检查使用未训练验证器和合成监督，不能作为检测有效性的结果。
- 训练结束后从保存的 OTV checkpoint 重新加载，7 个前缀分数与留档值逐一相同，最大绝对误差为 0。
- 所有 AUROC 另以正负样本逐对比较独立复算，一致；本地数据 SHA256 与冻结 manifest 一致。

## 本轮设置

- 当前训练模型：`/data1/kdy/LLMs/Qwen2.5-Coder-1.5B-Instruct`；属于 standalone 支持的全注意力 Qwen2 架构。**不是论文 Qwen3 数值复现。**
- 基于 standalone OTV，Q/K/V LoRA rank 8、alpha 16；head 128→64→1；MSE、线性 outcome-derived targets。
- 每条完整可解析轨迹用于所有响应前缀的 dense 监督。训练 3 epochs，AdamW，学习率 1e-4，梯度累积 4；仅用验证集 MSE 选 checkpoint。
- 对照：完整 OTV；无 LoRA、只训练 truth embedding 和 head；近似参数匹配的三层 hidden-state head。
- Hidden head 为 1536→1060→64→1，与完整 OTV 可训练参数相差约 0.02%；OTV head 为 1536→128→64→1。两者均无 dropout。
- 单轨迹 batch size 1，梯度裁剪 1，weight decay 0；OTV query chunk size 128。三组监督、轨迹和优化更新预算一致。
- 单 seed 42。相比论文，数据规模、有效 batch size、head 大小和 standalone 边界定义不同。
- 远程工作目录：`/data1/kdy/Project/otv-probe-pilot`。

## 数据与调整依据

4B 的首次补充批次也未产生足够错误（21 条完成且可解析，全部正确），因此保存数据并停止该采样，转用更小的 1.5B 模型完成首轮可判别训练。没有使用人工改错轨迹。

1.5B 在原有题目与补充题目上重新生成，temperature 1、top-p .95、每题 4 条、最多 768 新 token；训练/验证/测试题目在生成前固定。最终保留情况如下：

| 集合 | 独立题目 | 轨迹 | 正确 | 错误 |
|---|---:|---:|---:|---:|
| 训练 | 52 | 176 | 72 | 104 |
| 验证 | 14 | 49 | 25 | 24 |
| GSM8K 测试 | 32 | 128 | 59 | 69 |
| MATH 域内测试 | 13 | 39 | 13 | 26 |

生成题目数与保留题目数不同，因为部分题目的所有响应均截断或无法解析。丢弃数量和全部原始输出保存在 generation summary 和 `*-all.jsonl` 中。四个正式 split 的 question_id 无交叉。

以下为此前 4B 分支的准备记录，不是最终训练规模：

初始采样 32 道 MetaMathQA 训练题、8 道验证题、32 道 GSM8K 测试题，每题 4 次采样，temperature 1、top-p .95，最多 768 新 token。初始训练部分 128 条中 100 条完成可解析，其中只有 1 条错误，另 28 条截断。这不足以支撑可靠的判别实验。

因此在训练前增加未使用过的 MATH 题，按参考解答长度选取较长一半，再以 seed 43 抽取 32/8/16 道训练/验证/域内测试题；每题 8 条轨迹，最多 1536 新 token。此调整基于类别覆盖与题目审查，不使用任何 Probe 测试分数。

审查发现部分 SV/FOBAR 反向构题的条件不合理，因此最终组装时按来源类型排除这些样本。补充题只选 MATH_AnsAug/MATH_Rephrased。所有正式 split 按原题隔离。

只选标准答案可严格解析为整数、小数或有理数的题。要求模型给出最终 boxed 答案，按 Fraction 精确等价标注；截断和解析失败单独统计、排除。该筛选会改变样本分布，结果不能外推到所有数学题或未完成轨迹。参考答案匹配也不等同于人工验证每个推理步骤正确。

## 评估

- 前缀 25%/50%/75%/100%：离线诊断，百分比依赖完整响应长度。
- 固定 32/64 个响应 token：报告覆盖样本和类别数量，控制进度与总长度捷径。
- 最后一个 boxed 标记之前：不代表数值答案首次出现之前，不能据此证明早期错误定位。
- AUROC、average precision、错误类 average precision；按题目 bootstrap 的置信区间；模型间用配对题目 bootstrap。
- 同题存在正确与错误轨迹时，额外报告题内 AUROC，控制题目难度。
- 单类测试集 AUROC 必须记为未定义，不得记作 1 或 0.5。小样本宽置信区间不得解释为方法已证实有效或无效。
- 单模型题级 bootstrap 1000 次，配对差值与题内均值各 2000 次，seed 42；独立正/负题目不足 5 不给对应 CI。不同前缀为探索性分析，未做多重比较校正。
- 本文末端是无 EOS 的最后响应边界单次 Probe，不是论文末 100 token 均值或 best-of-N accuracy。
- 按题目切分只隔离 Probe 训练集，不能排除基础模型预训练见过公开题目。单 seed、小数据和有理数答案筛选限制了结论范围。

脚本：`run_probe_pilot.py`、`assemble_probe_pilot.py`、`analyze_active_probe_pilot.py`。远程版本与原始生成版本均保留，最终报告需记录实际执行版本与数据 manifest。

## 可复用权重与产物

远程根目录 `/data1/kdy/Project/otv-probe-pilot`：

- 完整 OTV：`runs/small-otv-001/best/`。
- 无 LoRA Probe：`runs/small-no-lora-001/best/`。
- Hidden baseline：`runs/small-hidden-001/best-head.pt`，结构见同目录 `pilot-config.json`。
- 数据与排除记录：`runs/small-traces-001/`；逐轨迹预测、指标在各模型运行目录。
- 配对统计：`runs/small-comparison-001.json`；环境、模型/数据校验和、执行代码在 `provenance/`；命令在 `otv-train-three.sh`。

科研仓库内的轻量产物见 [artifacts](../artifacts/) 和 [pilot-results.json](pilot-results.json)。完整轨迹及权重保存在上述 node32 目录；基础模型未纳入 Git。Checkpoint 引用 node32 的基础模型路径。

在 node32 确认授权 GPU 空闲后，使用项目环境和 `PYTHONPATH=src`，可通过 `python -m otv.score --checkpoint ../runs/small-otv-001/best ...` 加载。该 CLI 默认末 100 token 的 trace_score 与本文固定前缀指标不同；重跑训练必须换新的输出目录，保留原结果。

下一轮若用于科研结论，应回到目标推理模型、补足自然正负轨迹和至少 3 个 seed，继续保留参数匹配 hidden-state 和题内对照。本轮适合探索可读出性，尚不足以宣称主动 Probe 的独立优势。
