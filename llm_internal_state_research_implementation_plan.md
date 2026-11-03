
# LLM Internal-State-Aware Inference
## Research and Implementation Plan

> Version: 1.0
> Date: 2026-11-03
> Status: Research proposal and implementation roadmap
> Primary language: Chinese

---

# 1. 研究背景与总体目标

## 1.1 论文题目

中文：

**基于内部状态主动感知的大语言模型推理验证与计算优化方法研究**

英文：

**Research on Active Internal-State Probing for Verification and Computation Optimization in Large Language Model Inference**

## 1.2 核心研究问题

大语言模型在推理过程中会产生大量内部状态，包括：

- KV Cache
- Hidden States
- Attention 信息
- Logits 及概率分布
- 中间层特征
- 推理过程中的状态变化

这些内部状态不仅服务于下一 Token 的生成，还可能包含有关以下问题的信息：

1. 当前推理是否正确？
2. 推理过程是否已经出现错误？
3. 哪些步骤需要进一步验证？
4. 哪些计算可以提前执行、减少或跳过？
5. 如何根据内部状态动态优化推理引擎？

论文核心目标：

**研究 LLM 内部状态的主动读出方法，以及如何将内部状态转化为推理验证与计算控制能力。**

研究分为三个相对独立的工作包：

| 工作包 | 研究方向 | 核心创新 |
|---|---|---|
| WP1 | Multi-View KV Probing | 多视角主动探针与错误检测 |
| WP2 | Budget-Aware Verification | 基于风险与成本的自适应验证 |
| WP3 | State-Aware Speculative Decoding | 基于内部状态的动态投机解码 |

三个工作包应共享实验平台，但不能在技术上完全相互依赖。

---

# 2. 总体技术架构

## 2.1 核心架构

```text
                     LLM Inference
                           |
                           v
                  Internal States
              /         |          \
             /          |           \
       KV Cache     Hidden States    Logits
             \          |           /
              \         |          /
                       v
               State Observation
                       |
            +----------+----------+
            |          |          |
            v          v          v
           WP1        WP2        WP3
        KV Probing  Verification  Speculation
            |       Scheduling    Control
            |          |          |
            +----------+----------+
                       |
                       v
             Quality / Cost / Speed
```

## 2.2 基本研究原则

1. 研究重点不是简单训练一个二分类器。
2. 优先利用推理过程中已经存在的内部状态。
3. 控制额外计算与存储开销。
4. 通过严格的基线和消融实验验证创新性。
5. 将模型质量指标和系统性能指标结合。
6. 不以模型在内部状态上存在相关性直接推导因果关系。
7. 避免引入不必要的模型训练成本。
8. 优先复现现有工作，然后提出具体改进。
9. 不提前假设方法一定有效，设置明确的 Go/No-Go 检查点。

---

# 3. WP1：Multi-View Active KV Probing

## 3.1 研究目标

研究是否可以通过多个具有不同观察视角的主动 KV Probe，比单一 Probe 更准确地判断：

- 推理整体正确性
- 当前步骤正确性
- 推理步骤之间的依赖一致性
- 最早错误步骤的位置

工作名称：

**MV-KVP: Multi-View KV Probing**

## 3.2 文献基础

主要参考：

1. One-Token Verification for Reasoning Correctness Estimation (OTV)
   https://aclanthology.org/2026.emnlp-main.519/

2. OTV GitHub
   https://github.com/Multimedia-Analytics-Laboratory/OTV

3. ProcessBench
   https://aclanthology.org/2025.acl-long.50/

4. PRMBench
   https://aclanthology.org/2025.acl-long.1230/

首先必须完整研究 OTV 的：

- Probe Token 构造
- KV Cache 读取方式
- Attention 计算方式
- LoRA 适配方案
- 训练标签
- Loss Function
- 推理调用流程
- 验证开销

## 3.3 基础模型

设 LLM 已生成前缀：

```text
Prompt + Reasoning Step 1 + ... + Reasoning Step t
```

得到其历史 KV Cache：

```text
KV_t
```

在不改变主生成过程的旁路分支中，插入辅助 Probe Token。

Probe 通过现有 KV Cache 获取历史信息，输出对当前推理状态的判断。

主生成过程与 Probe 过程必须隔离：

```text
Generator KV -----> Normal Decoding
       |
       +----------> Probe Branch
```

Probe 的额外 KV 不应回写 Generator 主缓存。

## 3.4 多视角设计

设计三类 Probe。

### Probe A：Global Probe

观察完整历史。

预测：

- 当前前缀对应的最终推理成功概率
- 全局状态风险

### Probe B：Local Probe

主要观察最近的推理步骤。

预测：

- 当前步骤是否出错
- 局部推导是否合理

可尝试不同局部窗口：

```text
Last 1 Step
Last 2 Steps
Last 4 Steps
```

### Probe C：Dependency Probe

关注当前步骤与历史前提的关联。

预测：

- 是否存在前提使用错误
- 是否存在中间结果不一致
- 是否引用了错误的先前结论

初版可使用结构化算术任务明确依赖关系。

复杂自然语言推理中的依赖发现可以留作后续改进。

注意：

**限制 KV 可见性并不等于从模型计算历史中因果删除被遮蔽的文本。**

后续 Token 的 KV 已经包含此前上下文的信息。

因此需要谨慎解释 Mask 消融结果。

## 3.5 训练方案

初期：

- 冻结主模型
- 使用少量 LoRA 参数
- 为不同 Probe 配置独立输出 Head
- 比较共享 LoRA 和独立 LoRA

训练目标：

```text
L = λ1 * L_global
  + λ2 * L_local
  + λ3 * L_dependency
```

其中：

- `L_global`：整体正确性预测
- `L_local`：步骤错误检测
- `L_dependency`：依赖一致性检测

必须确保标签与预测目标匹配。

不能将最终答案正确性标签直接当作每一步的真实正确性标签。

## 3.6 数据集

第一阶段：

- GSM8K
- Synthetic Arithmetic Reasoning
- ProcessBench
- PRMBench

第二阶段：

- MATH
- 不同错误类型的构造数据
- 跨任务验证集

数据生成：

1. 收集模型生成的 Reasoning Traces。
2. 将推理文本划分为步骤。
3. 标记最终答案正确性。
4. 为可验证任务标记逐步正确性。
5. 构建受控错误样本。
6. 记录最早错误步骤。
7. 根据原始题目 ID 分割训练/验证/测试集。

避免同一原题的不同扰动版本跨数据集泄漏。

## 3.7 Baselines

必须包含：

- Single OTV-style Probe
- 2 Probes，完全相同观察视角
- 4 Probes，完全相同观察视角
- 2 Probes，不同观察视角
- 4 Probes，不同观察视角
- 多视角 + 不同监督目标
- 参数量匹配的大型 Single Probe
- 多个独立 Probe 模型的 Ensemble
- 基于 Logits 的简单置信度方法
- Length/Position-only Baselines

重点回答：

**性能改善来自观察视角、监督目标，还是仅仅来自额外参数和计算量？**

## 3.8 评价指标

模型质量：

- AUROC
- AUPRC
- F1
- Brier Score
- ECE
- First-error Localization Accuracy
- Best-of-N Selection Accuracy

系统成本：

- Probe Forward Latency
- Extra GPU Memory
- Tokens/s Degradation
- GPU Utilization
- Total Verification Overhead

分析不同 Probe 的预测相关性、错误召回重叠和互补性。

## 3.9 验收标准

最低要求：

- 成功复现单 Probe 方法
- 实现至少两种独立观察视角
- Probe 不改变主生成过程
- 完成同等成本 Baseline 对比
- 有错误检测与定位结果
- 有额外计算开销报告

研究假设：

多视角机制在合理成本下提供单探针以外的有效信息。

如果只是增加 Probe 数量而没有额外信息收益，不继续盲目扩大 Probe 数量。

---

# 4. WP2：Budget-Aware Adaptive Verification

## 4.1 研究目标

研究如何利用内部状态估计的风险信息，决定：

- 是否验证
- 何时验证
- 使用哪一级验证器
- 是否需要重新生成或修复

工作名称：

**BAV: Budget-Aware Verification**

核心问题：

**在有限的验证计算预算下，如何提高最终正确率？**

或者：

**在达到同样正确率时，如何降低总验证成本？**

## 4.2 验证动作

设计以下动作：

| Action | Description |
|---|---|
| Skip | 不进行验证 |
| Cheap | Logits/Entropy/轻量检查 |
| Single Probe | 单个 OTV-style Probe |
| Multi Probe | MV-KVP |
| External | 外部工具、测试执行或强模型验证 |

每种动作对应：

```text
Verification Cost
Detection Probability
False Positive Rate
Repair Probability
```

需要实测不同动作的代价，不假设 Probe 一定比所有工具验证更便宜。

## 4.3 决策机制

状态输入：

```text
State Features
Probe Confidence
Current Context Length
Historical Error Signals
Remaining Verification Budget
```

输出：

```text
Verification Action
```

初版可使用：

```text
score(action)
    = expected_error_reduction(action)
    - lambda * expected_cost(action)
```

对于多个候选动作，选择预算约束下收益最高的动作。

首先使用简单规则与校准风险模型，暂时不引入复杂强化学习。

## 4.4 技术路线

### Stage 1：Verification Profiling

收集不同任务、不同步骤上执行各验证动作的结果：

- 是否检测到错误
- 验证时间
- 显存使用
- 是否误报
- 是否能够纠正错误
- 纠错成本

### Stage 2：Offline Oracle

假设事先知道不同动作的实际效果，构造预算约束下的理想调度上界。

其作用：

判断当前任务是否真的存在优化空间。

如果最佳 Oracle 与固定阈值几乎没有差异，说明该任务不适合复杂调度。

### Stage 3：Online Scheduler

实现在线调度：

```python
action = scheduler.select(
    state_features,
    verification_budget,
    historical_statistics
)
```

支持：

- Risk Calibration
- Cost Profiling
- Verification Fallback
- Budget Tracking
- Repair Triggering

### Stage 4：End-to-End Integration

构建：

```text
LLM Generation
      |
      v
Risk Estimation
      |
      v
Verification Scheduler
      |
      v
Selected Verifier
      |
      v
Accept / Repair / Regenerate
```

必须将 Repair/Regeneration 的开销纳入总成本。

## 4.5 实验负载

优先：

1. Mathematical Reasoning
2. Code Generation
3. Multi-Step Agent Tasks

特别关注：

- 部分步骤容易出错
- 不同验证器具有真实不同的成本
- 错误检测后存在有效修复操作

如果某个任务的精确验证极其便宜，全量验证可能就是最优方法，不应强行证明自适应调度有效。

## 4.6 Baselines

- No Verification
- Verify All
- Random Verification
- Fixed Frequency
- Fixed Confidence Threshold
- Logits Entropy
- OTV Threshold
- MV-KVP Threshold
- Budgeted Oracle

## 4.7 评价指标

- Final Task Accuracy
- Accuracy per GPU-second
- Verification Calls
- Verification Cost
- End-to-End Latency
- GPU-ms
- P95 / P99 Latency
- Accuracy-Budget Pareto Curve
- Repair Success Rate

核心成果：

**在相同预算下提高最终任务正确率，或者在相同正确率下减少总计算开销。**

## 4.8 独立性与风险控制

WP2 不应依赖 WP1 必须成功。

如果 MV-KVP 不够可靠，可以使用：

- 原始 OTV
- Logits-based Signals
- 已有验证器

WP2 的独立创新重点是：

**风险感知、异质成本建模和验证资源分配。**

---

# 5. WP3：Internal-State-Aware Speculative Decoding

## 5.1 研究目标

研究能否利用 Target Model 在正常推理和 Draft Verification 中已经计算出来的内部状态，动态控制 Speculative Decoding。

工作名称：

**ISAS: Internal-State-Aware Speculation**

核心问题：

**当前 Hidden State 能否预测后续 Draft Token 的接受情况，从而选择更合适的 Draft Length？**

## 5.2 基础方案

采用：

- Target Model：Qwen3-8B 或兼容模型
- Draft Model：对应的 EAGLE-3 Speculator
- Runtime：vLLM

参考：

https://docs.vllm.ai/projects/speculators/en/stable/user_guide/algorithms/eagle3/

首先必须复现官方 EAGLE-3 的固定 Draft Length 推理。

## 5.3 核心机制

传统固定投机长度：

```text
Draft 4 Tokens
    |
Target Verification
    |
Accept / Reject
    |
Draft 4 Tokens
```

改进为：

```text
Previous Verification
       |
       v
Target Hidden Features
       |
       v
Acceptance Predictor
       |
       v
Adaptive Draft Length
       |
       v
Standard Target Verification
```

候选 Draft Length：

```python
K = [0, 2, 4, 6, 8]
```

具体候选集合以 Runtime 的实际支持为准。

## 5.4 状态特征

优先选择额外开销低的特征。

### Acceptance History

- 上一轮 Accepted Tokens
- 最近 N 轮 Acceptance Ratio
- Reject Position
- Acceptance Variance

### Hidden State Features

- 选定层的 Hidden State Norm
- Layer-to-Layer Similarity
- Consecutive Hidden State Shift
- 轻量低维投影

### Context Features

- Current Context Length
- Token Entropy
- Batch Size
- Recent Generation Statistics

注意：

**只使用当前决策时已经计算完成的信息。**

不能使用未来一轮 Target Verification 的结果训练当前轮控制器。

## 5.5 决策模型

目标预测：

```text
P(accepted_length >= j | current_state)
```

或直接预测：

```text
Expected Throughput(draft_length)
```

选择：

```python
best_k = argmax(
    expected_progress(k) / measured_cost(k)
)
```

成本包括：

- Draft Computation
- Target Verification
- Controller Overhead

初版使用轻量回归或分桶估计模型即可。

研究创新不在于预测器复杂，而在于：

**使用低开销内部状态控制真实投机计算，并在严格基线下证明收益。**

## 5.6 Lossless 约束

必须完整保留标准 Target Verification 与必要的 Residual Sampling。

动态改变 Draft Length 不允许：

- 跳过 Target Verification
- 擅自接受 Token
- 修改原始 Target Distribution
- 放宽 Acceptance Criteria

在正确实现的前提下，应保持相对于 Target 的采样分布。

但这不意味着不同执行方式在同一随机种子下必然生成完全相同的 Token 序列。

## 5.7 研究步骤

### Stage 1：Fixed EAGLE Baseline

收集：

- Draft Length
- Accepted Length
- Draft Time
- Verification Time
- TPOT
- TPS
- Context Length

### Stage 2：History-Only Controller

实现一个仅基于历史接受率的控制器。

这是必须比较的强基线。

### Stage 3：Hidden-State Predictor

从 Target 计算产生的状态中提取低成本特征。

训练接受长度估计器。

验证：

- 是否优于 History-only
- 是否具有跨上下文泛化能力
- 是否存在额外状态读取开销

### Stage 4：Runtime Integration

在 vLLM 投机解码循环中插入动态 Draft Length 决策。

只更改长度选择，不更改 Target Verification 核心语义。

### Stage 5：Benchmark

比较：

- No Speculative Decoding
- Fixed EAGLE-3
- History-only Adaptive
- Entropy-only Adaptive
- State-only Adaptive
- State + History Adaptive

## 5.8 评价指标

- Tokens/s
- Time Per Output Token
- Time To First Token
- P95 Latency
- Acceptance Length
- Draft/Target Compute Time
- Controller Overhead
- Peak GPU Memory
- Throughput Under Different Batch Sizes

实验维度：

- Short Context
- Medium Context
- Long Context
- Low Concurrency
- High Concurrency
- Different Sampling Temperatures

## 5.9 创新性风险

必须研究现有工作：

- KnapSpec
  https://proceedings.mlr.press/v306/cha26a.html

- ASPIRE
  https://arxiv.org/abs/2609.17943

这些工作已经研究过相关的状态预测或动态投机机制。

因此，不能直接把“Hidden State 预测接受长度”宣称为全新创新。

真正的研究空间应通过以下对照确认：

1. 内部状态是否提供超出历史接受率的额外信息？
2. 能否避免额外 Target Forward？
3. 长上下文下是否更有价值？
4. 实际吞吐收益是否大于状态提取开销？
5. 不同负载下是否能够动态调节？

## 5.10 Go/No-Go

如果 Hidden State 对 Acceptance 的预测性能没有超越 History-only：

- 首先检查特征、层和采样时机。
- 再测试更轻量的状态统计。
- 若仍无增量价值，停止继续扩大预测器规模。

备选研究方向：

**面向多 Probe 验证的高效 KV Query 批量处理与运行时优化。**

但转换方向前必须重新检索文献，并确认可以形成独立的系统贡献。

---

# 6. 代码项目结构

推荐仓库名：

`llm-state-aware-inference`

```text
llm-state-aware-inference/
├── README.md
├── AGENTS.md
├── pyproject.toml
├── configs/
│   ├── models/
│   ├── datasets/
│   ├── wp1_probe/
│   ├── wp2_verification/
│   └── wp3_speculation/
├── data/
│   ├── raw/
│   ├── processed/
│   ├── manifests/
│   └── examples/
├── src/
│   └── state_inference/
│       ├── models/
│       ├── state_capture/
│       ├── kv_probe/
│       ├── verifiers/
│       ├── scheduling/
│       ├── speculative/
│       ├── runtime/
│       └── metrics/
├── scripts/
│   ├── prepare_data.py
│   ├── generate_traces.py
│   ├── train_probe.py
│   ├── eval_probe.py
│   ├── build_verification_traces.py
│   ├── eval_scheduler.py
│   ├── collect_speculative_traces.py
│   └── benchmark_speculation.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── regression/
├── results/
├── reports/
└── docs/
    ├── experiment_protocol.md
    ├── dataset_protocol.md
    ├── model_interfaces.md
    └── thesis_outline.md
```

## 6.1 基础技术栈

模型研究：

- Python
- PyTorch
- Transformers
- PEFT
- Hugging Face Datasets

系统研究：

- vLLM
- CUDA
- PyTorch Profiler
- NVIDIA Nsight（必要时）

实验工具：

- pytest
- YAML Configurations
- JSONL / Parquet
- Pandas
- Matplotlib
- Weights & Biases 或本地日志

先保证实验正确与可复现。

只有确实存在运行时瓶颈时，再考虑集成 mini-infer 或实现 C++/CUDA 算子。

---

# 7. 硬件与实验资源

已知硬件：

- 4 × NVIDIA RTX A6000 48GB
- RTX 3090
- 可申请的额外 GPU 资源
- 双路 Intel Xeon 服务器

推荐安排：

GPU 0：

- WP1 OTV Baseline
- Probe Training

GPU 1：

- WP1 Ablation
- Data Generation

GPU 2：

- WP2 Verification Experiments

GPU 3：

- WP3 EAGLE-3 Experiments

以上仅是任务分配建议，不应同时开展相互干扰的正式性能 Benchmark。

性能实验期间：

- 固定硬件配置
- 固定模型精度
- 固定 Batch Size
- 记录 Warmup
- 记录 CPU/GPU 拓扑
- 避免其他任务占用同一 GPU
- 保持驱动、CUDA 和 Runtime 版本可追溯

优先使用单卡 4B/8B 模型。

只有必要时才进行 Tensor Parallelism。

---

# 8. 统一实验与复现规范

每个实验必须记录：

```text
Experiment ID
Git Commit SHA
Model Name and Revision
Tokenizer Revision
Dataset Version
Adapter Revision
Random Seed
CUDA Version
PyTorch Version
vLLM Version
GPU Information
Batch Size
Input Length
Output Length
Sampling Parameters
Measured Runtime
```

所有数据按原题分组切分：

```text
train
validation
test_id
test_ood
```

实验结果需要：

1. 多随机种子
2. 明确的 Baseline
3. 消融实验
4. 性能开销分析
5. 失败案例
6. 泛化测试
7. 可复现的运行命令

严禁虚构性能数字。

每次修改实验方法，都要记录变更原因。

---

# 9. 20 周实施计划

| Week | 工作内容 | 阶段产出 |
|---|---|---|
| W1 | 创建项目、环境、状态采集 | 可运行基础框架 |
| W2 | OTV 复现、准备数据 | Single Probe |
| W3 | 实现 2/4 Probes | 初步实验数据 |
| W4 | 同质 Probe 消融 | WP1 Go/No-Go |
| W5 | 不同 KV Mask、监督目标 | Multi-View Probe |
| W6 | 多任务训练、错误定位 | WP1 Ablation |
| W7 | WP1 全面测试、写作 | WP1 完整研究结果 |
| W8 | Verification Profiling | 多动作成本数据 |
| W9 | Offline Oracle | WP2 Go/No-Go |
| W10 | 实现自适应验证调度 | WP2 Scheduler |
| W11 | WP2 质量—成本实验 | Pareto 曲线 |
| W12 | WP2 章节、EAGLE 采样日志 | WP3 Go/No-Go |
| W13 | 动态 Draft Length 接口 | WP3 Controller |
| W14 | Hidden-State Controller 集成 | WP3 POC |
| W15 | 长短上下文性能测试 | TPS/TPOT 结果 |
| W16 | WP3 消融与系统优化 | WP3 完整结果 |
| W17 | 跨任务泛化验证 | OOD Evaluation |
| W18 | 综合性能与统计分析 | 全部实验报告 |
| W19 | 论文整合写作 | 完整初稿 |
| W20 | 复现、修改与答辩准备 | 完整论文与复现包 |

从第二周开始同步撰写方法和实验记录。

---

# 10. 第一周详细开发计划

## Day 1：项目初始化

任务：

- 创建目录结构
- 固定依赖
- 配置 pytest
- 创建 YAML 配置机制
- 实现日志与 Manifest
- 编写 README

验收：

在新环境中可以运行 Smoke Test。

## Day 2：内部状态采集

任务：

- 加载一个小型 Causal LM
- 获取 Hidden States
- 获取 Logits
- 读取 Past Key Values
- 支持指定层与指定位置采集

验收：

得到正确形状的状态数据。

启用状态采集不会改变主生成分支。

## Day 3：OTV 研究与最小复现

任务：

- 阅读官方论文及代码
- 实现 Single Probe Forward
- 配置 Trainable Parameters
- 验证 Backward
- 测量单 Probe 开销

验收：

能够完成一次有效的 Forward/Backward。

## Day 4：数据集处理

任务：

- 下载 GSM8K
- 解析 Reasoning Steps
- 构造 Synthetic Arithmetic
- 制作标签与 Manifest
- 编写防数据泄漏测试

验收：

生成 1000 条可复现实验样本。

## Day 5：Single Probe Baseline

任务：

- 开展小规模训练
- 实现模型评价
- 输出 AUROC/AUPRC/F1
- 测量运行时间

验收：

产生第一份 Baseline Report。

## Day 6：Dual Probe

任务：

- 实现双 Probe
- 保证独立 Attention
- 保证不影响 Generator KV
- 测试两个 Probe 的计算开销

验收：

Dual Probe 可以独立运行。

## Day 7：总结

输出：

`reports/week01.md`

内容：

- 已完成工作
- 运行命令
- 目前实验结果
- 遇到的问题
- 与论文目标的差距
- 下一周任务

---

# 11. 里程碑与风险管理

## Milestone 1：Week 4

目标：

验证多 Probe 方案是否存在足够研究空间。

继续条件：

多视角设计有明确的增量信息，或显示出值得深入的错误类型差异。

失败处理：

转向不同监督目标、结构化依赖或高效多 Probe 运行时。

## Milestone 2：Week 9

目标：

验证多级验证动作是否存在真实可利用的成本与效果差异。

继续条件：

Offline Oracle 相对简单策略存在可观的质量—成本优化空间。

失败处理：

更换负载或验证动作，不盲目增加调度器复杂性。

## Milestone 3：Week 12

目标：

验证 Hidden State 对接受长度是否提供增量预测信息。

继续条件：

相对 History-only 存在可靠的增量收益，且状态采集开销可接受。

失败处理：

停止复杂状态控制器开发，评估独立系统优化方向。

---

# 12. 论文建议目录

## 第一章 绪论

- 研究背景与意义
- LLM 内部状态的研究价值
- 推理验证研究现状
- 投机解码与计算控制研究现状
- 研究问题与主要贡献
- 论文结构

## 第二章 技术背景与统一实验平台

- Transformer 内部状态
- KV Cache 与 Hidden States
- 主动 Probe 方法
- Process Verification
- Speculative Decoding
- 统一平台设计

## 第三章 多视角主动 KV 探针方法

- 问题定义
- Multi-View Probe 设计
- 训练方法
- 错误检测与定位
- 消融实验
- 性能分析

## 第四章 基于内部状态的预算感知验证方法

- 验证资源分配问题
- 不同验证动作的成本建模
- 风险估计与校准
- 自适应调度算法
- 端到端实验
- 质量—成本分析

## 第五章 内部状态驱动的自适应投机解码

- 问题定义
- 接受长度预测
- 状态特征设计
- 动态 Draft Length 控制
- Lossless Verification
- 系统实现与性能评估

## 第六章 总结与展望

- 研究成果总结
- 三项贡献的联系
- 方法局限性
- 未来研究方向

---

# 13. Codex 实施规则

## Core Instructions

1. Read this plan completely before making changes.

2. Implement one milestone at a time.

3. Begin with the repository scaffold and reproducibility infrastructure.

4. Do not implement all three research packages simultaneously.

5. Prefer small, testable changes over large rewrites.

6. Never invent experimental results.

7. Every result must have a corresponding configuration, log, dataset manifest and reproducible command.

8. Ensure auxiliary Probe computations never alter the main generator KV state.

9. Ensure dynamic speculative decoding preserves the standard target-verification semantics.

10. Compare against strong baselines before claiming improvements.

11. Track model, runtime and dataset revisions.

12. Profile actual end-to-end runtime and GPU memory.

13. Every milestone must include tests and a written technical report.

14. Before long GPU runs, estimate the runtime, VRAM consumption and output storage.

15. At each Go/No-Go checkpoint, report whether the original research hypothesis is supported.

---

# 14. Codex First Task

## Objective

Implement the repository scaffold and reproducibility infrastructure.

## Scope

Implement ONLY:

- Project structure
- Dependency management
- Configuration loading
- Experiment manifests
- CLI entry points
- Logging
- Basic tests
- README

Do not yet implement the research methods.

## Acceptance Criteria

1. The repository is importable as a Python package.
2. A smoke test runs successfully.
3. An example configuration can be loaded.
4. An experiment manifest is generated.
5. Logs include versions, random seed and runtime information.
6. README includes exact setup and execution commands.

## Required Deliverables

- Updated codebase
- Passing tests
- README
- Example configuration
- Example experiment output
- Summary of modified files
- Known blockers
- Recommended next task

---

# 15. 最终研究成功标准

本项目成功不等于三个模块简单运行成功。

三个研究章节应分别具备：

1. 清晰的问题定义
2. 现有研究基线
3. 可解释的技术创新
4. 可执行实现
5. 严格的对照实验
6. 消融研究
7. 系统开销分析
8. 失败案例
9. 适用边界
10. 可复现的实验材料

最终希望得到一套统一的研究结论：

**大语言模型内部状态不仅可以参与 Token 生成，还可以通过主动读出与在线决策机制，为推理正确性判断、验证资源分配以及推理计算优化提供额外价值。**

这一结论的成立范围和实际收益必须由实验决定。
