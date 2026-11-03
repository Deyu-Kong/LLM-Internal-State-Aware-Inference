# PRM800K 诊断产物

`diagnostic-full.json` 包含固定验证集的三个随机种子、预训练/随机骨干控制、按原题的探索性重采样、加权 BCE 分数反变换及组内配对统计。`processbench-overlap.json` 是精确规范化题目核对；并未测试 benchmark，也不能发现语义近重复。

四个新增运行目录保存配置、manifest、日志、清单和验证预测，不包含题目原文或模型权重。原始 checkpoint 与带题目数据存放在被 Git 忽略的运行目录。原始 seed 42 运行见 `results/examples/prm800k-pilot/`。解释和复现信息见 `reports/prm800k_diagnostics.md`。
