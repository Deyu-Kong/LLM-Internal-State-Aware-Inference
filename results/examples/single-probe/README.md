# 单 Probe 原始检查产物

- `single-probe-{cpu,gpu0}-001`：提交 7c1e195；Probe 参数在不同设备初始化，同 seed 但权重不同，仅代表各自的机制检查，不能用于 CPU/GPU 数值对照。
- `single-probe-{cpu,gpu0}-002`：提交 02c65dc；统一在 CPU 初始化后迁移，是最终验收产物。
- `environment-packages.json`：项目环境的包版本快照，不包含凭据或 pip 配置。

修正原因、测量边界及运行命令见 `reports/single_probe.md` 和 `docs/otv_research.md`。没有优化器更新或任务学习结论；不要将初始化分数当作已校准置信度。
