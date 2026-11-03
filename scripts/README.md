# 脚本

- `state-inference smoke`：基础设施检查。
- `check_state_capture.py`：微型模型只读状态采集。
- `check_single_probe.py`：单 Probe forward/backward，无优化器更新。
- `prepare_data.py`：CPU 准备固定版本 GSM8K 与受控合成算术，见 `docs/dataset_protocol.md`。

训练、真实模型轨迹生成与 benchmark 脚本留待对应里程碑实现。
