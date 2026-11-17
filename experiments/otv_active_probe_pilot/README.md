# Active KV Probe：独立 OTV 重训 pilot

本目录归档 2026-11-17 在 node32 完成的实验：用主动验证 token 读取冻结模型的 KV Cache，并与无 LoRA Probe、参数匹配 hidden-state head 比较。**模型是 Qwen2.5-Coder-1.5B-Instruct，不是论文 Qwen3 数值复现。**

GSM8K 的 32 道独立题目、128 条轨迹：

| 方法 | 固定 64 token AUROC | 响应末端 AUROC |
|---|---:|---:|
| 完整 OTV | 0.644 | 0.721 |
| 无 LoRA Probe | 0.649 | 0.664 |
| 参数匹配 hidden-state | 0.689 | 0.727 |

初步支持正确性信号可被主动读出，但没有证明优于被动 hidden-state。差值置信区间跨 0；单 seed、小数据、早期检测不稳定。AUROC 不是分类准确率。

- [完整报告及结论边界](docs/retraining_pilot_zh.md)
- [版本化统计、配置与各模型指标](docs/pilot-results.json)
- [轻量产物、数据 manifest、环境与执行代码版本](artifacts/)
- [作者原始 README](UPSTREAM_README.md)、[源码来源和校验和](UPSTREAM_PROVENANCE.json)

`src/otv/` 来自作者 standalone 实现，保留上游来源；本项目新增 pilot 脚本、对照及归档，不将作者方法写成本项目创新。没有修改主项目 `src/state_inference`、WP1/WP2/WP3 方法或根环境依赖。

## 环境与运行

本目录是单独的 Python 包，Python 3.11+。在独立环境中运行，不向根项目 uv 环境添加 GPU 依赖：

```bash
cd experiments/otv_active_probe_pilot
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest -q
```

真实训练需要本地基础模型与 outcome-labeled traces。训练入口为 `scripts/run_probe_pilot.py`，支持 `generate`、`benchmark`、`train`、`hidden`。示例执行命令与真实路径见 `artifacts/provenance/otv-train-three.sh`，真实执行的两个脚本版本也在 provenance 内保留。**重跑须使用新的输出目录**，不能覆盖归档；在 node32 按项目 GPU 规则重新检查占用并按 UUID 限制可见设备。

完整 OTV 权重在 node32：

```text
/data1/kdy/Project/otv-probe-pilot/runs/small-otv-001/best/
```

基础模型在 `/data1/kdy/LLMs/Qwen2.5-Coder-1.5B-Instruct`。权重和完整原始轨迹未提交 Git；其配置、SHA256、统计与排除记录已归档。checkpoint 已通过真实模型重载检查，7 个前缀分数与留档值相同。

归档的启动脚本保留原始 node32 目录和脚本文件名，是实验命令证据；在新目录重跑时应将执行脚本映射为本目录 `scripts/run_probe_pilot.py`，并修改数据、模型和输出路径。
