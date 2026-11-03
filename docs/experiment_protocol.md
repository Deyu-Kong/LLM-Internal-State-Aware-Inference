# 实验与复现规范

每个正式实验必须保留配置、日志、数据清单、复现命令和技术报告。记录 Git SHA/dirty、模型及 revision、tokenizer revision、adapter revision、数据版本、随机种子、CUDA/PyTorch/vLLM 版本、GPU 信息、batch size、输入输出长度、采样参数及实际运行时间。未采集的信息必须明确标记，不能虚构。

正式研究使用多随机种子、强基线、消融、泛化、失败案例与质量/开销联合评价。模型质量与系统吞吐不可混为一谈。性能测试固定硬件、精度、batch、warmup，记录 CPU/GPU 拓扑与并发占用。修复/重生成和状态采集均计入端到端成本。

目前仅支持 CPU 基础设施 smoke；manifest 中的空模型与 GPU 字段不满足正式 GPU 实验要求。正式实验功能应在相应里程碑补齐，不能直接用 smoke 代替研究结果。

每次方法变更记录原因。Probe 必须隔离主生成 KV；动态投机长度必须保留标准 target verification/residual sampling。三个 WP 独立推进，按各自 Go/No-Go 判断是否继续。长 GPU 作业前估计时间、显存和存储。
