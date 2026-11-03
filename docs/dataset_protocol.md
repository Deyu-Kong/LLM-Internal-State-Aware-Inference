# 数据规范

按原始题目 ID 分组划分 train、validation、test_id、test_ood，同题扰动不得跨分区。数据清单应记录来源、版本、处理参数、种子、分区、样本数量与文件哈希。最终答案标签不能当作逐步正确性标签。

本阶段未引入真实数据集。smoke 的 dataset_manifest.json 明确标记未使用数据，不提供虚构样本或标签。
