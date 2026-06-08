# transformer-from-scratch
一个从零实现的 42M 参数中文小语言模型，在数十本网络小说上完成预训练。

## 核心特点
- 纯 PyTorch 实现，模型定义、数据加载、训练循环均从零编写
- 使用 Qwen2.5 tokenizer（词表大小 151936），彻底解决 `[UNK]` 问题
- 流式数据加载器：`IterableDataset` + shuffle buffer，支持无限量文本输入
- 混合精度训练（AMP, GradScaler），RTX 4090 利用率 95%+
- 完整的 checkpoint 保存与断点续训支持

- ## 模型架构

| 组件 | 配置 |
|------|------|
| 层数 | 4 |
| 隐藏维度 | 256 |
| 注意力头数 | 4 |
| 最大序列长度 | 256 |
| 词表大小 | 151936 |
| 总参数量 | ~42M |

## 训练数据

数十本中文网络小说，总规模约 2 亿 tokens。

数据加载使用滑动窗口（stride=128, max_len=256），配合 10000 样本容量的 shuffle buffer 提升数据随机性。

## 训练配置

- 硬件：NVIDIA RTX 4090 (24GB)   autodl
- batch_size：16
- 优化器：Adam, lr=3e-4
- 混合精度：FP16 (GradScaler)
- 训练时长：单个 epoch 约 1 小时（约 10 万步
