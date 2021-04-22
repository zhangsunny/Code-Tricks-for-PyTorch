# Code-Tricks-for-PyTorch
Some useful code blocks and functions for personal purpose
一些个人常用的老代码，备份一下

## utils.py

- def set_seed: 固定随机数种子，保持实验的一致性
- def gen_data_loader: 将tensor数据包装成DataLoader，方便迭代使用
- def flat_accuracy: 根据多分类模型输出的概率，计算准确率
- def flat_f1: 根据多分类模型输出的概率，计算Macro-F1和Micro-F1
- def parse_path: 将文件路径解析为目录、文件名、后缀
- def clockit: 装饰器，计算函数的执行时间
- class MetricCounter: 记录每个batch的指标，并返回整体数据上的平均值(训练时总是搞错批数据和完整数据的结果)
