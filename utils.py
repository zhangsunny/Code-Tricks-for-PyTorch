"""
@Date   : 2020/03/10
@Author : zhangsunny
@Comment: 个人定制老代码
"""
import random
import time
import numpy as np
import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score


def set_seed(seed=0):
    """
    固定随机数种子，保持实验的一致性
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def gen_data_loader(*tensors, batch_size=64, shuffle=True):
    """
    将tensor数据包装成DataLoader，方便迭代使用
    """
    dataset = TensorDataset(*tensors)
    data_loader = DataLoader(dataset, shuffle=shuffle, batch_size=batch_size)
    return data_loader


def flat_accuracy(preds, labels):
    """
    根据分类模型输出概率，计算准确率
    """
    if type(preds) == torch.Tensor:
        if preds.device.type != 'cpu':
            preds = preds.detach().cpu()
            labels = labels.detach().cpu()
        preds = preds.numpy()
        labels = labels.numpy()
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def flat_f1(preds, labels, average='macro'):
    assert average in ['macro', 'micro'], "只支持macro和micro"
    if type(preds) == torch.Tensor:
        if preds.device.type != 'cpu':
            preds = preds.detach().cpu()
            labels = labels.detach().cpu()
        preds = preds.numpy()
        labels = labels.numpy()
    pred_flat = np.argmax(preds, axis=1).flatten()
    score = f1_score(labels.flatten(), pred_flat, average=average)
    return score


def parse_path(file_path):
    """
    将文件路径解析为目录、文件名、后缀
    """
    file_dir, file_name = os.path.split(file_path)
    name, suffix = os.path.splitext(file_name)
    ret = dict([('dir', file_dir), ('name', name), ('suffix', suffix)])
    return ret


def clockit(func):
    """
    装饰器，计算函数的执行时间
    """
    def wrapper(*args, **kwargs):
        tic = time.time()
        func(*args, **kwargs)
        toc = time.time()
        print('{}-cost:{:.2f} s'.format(func.__name__, toc-tic))
    return wrapper


class MetricCounter():
    """
    记录每个batch的指标，并返回整体数据上的平均值
    """
    def __init__(self):
        self.metric_sum, self.count = 0, 0
        self.metric_list, self.count_list = [], []

    def update(self, metric, count):
        self.metric_sum += metric * count
        self.count += count
        self.metric_list.append(metric)
        self.count_list.append(count)

    def reset(self):
        self.metric_sum, self.count = 0, 0
        self.metric_list, self.count_list = [], []

    def get_avg_metric(self):
        return self.metric_sum / self.count
