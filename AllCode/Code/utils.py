"""
工具函数模块
包含：
1. AverageMeter - 用于跟踪统计平均值
2. accuracy - 计算准确率
"""

import torch

class AverageMeter:
    """
    用于计算和存储平均值和当前值
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


def accuracy(output, target, topk=(1,)):
    """
    计算topk准确率
    Args:
        output: 模型输出 [batch_size, class_count]
        target: 目标类别 [batch_size]
        topk: 要计算的top-k准确率
    Returns:
        res: 包含topk准确率的列表
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res 