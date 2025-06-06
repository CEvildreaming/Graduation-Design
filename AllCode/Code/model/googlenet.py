"""
GoogleNet模型定义
基于torchvision实现的GoogleNet(Inception v1)模型
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional

def create_googlenet(num_classes: int = 8, pretrained: bool = False) -> nn.Module:
    """
    创建GoogleNet (Inception v1)模型
    
    参数:
        num_classes: 分类类别数量
        pretrained: 是否使用预训练权重
    
    返回:
        配置好的GoogleNet模型
    """
    if pretrained:
        model = models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT)
    else:
        model = models.googlenet(weights=None)
    
    # 修改最后一层以匹配类别数
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    return model 