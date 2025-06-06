"""
EfficientNet模型定义
基于torchvision实现的EfficientNet系列模型
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional

def create_efficientnet_b0(num_classes: int = 8, pretrained: bool = False) -> nn.Module:
    """
    创建EfficientNet-B0模型
    
    参数:
        num_classes: 分类类别数量
        pretrained: 是否使用预训练权重
    
    返回:
        配置好的EfficientNet-B0模型
    """
    if pretrained:
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    else:
        model = models.efficientnet_b0(weights=None)
    
    # 修改分类器以匹配类别数
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features, num_classes)
    )
    
    return model

def create_efficientnet_b1(num_classes: int = 8, pretrained: bool = False) -> nn.Module:
    """
    创建EfficientNet-B1模型
    
    参数:
        num_classes: 分类类别数量
        pretrained: 是否使用预训练权重
    
    返回:
        配置好的EfficientNet-B1模型
    """
    if pretrained:
        model = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.DEFAULT)
    else:
        model = models.efficientnet_b1(weights=None)
    
    # 修改分类器以匹配类别数
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features, num_classes)
    )
    
    return model

def create_efficientnet_b2(num_classes: int = 8, pretrained: bool = False) -> nn.Module:
    """
    创建EfficientNet-B2模型
    
    参数:
        num_classes: 分类类别数量
        pretrained: 是否使用预训练权重
    
    返回:
        配置好的EfficientNet-B2模型
    """
    if pretrained:
        model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)
    else:
        model = models.efficientnet_b2(weights=None)
    
    # 修改分类器以匹配类别数
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, num_classes)
    )
    
    return model 