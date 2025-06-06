"""
Swin Transformer模型定义
基于torchvision实现的Swin Transformer模型
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional

def create_swin_t(num_classes: int = 8, pretrained: bool = False) -> nn.Module:
    """
    创建Swin Transformer Tiny模型
    
    参数:
        num_classes: 分类类别数量
        pretrained: 是否使用预训练权重
    
    返回:
        配置好的Swin Transformer模型
    """
    if pretrained:
        model = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)
    else:
        model = models.swin_t(weights=None)
    
    # 修改最后一层以匹配类别数
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, num_classes)
    
    return model

def create_swin_s(num_classes: int = 8, pretrained: bool = False) -> nn.Module:
    """
    创建Swin Transformer Small模型
    
    参数:
        num_classes: 分类类别数量
        pretrained: 是否使用预训练权重
    
    返回:
        配置好的Swin Transformer模型
    """
    if pretrained:
        model = models.swin_s(weights=models.Swin_S_Weights.DEFAULT)
    else:
        model = models.swin_s(weights=None)
    
    # 修改最后一层以匹配类别数
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, num_classes)
    
    return model

def create_swin_b(num_classes: int = 8, pretrained: bool = False) -> nn.Module:
    """
    创建Swin Transformer Base模型
    
    参数:
        num_classes: 分类类别数量
        pretrained: 是否使用预训练权重
    
    返回:
        配置好的Swin Transformer模型
    """
    if pretrained:
        model = models.swin_b(weights=models.Swin_B_Weights.DEFAULT)
    else:
        model = models.swin_b(weights=None)
    
    # 修改最后一层以匹配类别数
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, num_classes)
    
    return model 