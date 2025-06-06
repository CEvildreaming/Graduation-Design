"""
模型定义模块
包含:
1. 常用卷积神经网络模型的字典
2. 支持的模型包括resnet18, resnet50, resnet101, vit, swin_transformer, googlenet, efficientnet等
"""

import torch
import torch.nn as nn
import torchvision.models as models

# 导入自定义模型创建函数
from .swin_transformer import create_swin_t, create_swin_s, create_swin_b
from .googlenet import create_googlenet
from .efficientnet import create_efficientnet_b0, create_efficientnet_b1, create_efficientnet_b2

def create_resnet18(num_classes=8, pretrained=False):
    """创建ResNet18模型"""
    if pretrained:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    else:
        model = models.resnet18(weights=None)
    
    # 修改最后一层以匹配类别数
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

def create_resnet50(num_classes=8, pretrained=False):
    """创建ResNet50模型"""
    if pretrained:
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    else:
        model = models.resnet50(weights=None)
    
    # 修改最后一层以匹配类别数
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

def create_resnet101(num_classes=8, pretrained=False):
    """创建ResNet101模型"""
    if pretrained:
        model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
    else:
        model = models.resnet101(weights=None)
    
    # 修改最后一层以匹配类别数
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

def create_vit(num_classes=8, pretrained=False):
    """创建ViT模型"""
    if pretrained:
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
    else:
        model = models.vit_b_16(weights=None)
    
    # 修改最后一层以匹配类别数
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)
    return model

# 模型字典：将模型名称映射到对应的创建函数
model_dict = {
    'resnet18': create_resnet18,
    'resnet50': create_resnet50,
    'resnet101': create_resnet101,
    'vit': create_vit,
    # 新增模型
    'swin-t': create_swin_t,
    'swin-s': create_swin_s,
    'swin-b': create_swin_b,
    'googlenet': create_googlenet,
    'efficientnet-b0': create_efficientnet_b0,
    'efficientnet-b1': create_efficientnet_b1,
    'efficientnet-b2': create_efficientnet_b2,
}

def create_model(model_name, num_classes=8):
    """创建指定模型的函数"""
    if model_name not in model_dict:
        raise ValueError(f"不支持的模型: {model_name}，可用模型: {list(model_dict.keys())}")
    return model_dict[model_name](num_classes=num_classes)