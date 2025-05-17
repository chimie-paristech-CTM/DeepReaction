"""
model/head.py
"""
import torch
import torch.nn as nn

class MLPHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.0, use_layer_norm: bool = False):
        """
        Args:
            input_dim (int): 输入维度
            output_dim (int): 输出维度（如 1 用于回归任务）
            dropout (float): dropout 概率，默认为 0.0
            use_layer_norm (bool): 是否使用 LayerNorm，默认为 False
        """
        super(MLPHead, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, input_dim))
        layers.append(nn.ReLU())
        if use_layer_norm:
            layers.append(nn.LayerNorm(input_dim))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(input_dim, input_dim // 2))
        layers.append(nn.ReLU())
        if use_layer_norm:
            layers.append(nn.LayerNorm(input_dim // 2))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(input_dim // 2, input_dim // 2))
        layers.append(nn.ReLU())
        if use_layer_norm:
            layers.append(nn.LayerNorm(input_dim // 2))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(input_dim // 2, output_dim))
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.mlp(x)
