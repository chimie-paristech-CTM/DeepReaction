"""
Activation functions for molecular prediction models.
"""

import torch
import torch.nn as nn


class Swish(nn.Module):
    """
    Swish activation function: x * sigmoid(x)
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * x.sigmoid()


def swish(x):
    """
    Swish activation function: x * sigmoid(x)
    
    Args:
        x: Input tensor
        
    Returns:
        torch.Tensor: Activated tensor
    """
    return x * x.sigmoid()