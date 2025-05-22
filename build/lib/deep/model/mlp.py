"""
MLP modules for prediction models.

This module contains MLP architectures used for prediction
after readout operations.
"""

import torch
import torch.nn as nn
from .activations import Swish


class PredictionMLP(nn.Module):
    """
    MLP for prediction after readout.
    """
    def __init__(
        self,
        input_dim,
        output_dim=1,
        hidden_dim=None,
        dropout=0.0,
        use_layer_norm=False
    ):
        """
        Initialize prediction MLP.
        
        Args:
            input_dim (int): Dimension of input features
            output_dim (int, optional): Dimension of output features
            hidden_dim (int, optional): Dimension of hidden layer
            dropout (float, optional): Dropout rate
            use_layer_norm (bool, optional): Whether to use layer normalization
        """
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = input_dim
            
        # Define MLP layers
        layers = [
            nn.Linear(input_dim, input_dim),
            Swish()
        ]
        
        if use_layer_norm:
            layers.append(nn.LayerNorm(input_dim))
        
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        
        layers.extend([
            nn.Linear(input_dim, hidden_dim // 2),
            Swish()
        ])
        
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_dim // 2))
        
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        
        layers.extend([
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            Swish()
        ])
        
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_dim // 2))
        
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        
        # Output layer
        layers.append(nn.Linear(hidden_dim // 2, output_dim))
        
        # Create sequential model
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass through MLP.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        return self.mlp(x)