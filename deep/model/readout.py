"""
Readout modules for molecular property prediction models.

This module contains various readout mechanisms to convert node embeddings
to graph-level embeddings, including sum, mean, max, attention, and set transformer.
"""

import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch
from .activations import Swish


class ReadoutFactory:
    """
    Factory class to create readout modules based on type.
    """
    @staticmethod
    def create_readout(readout_type, **kwargs):
        """
        Create a readout module based on specified type.
        
        Args:
            readout_type (str): Type of readout ('sum', 'mean', 'max', 'attention', 'set_transformer')
            **kwargs: Additional arguments for specific readout types
            
        Returns:
            ReadoutBase: The readout module
        """
        if readout_type == 'sum':
            return SumReadout()
        elif readout_type == 'mean':
            return MeanReadout()
        elif readout_type == 'max':
            return MaxReadout()
        elif readout_type == 'attention':
            return AttentionReadout(
                node_dim=kwargs.get('node_dim', 128),
                hidden_dim=kwargs.get('hidden_dim', 128)
            )
        elif readout_type == 'set_transformer':
            # Importing here to avoid circular imports
            from .set_transformer import SetTransformer
            return SetTransformerReadout(
                dim_input=kwargs.get('node_dim', 128),
                dim_hidden=kwargs.get('hidden_dim', 1024),
                num_heads=kwargs.get('num_heads', 16),
                num_sabs=kwargs.get('num_sabs', 2),
                layer_norm=kwargs.get('layer_norm', False)
            )
        else:
            # Default to sum if an invalid type is provided
            return SumReadout()


class ReadoutBase(nn.Module):
    """
    Base class for readout modules.
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, x_dense, mask=None):
        """
        Forward pass for readout.
        
        Args:
            x_dense (torch.Tensor): Node embeddings in dense batch format
            mask (torch.Tensor, optional): Mask for valid nodes
            
        Returns:
            torch.Tensor: Graph-level embeddings
        """
        raise NotImplementedError("Subclasses must implement forward method")


class SumReadout(ReadoutBase):
    """
    Sum readout module.
    """
    def forward(self, x_dense, mask=None):
        """
        Sum the node features to get graph-level embeddings.
        
        Args:
            x_dense (torch.Tensor): Node embeddings in dense batch format
            mask (torch.Tensor, optional): Mask for valid nodes (unused in sum)
            
        Returns:
            torch.Tensor: Graph-level embeddings
        """
        return x_dense.sum(dim=1)


class MeanReadout(ReadoutBase):
    """
    Mean readout module.
    """
    def forward(self, x_dense, mask=None):
        """
        Average the node features to get graph-level embeddings.
        
        Args:
            x_dense (torch.Tensor): Node embeddings in dense batch format
            mask (torch.Tensor, optional): Mask for valid nodes
            
        Returns:
            torch.Tensor: Graph-level embeddings
        """
        if mask is None:
            # Create mask based on zero elements
            mask = (x_dense.sum(dim=-1) != 0).float()
        
        # Calculate mean, avoiding division by zero
        counts = mask.sum(dim=1).clamp(min=1.0)
        return (x_dense.sum(dim=1) / counts.unsqueeze(-1))


class MaxReadout(ReadoutBase):
    """
    Max readout module.
    """
    def forward(self, x_dense, mask=None):
        """
        Take the maximum value of node features to get graph-level embeddings.
        
        Args:
            x_dense (torch.Tensor): Node embeddings in dense batch format
            mask (torch.Tensor, optional): Mask for valid nodes
            
        Returns:
            torch.Tensor: Graph-level embeddings
        """
        if mask is None:
            # Create mask based on zero elements
            mask = (x_dense.sum(dim=-1) != 0).float().unsqueeze(-1)
        else:
            mask = mask.unsqueeze(-1)
        
        # Apply mask to handle padding
        x_masked = x_dense * mask - 1e9 * (1 - mask)
        return x_masked.max(dim=1)[0]


class AttentionReadout(ReadoutBase):
    """
    Attention-based readout module.
    """
    def __init__(self, node_dim, hidden_dim):
        """
        Initialize attention readout.
        
        Args:
            node_dim (int): Dimension of node features
            hidden_dim (int): Dimension of hidden layer
        """
        super().__init__()
        
        self.attention_model = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x_dense, mask=None):
        """
        Apply attention mechanism to get graph-level embeddings.
        
        Args:
            x_dense (torch.Tensor): Node embeddings in dense batch format
            mask (torch.Tensor, optional): Mask for valid nodes
            
        Returns:
            torch.Tensor: Graph-level embeddings
        """
        if mask is None:
            # Create mask based on zero elements
            mask = (x_dense.sum(dim=-1) != 0).float().unsqueeze(-1)
        else:
            mask = mask.unsqueeze(-1)
        
        # Calculate attention weights
        attention_weights = self.attention_model(x_dense).softmax(dim=1) * mask
        return (x_dense * attention_weights).sum(dim=1)


class SetTransformerReadout(ReadoutBase):
    """
    Set Transformer readout module.
    """
    def __init__(self, dim_input, dim_hidden, num_heads, num_sabs, layer_norm):
        """
        Initialize Set Transformer readout.
        
        Args:
            dim_input (int): Dimension of input features
            dim_hidden (int): Dimension of hidden layer
            num_heads (int): Number of attention heads
            num_sabs (int): Number of self-attention blocks
            layer_norm (bool): Whether to use layer normalization
        """
        super().__init__()
        # Import here to avoid circular imports
        from .set_transformer import SetTransformer
        
        self.st = SetTransformer(
            dim_input=dim_input,
            num_outputs=1,
            dim_output=dim_input,
            num_inds=None,
            dim_hidden=dim_hidden,
            num_heads=num_heads,
            ln=layer_norm,
            num_sabs=num_sabs
        )
    
    def forward(self, x_dense, mask=None):
        """
        Apply Set Transformer to get graph-level embeddings.
        
        Args:
            x_dense (torch.Tensor): Node embeddings in dense batch format
            mask (torch.Tensor, optional): Mask for valid nodes (unused in Set Transformer)
            
        Returns:
            torch.Tensor: Graph-level embeddings
        """
        transformed = self.st(x_dense)
        return transformed.mean(dim=1)