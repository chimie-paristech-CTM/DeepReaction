import torch
import torch.nn as nn
from typing import Dict, Any
import torch.nn.functional as F
import math

class SumReadout(nn.Module):
    def __init__(self, node_dim=128, **kwargs):
        super().__init__()
        self.node_dim = node_dim

    def forward(self, x, batch=None):
        if batch is None:
            return x.sum(dim=0, keepdim=True)
        else:
            from torch_scatter import scatter
            return scatter(x, batch, dim=0, reduce='sum')

class MeanReadout(nn.Module):
    def __init__(self, node_dim=128, **kwargs):
        super().__init__()
        self.node_dim = node_dim

    def forward(self, x, batch=None):
        if batch is None:
            return x.mean(dim=0, keepdim=True)
        else:
            from torch_scatter import scatter
            return scatter(x, batch, dim=0, reduce='mean')

class MaxReadout(nn.Module):
    def __init__(self, node_dim=128, **kwargs):
        super().__init__()
        self.node_dim = node_dim

    def forward(self, x, batch=None):
        if batch is None:
            return x.max(dim=0, keepdim=True)[0]
        else:
            from torch_scatter import scatter
            return scatter(x, batch, dim=0, reduce='max')

class AttentionReadout(nn.Module):
    def __init__(self, node_dim=128, hidden_dim=256, num_heads=8, layer_norm=False, **kwargs):
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        self.query = nn.Linear(node_dim, hidden_dim * num_heads)
        self.key = nn.Linear(node_dim, hidden_dim * num_heads)
        self.value = nn.Linear(node_dim, hidden_dim * num_heads)
        
        self.out = nn.Linear(hidden_dim * num_heads, node_dim)
        self.layer_norm = nn.LayerNorm(node_dim) if layer_norm else nn.Identity()
        
    def forward(self, x, batch=None):
        if batch is None:
            return x.mean(dim=0, keepdim=True)  # Fallback to mean pooling
        
        from torch_scatter import scatter
        from torch_geometric.utils import to_dense_batch
        
        # Create batched tensor
        x_dense, mask = to_dense_batch(x, batch)
        
        # Multi-head attention
        B, N, D = x_dense.shape
        H = self.num_heads
        
        q = self.query(x_dense).view(B, N, H, -1)
        k = self.key(x_dense).view(B, N, H, -1)
        v = self.value(x_dense).view(B, N, H, -1)
        
        # Compute attention scores
        scores = torch.einsum('bnhd,bmhd->bnmh', q, k) / (D ** 0.5)
        
        # Apply mask
        mask = mask.unsqueeze(2).expand(B, N, N)
        scores = scores.masked_fill(~mask.unsqueeze(-1), -1e9)
        
        # Apply softmax
        attn = torch.softmax(scores, dim=2)
        
        # Apply attention
        out = torch.einsum('bnmh,bmhd->bnhd', attn, v)
        out = out.reshape(B, N, -1)
        
        # Apply output projection
        out = self.out(out)
        out = self.layer_norm(out)
        
        # Sum over nodes to get graph-level representation
        out = out.sum(dim=1)
        
        return out




def swish(x):
    return x * x.sigmoid()


class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1,2)) / math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        # O = O + F.relu(self.fc_o(O))
        O = O + swish(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O

class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)

class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)

class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)

class SetTransformer(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output, num_inds=32, dim_hidden=128, num_heads=4, ln=False, num_sabs=2):
        super(SetTransformer, self).__init__()

        if num_sabs == 2:
            self.enc = nn.Sequential(
                SAB(dim_input, dim_hidden, num_heads, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
            )
        elif num_sabs == 3:
            self.enc = nn.Sequential(
                SAB(dim_input, dim_hidden, num_heads, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
            )
            
        self.dec = nn.Sequential(
                PMA(dim_hidden, num_heads, num_outputs, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                nn.Linear(dim_hidden, dim_output)
        )

    def forward(self, X):
        return self.dec(self.enc(X))

class ReadoutFactory:
    @staticmethod
    def create_readout(readout_type: str, **kwargs):
        readout_type = readout_type.lower()
        
        if readout_type == 'sum':
            return SumReadout(**kwargs)
        elif readout_type == 'mean':
            return MeanReadout(**kwargs)
        elif readout_type == 'max':
            return MaxReadout(**kwargs)
        elif readout_type == 'attention' or readout_type == 'multihead_attention':
            return AttentionReadout(**kwargs)
        elif readout_type == 'set_transformer':
            return SetTransformerReadout(**kwargs)
        else:
            # Default to mean readout
            print(f"Warning: Readout type '{readout_type}' not recognized. Using mean readout.")
            return MeanReadout(**kwargs)