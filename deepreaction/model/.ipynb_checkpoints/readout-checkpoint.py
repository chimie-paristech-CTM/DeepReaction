import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch
import math


def swish(x):
    return x * x.sigmoid()


class Swish(nn.Module):
    def forward(self, x):
        return swish(x)


class ReadoutFactory:
    @staticmethod
    def create_readout(readout_type, **kwargs):
        if readout_type == 'sum':
            return SumReadout()
        elif readout_type == 'mean':
            return MeanReadout()
        elif readout_type == 'max':
            return MaxReadout()
        elif readout_type == 'attention':
            return AttentionReadout(
                node_dim=kwargs.get('node_dim', 128),
                readout_hidden_dim=kwargs.get('readout_hidden_dim', 128)
            )
        elif readout_type == 'set_transformer':
            return SetTransformer(
                dim_input=kwargs.get('node_dim', 128),
                readout_dim_hidden=kwargs.get('readout_hidden_dim', 1024),
                readout_num_heads=kwargs.get('readout_num_heads', 16),
                readout_num_sabs=kwargs.get('readout_num_sabs', 2),
                readout_layer_norm=kwargs.get('readout_layer_norm', False)
            )
        else:
            return SumReadout()


class ReadoutBase(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x_dense, mask=None):
        raise NotImplementedError("Subclasses must implement forward method")


class SumReadout(ReadoutBase):
    def forward(self, x_dense, mask=None):
        return x_dense.sum(dim=1)


class MeanReadout(ReadoutBase):
    def forward(self, x_dense, mask=None):
        if mask is None:
            mask = (x_dense.sum(dim=-1) != 0).float()
        
        counts = mask.sum(dim=1).clamp(min=1.0)
        return (x_dense.sum(dim=1) / counts.unsqueeze(-1))


class MaxReadout(ReadoutBase):
    def forward(self, x_dense, mask=None):
        if mask is None:
            mask = (x_dense.sum(dim=-1) != 0).float().unsqueeze(-1)
        else:
            mask = mask.unsqueeze(-1)
        
        x_masked = x_dense * mask - 1e9 * (1 - mask)
        return x_masked.max(dim=1)[0]


class AttentionReadout(ReadoutBase):
    def __init__(self, node_dim, readout_hidden_dim):
        super().__init__()
        
        self.attention_model = nn.Sequential(
            nn.Linear(node_dim, readout_hidden_dim),
            Swish(),
            nn.Linear(readout_hidden_dim, 1)
        )
    
    def forward(self, x_dense, mask=None):
        if mask is None:
            mask = (x_dense.sum(dim=-1) != 0).float().unsqueeze(-1)
        else:
            mask = mask.unsqueeze(-1)
        
        attention_weights = self.attention_model(x_dense).softmax(dim=1) * mask
        return (x_dense * attention_weights).sum(dim=1)


class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, readout_num_heads, readout_ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.readout_num_heads = readout_num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if readout_ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.readout_num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1,2)) / math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + swish(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O


class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, readout_num_heads, readout_ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, readout_num_heads, readout_ln=readout_ln)

    def forward(self, X):
        return self.mab(X, X)


class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, readout_num_heads, num_inds, readout_ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, readout_num_heads, readout_ln=readout_ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, readout_num_heads, readout_ln=readout_ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)


class PMA(nn.Module):
    def __init__(self, dim, readout_num_heads, num_seeds, readout_ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, readout_num_heads, readout_ln=readout_ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)


class SetTransformer(ReadoutBase):
    def __init__(self, dim_input, readout_dim_hidden=128, readout_num_heads=4, readout_layer_norm=False, readout_num_sabs=2, **kwargs):
        super().__init__()
        
        if readout_num_sabs == 2:
            self.enc = nn.Sequential(
                SAB(dim_input, readout_dim_hidden, readout_num_heads, readout_ln=readout_layer_norm),
                SAB(readout_dim_hidden, readout_dim_hidden, readout_num_heads, readout_ln=readout_layer_norm),
            )
        elif readout_num_sabs == 3:
            self.enc = nn.Sequential(
                SAB(dim_input, readout_dim_hidden, readout_num_heads, readout_ln=readout_layer_norm),
                SAB(readout_dim_hidden, readout_dim_hidden, readout_num_heads, readout_ln=readout_layer_norm),
                SAB(readout_dim_hidden, readout_dim_hidden, readout_num_heads, readout_ln=readout_layer_norm),
            )
            
        self.dec = nn.Sequential(
                PMA(readout_dim_hidden, readout_num_heads, 1, readout_ln=readout_layer_norm),
                SAB(readout_dim_hidden, readout_dim_hidden, readout_num_heads, readout_ln=readout_layer_norm),
                nn.Linear(readout_dim_hidden, dim_input)
        )

    def forward(self, x_dense, mask=None):
        x = self.enc(x_dense)
        x = self.dec(x)
        return x.squeeze(1)