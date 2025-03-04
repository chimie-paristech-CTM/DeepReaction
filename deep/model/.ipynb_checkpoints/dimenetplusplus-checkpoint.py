"""
DimeNetPlusPlus.py

"Fast and Uncertainty-Aware Directional Message Passing for Non-Equilibrium Molecules"
"""

import os
import os.path as osp
from math import pi as PI, sqrt
from typing import Callable, Union

import numpy as np
import torch
from torch import nn
from torch.nn import Embedding, Linear
from torch_scatter import scatter
from torch_sparse import SparseTensor

from torch_geometric.data import Dataset, download_url
from torch_geometric.data.makedirs import makedirs
from torch_geometric.nn import radius_graph
from torch_geometric.nn.inits import glorot_orthogonal
from torch_geometric.nn.resolver import activation_resolver

import sympy as sym
import tensorflow as tf

# QM9 目标属性映射（如需使用预训练权重时用到）
qm9_target_dict = {
    0: 'mu',
    1: 'alpha',
    2: 'homo',
    3: 'lumo',
    5: 'r2',
    6: 'zpve',
    7: 'U0',
    8: 'U',
    9: 'H',
    10: 'G',
    11: 'Cv',
}


# -------------------------------
# 数学工具函数
# -------------------------------
def bessel_basis(num_spherical, num_radial):
    """
    构造 Bessel 基函数的符号表达式列表。
    返回一个二维列表，形状为 [num_spherical][num_radial]。
    这里采用 sin(j*pi*x)/(j*pi*x) 作为示例基函数。
    """
    x = sym.symbols('x')
    bessel_forms = []
    for i in range(num_spherical):
        forms_i = []
        for j in range(1, num_radial + 1):
            expr = sym.sin(j * PI * x) / (j * PI * x)
            forms_i.append(expr)
        bessel_forms.append(forms_i)
    return bessel_forms


def real_sph_harm(num_spherical):
    """
    构造实值球谐函数的符号表达式列表。
    返回一个列表，每个元素为一个包含球谐函数表达式的元组。
    对于 l=0 返回常数，对于 l>0 这里采用 sin(l*theta) 作为示例。
    """
    theta = sym.symbols('theta')
    sph_harm = []
    for l in range(num_spherical):
        if l == 0:
            expr = 1  # 常数函数
        else:
            expr = sym.sin(l * theta)
        sph_harm.append((expr,))
    return sph_harm


# -------------------------------
# 模型辅助模块
# -------------------------------
class Envelope(nn.Module):
    def __init__(self, exponent):
        super().__init__()
        self.p = exponent + 1
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2

    def forward(self, x):
        p, a, b, c = self.p, self.a, self.b, self.c
        x_pow_p0 = x.pow(p - 1)
        x_pow_p1 = x_pow_p0 * x
        x_pow_p2 = x_pow_p1 * x
        return (1. / x + a * x_pow_p0 + b * x_pow_p1 + c * x_pow_p2) * (x < 1.0).to(x.dtype)


class BesselBasisLayer(nn.Module):
    def __init__(self, num_radial, cutoff=5.0, envelope_exponent=5):
        super().__init__()
        self.cutoff = cutoff
        self.envelope = Envelope(envelope_exponent)
        self.freq = nn.Parameter(torch.Tensor(num_radial))
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            torch.arange(1, self.freq.numel() + 1, out=self.freq).mul_(PI)
        self.freq.requires_grad_()

    def forward(self, dist):
        dist = (dist.unsqueeze(-1) / self.cutoff)
        return self.envelope(dist) * (self.freq * dist).sin()


class SphericalBasisLayer(nn.Module):
    def __init__(self, num_spherical, num_radial, cutoff=5.0, envelope_exponent=5):
        super().__init__()
        self.num_spherical = num_spherical
        self.num_radial = num_radial
        self.cutoff = cutoff
        self.envelope = Envelope(envelope_exponent)

        bessel_forms = bessel_basis(num_spherical, num_radial)
        sph_harm_forms = real_sph_harm(num_spherical)

        self.sph_funcs = []
        self.bessel_funcs = []

        x, theta = sym.symbols('x theta')
        modules = {'sin': torch.sin, 'cos': torch.cos, 'Abs': torch.abs}
        for i in range(num_spherical):
            if i == 0:
                sph_val = sym.lambdify([theta], sph_harm_forms[i][0], modules)(0)
                self.sph_funcs.append(lambda theta_val, s=sph_val: torch.zeros_like(theta_val) + s)
            else:
                func = sym.lambdify([theta], sph_harm_forms[i][0], modules)
                self.sph_funcs.append(func)
            for j in range(num_radial):
                func_bessel = sym.lambdify([x], bessel_forms[i][j], modules)
                self.bessel_funcs.append(func_bessel)

    def forward(self, dist, angle, idx_kj):
        dist = dist / self.cutoff
        rbf = torch.stack([f(dist) for f in self.bessel_funcs], dim=1)
        rbf = self.envelope(dist).unsqueeze(-1) * rbf
        cbf = torch.stack([f(angle) for f in self.sph_funcs], dim=1)
        n, k = self.num_spherical, self.num_radial
        out = (rbf[idx_kj].view(-1, n, k) * cbf.view(-1, n, 1)).view(-1, n * k)
        return out


class EmbeddingBlock(nn.Module):
    def __init__(self, num_radial, hidden_channels, act):
        super().__init__()
        self.act = act
        self.emb = Embedding(95, hidden_channels)
        self.lin_rbf = Linear(num_radial, hidden_channels)
        self.lin = Linear(3 * hidden_channels, hidden_channels)
        self.reset_parameters()

    def reset_parameters(self):
        self.emb.weight.data.uniform_(-sqrt(3), sqrt(3))
        self.lin_rbf.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, x, rbf, i, j):
        x = self.emb(x)
        rbf = self.act(self.lin_rbf(rbf))
        return self.act(self.lin(torch.cat([x[i], x[j], rbf], dim=-1)))


class ResidualLayer(nn.Module):
    def __init__(self, hidden_channels, act):
        super().__init__()
        self.act = act
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)
        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin1.weight, scale=2.0)
        self.lin1.bias.data.fill_(0)
        glorot_orthogonal(self.lin2.weight, scale=2.0)
        self.lin2.bias.data.fill_(0)

    def forward(self, x):
        return x + self.act(self.lin2(self.act(self.lin1(x))))


class InteractionPPBlock(nn.Module):
    def __init__(self, hidden_channels, int_emb_size, basis_emb_size,
                 num_spherical, num_radial, num_before_skip, num_after_skip,
                 act):
        super().__init__()
        self.act = act
        # 对 Bessel 与球谐函数表示进行变换
        self.lin_rbf1 = Linear(num_radial, basis_emb_size, bias=False)
        self.lin_rbf2 = Linear(basis_emb_size, hidden_channels, bias=False)
        self.lin_sbf1 = Linear(num_spherical * num_radial, basis_emb_size, bias=False)
        self.lin_sbf2 = Linear(basis_emb_size, int_emb_size, bias=False)
        # 对输入信息的变换
        self.lin_kj = Linear(hidden_channels, hidden_channels)
        self.lin_ji = Linear(hidden_channels, hidden_channels)
        # 降维与升维
        self.lin_down = Linear(hidden_channels, int_emb_size, bias=False)
        self.lin_up = Linear(int_emb_size, hidden_channels, bias=False)
        # 残差层
        self.layers_before_skip = nn.ModuleList([
            nn.Sequential(Linear(hidden_channels, hidden_channels), nn.ReLU())
            for _ in range(num_before_skip)
        ])
        self.lin = Linear(hidden_channels, hidden_channels)
        self.layers_after_skip = nn.ModuleList([
            nn.Sequential(Linear(hidden_channels, hidden_channels), nn.ReLU())
            for _ in range(num_after_skip)
        ])
        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin_rbf1.weight, scale=2.0)
        glorot_orthogonal(self.lin_rbf2.weight, scale=2.0)
        glorot_orthogonal(self.lin_sbf1.weight, scale=2.0)
        glorot_orthogonal(self.lin_sbf2.weight, scale=2.0)
        glorot_orthogonal(self.lin_kj.weight, scale=2.0)
        self.lin_kj.bias.data.fill_(0)
        glorot_orthogonal(self.lin_ji.weight, scale=2.0)
        self.lin_ji.bias.data.fill_(0)
        glorot_orthogonal(self.lin_down.weight, scale=2.0)
        glorot_orthogonal(self.lin_up.weight, scale=2.0)
        for layer in self.layers_before_skip:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        glorot_orthogonal(self.lin.weight, scale=2.0)
        self.lin.bias.data.fill_(0)
        for layer in self.layers_after_skip:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, x, rbf, sbf, idx_kj, idx_ji):
        x_ji = self.act(self.lin_ji(x))
        x_kj = self.act(self.lin_kj(x))
        rbf = self.lin_rbf1(rbf)
        rbf = self.lin_rbf2(rbf)
        x_kj = x_kj * rbf
        x_kj = self.act(self.lin_down(x_kj))
        sbf = self.lin_sbf1(sbf)
        sbf = self.lin_sbf2(sbf)
        x_kj = x_kj[idx_kj] * sbf
        x_kj = scatter(x_kj, idx_ji, dim=0, dim_size=x.size(0))
        x_kj = self.act(self.lin_up(x_kj))
        h = x_ji + x_kj
        for layer in self.layers_before_skip:
            h = layer(h)
        h = self.act(self.lin(h)) + x
        for layer in self.layers_after_skip:
            h = layer(h)
        return h


class OutputPPBlock(nn.Module):
    def __init__(self, num_radial, hidden_channels, out_emb_channels,
                 out_channels, num_layers, act):
        super().__init__()
        self.act = act
        self.lin_rbf = Linear(num_radial, hidden_channels, bias=False)
        self.lin_up = Linear(hidden_channels, out_emb_channels, bias=False)
        self.lins = nn.ModuleList()
        for _ in range(num_layers):
            self.lins.append(Linear(out_emb_channels, out_emb_channels))
        self.lin = Linear(out_emb_channels, out_channels, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin_rbf.weight, scale=2.0)
        glorot_orthogonal(self.lin_up.weight, scale=2.0)
        for lin in self.lins:
            glorot_orthogonal(lin.weight, scale=2.0)
            lin.bias.data.fill_(0)
        self.lin.weight.data.fill_(0)

    def forward(self, x, rbf, i, num_nodes=None):
        x = self.lin_rbf(rbf) * x
        x = scatter(x, i, dim=0, dim_size=num_nodes)
        x = self.lin_up(x)
        for lin in self.lins:
            x = self.act(lin(x))
        return self.lin(x)


# -------------------------------
# DimeNet++ 模型实现（独立实现，不依赖外部 DimeNet）
# -------------------------------
class DimeNetPlusPlus(nn.Module):
    r"""DimeNet++ 网络实现
    该模型采用多视角（pos0, pos1, pos2）的输入，分别计算预测后取平均。
    """
    url = ('https://raw.githubusercontent.com/gasteigerjo/dimenet/'
           'master/pretrained/dimenet_pp')

    def __init__(self, hidden_channels: int, out_channels: int,
                 num_blocks: int, int_emb_size: int, basis_emb_size: int,
                 out_emb_channels: int, num_spherical: int, num_radial: int,
                 cutoff: float = 5.0, max_num_neighbors: int = 32,
                 envelope_exponent: int = 5, num_before_skip: int = 1,
                 num_after_skip: int = 2, num_output_layers: int = 3,
                 act: Union[str, Callable] = 'swish'):
        super().__init__()
        if num_spherical < 2:
            raise ValueError("num_spherical should be greater than 1")
        act = activation_resolver(act)
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.num_blocks = num_blocks

        self.rbf = BesselBasisLayer(num_radial, cutoff, envelope_exponent)
        self.sbf = SphericalBasisLayer(num_spherical, num_radial, cutoff, envelope_exponent)
        self.emb = EmbeddingBlock(num_radial, hidden_channels, act)
        self.output_blocks = nn.ModuleList([
            OutputPPBlock(num_radial, hidden_channels, out_emb_channels,
                          out_channels, num_output_layers, act)
            for _ in range(num_blocks + 1)
        ])
        self.interaction_blocks = nn.ModuleList([
            InteractionPPBlock(hidden_channels, int_emb_size, basis_emb_size,
                               num_spherical, num_radial, num_before_skip,
                               num_after_skip, act)
            for _ in range(num_blocks)
        ])
        self.reset_parameters()

    def reset_parameters(self):
        self.rbf.reset_parameters()
        self.emb.reset_parameters()
        for out_block in self.output_blocks:
            out_block.reset_parameters()
        for interaction in self.interaction_blocks:
            interaction.reset_parameters()

    def triplets(self, edge_index, num_nodes):
        # edge_index: (row, col) 表示边 j->i
        row, col = edge_index
        value = torch.arange(row.size(0), device=row.device)
        adj_t = SparseTensor(row=col, col=row, value=value,
                             sparse_sizes=(num_nodes, num_nodes))
        adj_t_row = adj_t[row]
        num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)
        idx_i = col.repeat_interleave(num_triplets)
        idx_j = row.repeat_interleave(num_triplets)
        idx_k = adj_t_row.storage.col()
        mask = idx_i != idx_k
        idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]
        idx_kj = adj_t_row.storage.value()[mask]
        idx_ji = adj_t_row.storage.row()[mask]
        return col, row, idx_i, idx_j, idx_k, idx_kj, idx_ji

    def _forward_single(self, z, pos, batch=None):
        # 利用 radius_graph 构造边
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch,
                                  max_num_neighbors=self.max_num_neighbors)
        i, j, idx_i, idx_j, idx_k, idx_kj, idx_ji = self.triplets(edge_index, num_nodes=z.size(0))
        dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()
        pos_i = pos[idx_i]
        pos_ji = pos[idx_j] - pos_i
        pos_ki = pos[idx_k] - pos_i
        a = (pos_ji * pos_ki).sum(dim=-1)
        b = torch.cross(pos_ji, pos_ki).norm(dim=-1)
        angle = torch.atan2(b, a)
        rbf = self.rbf(dist)
        sbf = self.sbf(dist, angle, idx_kj)
        x = self.emb(z, rbf, i, j)
        P = self.output_blocks[0](x, rbf, i, num_nodes=pos.size(0))
        for interaction_block, output_block in zip(self.interaction_blocks, self.output_blocks[1:]):
            x = interaction_block(x, rbf, sbf, idx_kj, idx_ji)
            P += output_block(x, rbf, i)
        return x, P

    def forward(self, z, pos0, pos1, pos2, batch=None):
        # 分别对三个不同视角计算预测，再取平均
        _, P0 = self._forward_single(z, pos0, batch)
        _, P1 = self._forward_single(z, pos1, batch)
        _, P2 = self._forward_single(z, pos2, batch)
        P = (P0 + P1 + P2) / 3
        return None, P

    @classmethod
    def from_qm9_pretrained(cls, root: str, dataset: Dataset, target: int):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        if target not in qm9_target_dict or target == 4:
            raise ValueError("Invalid target for QM9")
        root = osp.expanduser(osp.normpath(root))
        path = osp.join(root, 'pretrained_dimenet_pp', qm9_target_dict[target])
        makedirs(path)
        url = f'{cls.url}/{qm9_target_dict[target]}'
        if not osp.exists(osp.join(path, 'checkpoint')):
            download_url(f'{url}/checkpoint', path)
            download_url(f'{url}/ckpt.data-00000-of-00002', path)
            download_url(f'{url}/ckpt.data-00001-of-00002', path)
            download_url(f'{url}/ckpt.index', path)
        path = osp.join(path, 'ckpt')
        reader = tf.train.load_checkpoint(path)

        model = cls(
            hidden_channels=128,
            out_channels=1,
            num_blocks=4,
            int_emb_size=64,
            basis_emb_size=8,
            out_emb_channels=256,
            num_spherical=7,
            num_radial=6,
            cutoff=5.0,
            max_num_neighbors=32,
            envelope_exponent=5,
            num_before_skip=1,
            num_after_skip=2,
            num_output_layers=3,
            act='swish',
        )

        def copy_(src, name):
            init = reader.get_tensor(f'{name}/.ATTRIBUTES/VARIABLE_VALUE')
            init = torch.from_numpy(init)
            if name.endswith('kernel'):
                init = init.t()
            src.data.copy_(init)

        copy_(model.rbf.freq, 'rbf_layer/frequencies')
        copy_(model.emb.emb.weight, 'emb_block/embeddings')
        copy_(model.emb.lin_rbf.weight, 'emb_block/dense_rbf/kernel')
        copy_(model.emb.lin_rbf.bias, 'emb_block/dense_rbf/bias')
        copy_(model.emb.lin.weight, 'emb_block/dense/kernel')
        copy_(model.emb.lin.bias, 'emb_block/dense/bias')

        for i, block in enumerate(model.output_blocks):
            copy_(block.lin_rbf.weight, f'output_blocks/{i}/dense_rbf/kernel')
            copy_(block.lin_up.weight, f'output_blocks/{i}/up_projection/kernel')
            for j, lin in enumerate(block.lins):
                copy_(lin.weight, f'output_blocks/{i}/dense_layers/{j}/kernel')
                copy_(lin.bias, f'output_blocks/{i}/dense_layers/{j}/bias')
            copy_(block.lin.weight, f'output_blocks/{i}/dense_final/kernel')

        for i, block in enumerate(model.interaction_blocks):
            copy_(block.lin_rbf1.weight, f'int_blocks/{i}/dense_rbf1/kernel')
            copy_(block.lin_rbf2.weight, f'int_blocks/{i}/dense_rbf2/kernel')
            copy_(block.lin_sbf1.weight, f'int_blocks/{i}/dense_sbf1/kernel')
            copy_(block.lin_sbf2.weight, f'int_blocks/{i}/dense_sbf2/kernel')
            copy_(block.lin_ji.weight, f'int_blocks/{i}/dense_ji/kernel')
            copy_(block.lin_ji.bias, f'int_blocks/{i}/dense_ji/bias')
            copy_(block.lin_kj.weight, f'int_blocks/{i}/dense_kj/kernel')
            copy_(block.lin_kj.bias, f'int_blocks/{i}/dense_kj/bias')
        random_state = np.random.RandomState(seed=42)
        perm = torch.from_numpy(random_state.permutation(np.arange(130831)))
        train_idx = perm[:110000]
        val_idx = perm[110000:120000]
        test_idx = perm[120000:]
        return model, (dataset[train_idx], dataset[val_idx], dataset[test_idx])


