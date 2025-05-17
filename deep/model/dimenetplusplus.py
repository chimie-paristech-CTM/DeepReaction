"""
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




# -------------------------------
# Mathematical Utility Functions
# -------------------------------
def bessel_basis(num_spherical, num_radial):
    """
    Construct a list of symbolic expressions for Bessel basis functions.
    Returns a two-dimensional list with shape [num_spherical][num_radial].
    Here, sin(j*pi*x)/(j*pi*x) is used as an example basis function.
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
    Construct a list of symbolic expressions for real spherical harmonics.
    Returns a list where each element is a tuple containing the spherical harmonic expression.
    For l=0, returns a constant; for l>0, sin(l*theta) is used as an example.
    """
    theta = sym.symbols('theta')
    sph_harm = []
    for l in range(num_spherical):
        if l == 0:
            expr = 1  # Constant function
        else:
            expr = sym.sin(l * theta)
        sph_harm.append((expr,))
    return sph_harm


# -------------------------------
# Model Helper Modules
# -------------------------------
class Envelope(nn.Module):
    """
    Envelope function module that applies a polynomial envelope to distances.
    """
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
        # Apply the envelope function only for x < 1.0
        return (1. / x + a * x_pow_p0 + b * x_pow_p1 + c * x_pow_p2) * (x < 1.0).to(x.dtype)


class BesselBasisLayer(nn.Module):
    """
    Bessel basis layer that computes radial basis functions using an envelope.
    """
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
    """
    Spherical basis layer that computes angular basis functions using spherical harmonics and Bessel functions.
    """
    def __init__(self, num_spherical, num_radial, cutoff=5.0, envelope_exponent=5):
        super().__init__()
        self.num_spherical = num_spherical
        self.num_radial = num_radial
        self.cutoff = cutoff
        self.envelope = Envelope(envelope_exponent)

        # Initialize symbolic expressions for Bessel basis and spherical harmonics.
        bessel_forms = bessel_basis(num_spherical, num_radial)
        sph_harm_forms = real_sph_harm(num_spherical)

        self.sph_funcs = []
        self.bessel_funcs = []

        x, theta = sym.symbols('x theta')
        modules = {'sin': torch.sin, 'cos': torch.cos, 'Abs': torch.abs}
        for i in range(num_spherical):
            if i == 0:
                sph_val = sym.lambdify([theta], sph_harm_forms[i][0], modules)(0)
                # For l=0, use a constant function
                self.sph_funcs.append(lambda theta_val, s=sph_val: torch.zeros_like(theta_val) + s)
            else:
                func = sym.lambdify([theta], sph_harm_forms[i][0], modules)
                self.sph_funcs.append(func)
            for j in range(num_radial):
                func_bessel = sym.lambdify([x], bessel_forms[i][j], modules)
                self.bessel_funcs.append(func_bessel)

    def forward(self, dist, angle, idx_kj):
        dist = dist / self.cutoff
        # Compute the radial basis functions (rbf) using the Bessel functions
        rbf = torch.stack([f(dist) for f in self.bessel_funcs], dim=1)
        rbf = self.envelope(dist).unsqueeze(-1) * rbf
        # Compute the angular basis functions (cbf) using the spherical harmonic functions
        cbf = torch.stack([f(angle) for f in self.sph_funcs], dim=1)
        n, k = self.num_spherical, self.num_radial
        # Multiply and reshape the basis function outputs
        out = (rbf[idx_kj].view(-1, n, k) * cbf.view(-1, n, 1)).view(-1, n * k)
        return out


class EmbeddingBlock(nn.Module):
    """
    Embedding block that combines node embeddings and radial basis function embeddings.
    """
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
        # Lookup node embeddings
        x = self.emb(x)
        # Transform radial basis functions
        rbf = self.act(self.lin_rbf(rbf))
        # Concatenate features and transform
        return self.act(self.lin(torch.cat([x[i], x[j], rbf], dim=-1)))


class ResidualLayer(nn.Module):
    """
    Residual layer with two linear transformations and a skip connection.
    """
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
    """
    Interaction block for DimeNet++ that transforms Bessel and spherical basis representations,
    performs message passing and feature update.
    """
    def __init__(self, hidden_channels, int_emb_size, basis_emb_size,
                 num_spherical, num_radial, num_before_skip, num_after_skip,
                 act):
        super().__init__()
        self.act = act
        # Transformations for the radial basis functions (rbf) and spherical basis functions (sbf)
        self.lin_rbf1 = Linear(num_radial, basis_emb_size, bias=False)
        self.lin_rbf2 = Linear(basis_emb_size, hidden_channels, bias=False)
        self.lin_sbf1 = Linear(num_spherical * num_radial, basis_emb_size, bias=False)
        self.lin_sbf2 = Linear(basis_emb_size, int_emb_size, bias=False)
        # Transform input messages
        self.lin_kj = Linear(hidden_channels, hidden_channels)
        self.lin_ji = Linear(hidden_channels, hidden_channels)
        # Down-projection and up-projection layers
        self.lin_down = Linear(hidden_channels, int_emb_size, bias=False)
        self.lin_up = Linear(int_emb_size, hidden_channels, bias=False)
        # Residual layers before skip connection
        self.layers_before_skip = nn.ModuleList([
            nn.Sequential(Linear(hidden_channels, hidden_channels), nn.ReLU())
            for _ in range(num_before_skip)
        ])
        self.lin = Linear(hidden_channels, hidden_channels)
        # Residual layers after skip connection
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
    """
    Output block for DimeNet++ that aggregates features to produce final predictions.
    """
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
# DimeNet++ Model Implementation
# -------------------------------

class DimeNetPlusPlus(nn.Module):
    r"""DimeNet++ network implementation.
    This model takes multi-view inputs (pos0, pos1, pos2 with corresponding z0, z1, z2),
    computes predictions for each view, and combines them.
    """
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

        self.output_combination = Linear(out_channels * 3, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.rbf.reset_parameters()
        self.emb.reset_parameters()
        for out_block in self.output_blocks:
            out_block.reset_parameters()
        for interaction in self.interaction_blocks:
            interaction.reset_parameters()
        glorot_orthogonal(self.output_combination.weight, scale=2.0)
        self.output_combination.bias.data.fill_(0)

    def triplets(self, edge_index, num_nodes):
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
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch,
                                  max_num_neighbors=self.max_num_neighbors)
        i, j, idx_i, idx_j, idx_k, idx_kj, idx_ji = self.triplets(edge_index, num_nodes=z.size(0))
        dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()
        pos_i = pos[idx_i]
        pos_ji = pos[idx_j] - pos_i
        pos_ki = pos[idx_k] - pos_i

        a = (pos_ji * pos_ki).sum(dim=-1)
        b = torch.linalg.cross(pos_ji, pos_ki).norm(dim=-1)
        angle = torch.atan2(b, a)
        rbf = self.rbf(dist)
        sbf = self.sbf(dist, angle, idx_kj)
        x = self.emb(z, rbf, i, j)
        P = self.output_blocks[0](x, rbf, i, num_nodes=pos.size(0))
        for interaction_block, output_block in zip(self.interaction_blocks, self.output_blocks[1:]):
            x = interaction_block(x, rbf, sbf, idx_kj, idx_ji)
            P += output_block(x, rbf, i)
        return x, P

    def forward(self, z0, z1, z2, pos0, pos1, pos2, batch=None):
        _, P0 = self._forward_single(z0, pos0, batch)
        _, P1 = self._forward_single(z1, pos1, batch)
        _, P2 = self._forward_single(z2, pos2, batch)

        if batch is not None:
            from torch_scatter import scatter

            num_graphs = batch.max().item() + 1

            if P0.size(0) != num_graphs:
                P0 = scatter(P0, batch, dim=0, dim_size=num_graphs, reduce='mean')
            if P1.size(0) != num_graphs:
                P1 = scatter(P1, batch, dim=0, dim_size=num_graphs, reduce='mean')
            if P2.size(0) != num_graphs:
                P2 = scatter(P2, batch, dim=0, dim_size=num_graphs, reduce='mean')

        P_concat = torch.cat([P0, P1, P2], dim=1)
        P = self.output_combination(P_concat)
        return None, P