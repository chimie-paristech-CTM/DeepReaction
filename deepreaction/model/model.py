import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch
from typing import List, Union, Optional, Dict, Any

from .model_factory import ModelFactory
from .readout import ReadoutFactory


class PredictionMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        pred_hidden_dim: int = 128,
        pred_num_hidden_layers: int = 3,
        pred_dropout: float = 0.0,
        pred_use_layer_norm: bool = False
    ):
        super().__init__()
        
        self.pred_num_hidden_layers = pred_num_hidden_layers
        self.pred_hidden_dim = pred_hidden_dim
        
        self.pred_layers = nn.ModuleList()
        self.pred_layers.append(nn.Linear(input_dim, pred_hidden_dim))
        
        for _ in range(1, pred_num_hidden_layers):
            self.pred_layers.append(nn.Linear(pred_hidden_dim, pred_hidden_dim))
        
        self.pred_fc_out = nn.Linear(pred_hidden_dim, output_dim)
        self.pred_activation = nn.ReLU()
        self.pred_dropout = nn.Dropout(pred_dropout) if pred_dropout > 0 else nn.Identity()
        
        self.pred_layer_norms = nn.ModuleList()
        for _ in range(pred_num_hidden_layers):
            self.pred_layer_norms.append(nn.LayerNorm(pred_hidden_dim) if pred_use_layer_norm else nn.Identity())
    
    def forward(self, x):
        for i in range(self.pred_num_hidden_layers):
            x = self.pred_layers[i](x)
            x = self.pred_activation(x)
            x = self.pred_dropout(x)
            x = self.pred_layer_norms[i](x)
        
        return self.pred_fc_out(x)


class MultiTargetPredictionHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        pred_num_targets: int = 1,
        pred_hidden_dim: int = 128,
        pred_num_hidden_layers: int = 3,
        pred_dropout: float = 0.0,
        pred_use_layer_norm: bool = False,
        pred_additional_features_dim: int = 2,
        pred_use_xtb_features: bool = True
    ):
        super().__init__()
        
        self.pred_num_targets = pred_num_targets
        self.pred_use_xtb_features = pred_use_xtb_features
        
        self.pred_heads = nn.ModuleList()
        for _ in range(pred_num_targets):
            head = PredictionMLP(
                input_dim=input_dim,
                output_dim=1,
                pred_hidden_dim=pred_hidden_dim,
                pred_num_hidden_layers=pred_num_hidden_layers,
                pred_dropout=pred_dropout,
                pred_use_layer_norm=pred_use_layer_norm
            )
            self.pred_heads.append(head)
    
    def forward(self, x):
        predictions = [head(x) for head in self.pred_heads]
        return torch.cat(predictions, dim=1)


class MoleculePredictionModel(nn.Module):
    def __init__(
        self,
        model_type: str = 'dimenet++',
        readout_type: str = 'sum',
        max_num_atoms: int = 100,
        node_dim: int = 128,
        output_dim: int = 1,
        pred_dropout: float = 0.0,
        pred_use_layer_norm: bool = False,
        readout_kwargs: Optional[Dict[str, Any]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        pred_use_xtb_features: bool = True,
        pred_hidden_layers: int = 3,
        pred_hidden_dim: int = 128,
        pred_num_xtb_features: int = 2
    ):
        super().__init__()

        self.model_type = model_type
        self.readout_type = readout_type
        self.max_num_atoms = max_num_atoms
        self.node_dim = node_dim
        self.output_dim = output_dim
        self.pred_use_xtb_features = pred_use_xtb_features
        self.pred_hidden_layers = pred_hidden_layers
        self.pred_hidden_dim = pred_hidden_dim
        self.pred_num_xtb_features = pred_num_xtb_features

        if readout_kwargs is None:
            readout_kwargs = {}
        if model_kwargs is None:
            model_kwargs = {}

        model_kwargs['out_channels'] = node_dim

        self.base_model = ModelFactory.create_model(
            model_type=model_type,
            **model_kwargs
        )

        readout_params = {
            'node_dim': node_dim,
            'readout_hidden_dim': readout_kwargs.get('readout_hidden_dim', 128),
            'readout_num_heads': readout_kwargs.get('readout_num_heads', 4),
            'readout_layer_norm': pred_use_layer_norm,
            'readout_num_sabs': readout_kwargs.get('readout_num_sabs', 2)
        }
        self.readout = ReadoutFactory.create_readout(readout_type, **readout_params)
        
        self.graph_combination = nn.Linear(node_dim * 3, node_dim)
        
        input_dim_for_mlp = node_dim + (pred_num_xtb_features if pred_use_xtb_features else 0)
        
        self.prediction_mlp = MultiTargetPredictionHead(
            input_dim=input_dim_for_mlp,
            pred_num_targets=output_dim,
            pred_hidden_dim=pred_hidden_dim,
            pred_num_hidden_layers=pred_hidden_layers,
            pred_dropout=pred_dropout,
            pred_use_layer_norm=pred_use_layer_norm,
            pred_additional_features_dim=pred_num_xtb_features,
            pred_use_xtb_features=pred_use_xtb_features
        )

    def forward(self, pos0, pos1, pos2, z0, z1, z2, batch_mapping, xtb_features=None):
        (x0, x1, x2), (P0, P1, P2) = self.base_model(
            z0=z0, z1=z1, z2=z2, 
            pos0=pos0, pos1=pos1, pos2=pos2, 
            batch=batch_mapping
        )
        
        P0_dense, mask0 = to_dense_batch(P0, batch_mapping, 0, self.max_num_atoms)
        P1_dense, mask1 = to_dense_batch(P1, batch_mapping, 0, self.max_num_atoms)
        P2_dense, mask2 = to_dense_batch(P2, batch_mapping, 0, self.max_num_atoms)
        
        graph_emb0 = self.readout(P0_dense, mask0)
        graph_emb1 = self.readout(P1_dense, mask1)
        graph_emb2 = self.readout(P2_dense, mask2)
        
        combined_graph_emb = torch.cat([graph_emb0, graph_emb1, graph_emb2], dim=1)
        graph_embeddings = self.graph_combination(combined_graph_emb)

        if self.pred_use_xtb_features and xtb_features is not None:
            if self.pred_num_xtb_features is not None and xtb_features.shape[1] > self.pred_num_xtb_features:
                features_to_use = xtb_features[:, :self.pred_num_xtb_features]
            else:
                features_to_use = xtb_features
            
            combined_features = torch.cat([graph_embeddings, features_to_use], dim=1)
        else:
            combined_features = graph_embeddings

        predictions = self.prediction_mlp(combined_features)
        
        return P0_dense, graph_embeddings, predictions