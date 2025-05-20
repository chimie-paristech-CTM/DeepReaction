import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch
from typing import List, Union, Optional, Dict, Any

from .factory import ModelFactory
from .readout import ReadoutFactory


class EnhancedPredictionMLP(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int = 1,
            hidden_dim: int = 128,
            num_hidden_layers: int = 3,
            dropout: float = 0.0,
            use_layer_norm: bool = False
    ):
        super().__init__()

        self.num_hidden_layers = num_hidden_layers
        self.hidden_dim = hidden_dim

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))

        for _ in range(1, num_hidden_layers):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.layer_norms = nn.ModuleList()
        for _ in range(num_hidden_layers):
            self.layer_norms.append(nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity())

    def forward(self, x):
        for i in range(self.num_hidden_layers):
            x = self.layers[i](x)
            x = self.activation(x)
            x = self.dropout(x)
            x = self.layer_norms[i](x)

        return self.fc_out(x)


class MultiTargetPredictionHead(nn.Module):
    def __init__(
            self,
            input_dim: int,
            num_targets: int = 1,
            hidden_dim: int = 128,
            num_hidden_layers: int = 3,
            dropout: float = 0.0,
            use_layer_norm: bool = False,
            additional_features_dim: int = 2,
            use_xtb_features: bool = True
    ):
        super().__init__()

        self.num_targets = num_targets
        self.use_xtb_features = use_xtb_features

        # 调试信息
        print(f"DEBUG - MultiTargetPredictionHead initialized with {num_targets} targets")
        print(f"DEBUG - MultiTargetPredictionHead input_dim: {input_dim}")

        # 创建多个预测头，每个目标一个
        self.prediction_heads = nn.ModuleList()
        for i in range(num_targets):
            print(f"DEBUG - Creating head for target {i}")
            head = EnhancedPredictionMLP(
                input_dim=input_dim,
                output_dim=1,  # 每个头只预测一个值
                hidden_dim=hidden_dim,
                num_hidden_layers=num_hidden_layers,
                dropout=dropout,
                use_layer_norm=use_layer_norm
            )
            self.prediction_heads.append(head)

    def forward(self, x):
        # 从每个头获取预测
        predictions = []
        for i, head in enumerate(self.prediction_heads):
            pred = head(x)
            predictions.append(pred)

        # 在维度1上连接预测结果
        if len(predictions) > 1:
            result = torch.cat(predictions, dim=1)
        else:
            result = predictions[0]

        # 调试输出形状
        if not hasattr(self, 'debug_printed'):
            print(f"DEBUG - MultiTargetPredictionHead output shape: {result.shape}")
            self.debug_printed = True

        return result


class MoleculePredictionModel(nn.Module):
    def __init__(
            self,
            model_type: str = 'dimenet++',
            readout_type: str = 'sum',
            max_num_atoms: int = 100,
            node_dim: int = 128,
            output_dim: int = 1,
            dropout: float = 0.0,
            use_layer_norm: bool = False,
            readout_kwargs: Optional[Dict[str, Any]] = None,
            model_kwargs: Optional[Dict[str, Any]] = None,
            use_xtb_features: bool = False,
            prediction_hidden_layers: int = 3,
            prediction_hidden_dim: int = 128,
            num_xtb_features: int = 0
    ):
        super().__init__()

        self.model_type = model_type
        self.readout_type = readout_type
        self.max_num_atoms = max_num_atoms
        self.node_dim = node_dim
        self.output_dim = output_dim
        self.use_xtb_features = use_xtb_features
        self.prediction_hidden_layers = prediction_hidden_layers
        self.prediction_hidden_dim = prediction_hidden_dim
        self.num_xtb_features = num_xtb_features

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
            'hidden_dim': readout_kwargs.get('hidden_dim', 128),
            'num_heads': readout_kwargs.get('num_heads', 4),
            'layer_norm': use_layer_norm,
            'num_sabs': readout_kwargs.get('num_sabs', 2)
        }
        self.readout = ReadoutFactory.create_readout(readout_type, **readout_params)

        input_dim_for_mlp = node_dim + (num_xtb_features if use_xtb_features else 0)

        self.prediction_mlp = MultiTargetPredictionHead(
            input_dim=input_dim_for_mlp,
            num_targets=output_dim,
            hidden_dim=prediction_hidden_dim,
            num_hidden_layers=prediction_hidden_layers,
            dropout=dropout,
            use_layer_norm=use_layer_norm,
            additional_features_dim=num_xtb_features,
            use_xtb_features=use_xtb_features
        )

    def forward(self, pos0, pos1, pos2, z0, z1, z2, batch_mapping, xtb_features=None):
        _, graph_embeddings = self.base_model(
            z0=z0, z1=z1, z2=z2,
            pos0=pos0, pos1=pos1, pos2=pos2,
            batch=batch_mapping
        )

        dummy_nodes = torch.zeros((z0.size(0), self.node_dim), device=z0.device)
        node_embeddings_dense, mask = to_dense_batch(
            dummy_nodes, batch_mapping, 0, self.max_num_atoms
        )

        if self.use_xtb_features and xtb_features is not None and xtb_features.size(1) > 0:
            if self.num_xtb_features is not None and xtb_features.shape[1] > self.num_xtb_features:
                features_to_use = xtb_features[:, :self.num_xtb_features]
            else:
                features_to_use = xtb_features

            combined_features = torch.cat([graph_embeddings, features_to_use], dim=1)
        else:
            combined_features = graph_embeddings

        predictions = self.prediction_mlp(combined_features)

        return node_embeddings_dense, graph_embeddings, predictions