"""
model/model.py
"""

import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch

# Import existing models and components from the current codebase
from .dimenet import DimeNet
from .dimenetplusplus import DimeNetPlusPlus
from .readout import ReadoutFactory
from .mlp import PredictionMLP


class MoleculeModel(nn.Module):
    def __init__(self, model_name: str = 'dimenet++', **model_kwargs):
        """
        Initialize the main model interface by selecting and loading a model based on the model_name parameter.

        Args:
            model_name (str): The model name. Supported values are 'dimenet' and 'dimenet++' (or 'dimenet_pp').
            **model_kwargs: Parameters to pass to the corresponding model constructor.
                Common parameters include:
                    - hidden_channels: Dimension of hidden layers
                    - out_channels: Output dimension (typically the latent representation dimension of nodes)
                    - num_blocks: Number of interaction blocks
                    - num_bilinear / int_emb_size / basis_emb_size / out_emb_channels: Specific parameters for DimeNet++
                    - num_spherical, num_radial, cutoff, envelope_exponent, etc.
        """
        super(MoleculeModel, self).__init__()

        model_name = model_name.lower()
        if model_name == 'dimenet':
            # Call the original DimeNet model (ensure that model_kwargs are set correctly)
            self.model = DimeNet(**model_kwargs)
        elif model_name in ['dimenet++', 'dimenet_pp']:
            # Call the DimeNet++ model
            self.model = DimeNetPlusPlus(**model_kwargs)
        else:
            raise ValueError(f"Unknown model_name: {model_name}. Supported: 'dimenet', 'dimenet++'.")

    def forward(self, *args, **kwargs):
        """
        The forward function directly calls the internal model's forward.

        Args:
            *args, **kwargs: Parameters passed to the specific model's forward method.
        Returns:
            The output of the model.
        """
        return self.model(*args, **kwargs)


class MoleculePredictionModel(nn.Module):
    """
    A complete molecular prediction model that includes a feature extractor, a readout layer, and a prediction head.
    This unified interface allows assembling different components to build a complete model architecture.
    """
    def __init__(
        self,
        base_model_name: str = 'dimenet++',
        readout_type: str = 'sum',
        max_num_atoms: int = 100,
        node_dim: int = 128,
        output_dim: int = 1,
        dropout: float = 0.0,
        use_layer_norm: bool = False,
        readout_kwargs: dict = None,
        base_model_kwargs: dict = None,
    ):
        """
        Initialize the molecular prediction model.

        Args:
            base_model_name (str): The name of the base model, supports 'dimenet' and 'dimenet++'.
            readout_type (str): The type of readout layer, supports 'sum', 'mean', 'max', 'attention', 'set_transformer'.
            max_num_atoms (int): Maximum number of atoms.
            node_dim (int): Node feature dimension.
            output_dim (int): Output dimension.
            dropout (float): Dropout rate.
            use_layer_norm (bool): Whether to use layer normalization.
            readout_kwargs (dict): Parameters for the readout layer.
            base_model_kwargs (dict): Parameters for the base model.
        """
        super(MoleculePredictionModel, self).__init__()

        # Initialize parameters
        self.base_model_name = base_model_name
        self.readout_type = readout_type
        self.max_num_atoms = max_num_atoms
        self.node_dim = node_dim

        # Set default parameters
        if readout_kwargs is None:
            readout_kwargs = {}

        if base_model_kwargs is None:
            base_model_kwargs = {}

        # Add default parameter for output channels if not provided
        if 'out_channels' not in base_model_kwargs:
            base_model_kwargs['out_channels'] = node_dim

        # Initialize the base model
        self.base_model = MoleculeModel(model_name=base_model_name, **base_model_kwargs)

        # Initialize the readout layer
        readout_params = {
            'node_dim': node_dim,
            'hidden_dim': readout_kwargs.get('hidden_dim', 128),
            'num_heads': readout_kwargs.get('num_heads', 4),
            'layer_norm': use_layer_norm,
            'num_sabs': readout_kwargs.get('num_sabs', 2)
        }
        self.readout = ReadoutFactory.create_readout(readout_type, **readout_params)

        # Initialize the prediction head (MLP)
        self.prediction_mlp = PredictionMLP(
            input_dim=node_dim,
            output_dim=output_dim,
            hidden_dim=node_dim,
            dropout=dropout,
            use_layer_norm=use_layer_norm
        )

    def forward(self, pos0, pos1, pos2, atom_z, batch_mapping):
        """
        Forward pass through the model.

        Args:
            pos0: First position tensor.
            pos1: Second position tensor.
            pos2: Third position tensor.
            atom_z: Atomic numbers.
            batch_mapping: Batch mapping indices.

        Returns:
            Tuple: (node_embeddings, graph_embeddings, predictions)
        """
        # Pass input through the DimeNet++ base model
        _, node_embeddings = self.base_model(z=atom_z, pos0=pos0, pos1=pos1, pos2=pos2, batch=batch_mapping)

        # Convert node embeddings to a dense batch representation
        node_embeddings_dense, mask = to_dense_batch(node_embeddings, batch_mapping, 0, self.max_num_atoms)

        # Apply the readout layer to aggregate node features into graph-level embeddings
        graph_embeddings = self.readout(node_embeddings_dense, mask)

        # Apply the prediction MLP and flatten the output for predictions
        predictions = torch.flatten(self.prediction_mlp(graph_embeddings))

        return node_embeddings_dense, graph_embeddings, predictions

    # Retain the original forward method as an alias for backward compatibility
    def forward_legacy(self, z, pos0, pos1, pos2, batch):
        """
        Complete forward pass.

        Args:
            z (torch.Tensor): Atomic numbers.
            pos0 (torch.Tensor): First atomic position.
            pos1 (torch.Tensor): Second atomic position.
            pos2 (torch.Tensor): Third atomic position.
            batch (torch.Tensor): Batch indices.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                (node embeddings, graph embeddings, predictions)
        """
        return self.forward(pos0, pos1, pos2, z, batch)



