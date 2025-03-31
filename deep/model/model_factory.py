import torch
import torch.nn as nn
from typing import Dict, Any


class ModelFactory:
    @staticmethod
    def create_model(model_type: str, **model_params):
        """
        Factory method to create a model based on the specified type.

        Args:
            model_type (str): The type of model to create.
            **model_params: Additional parameters for model initialization.

        Returns:
            The initialized model instance.
        """
        model_type = model_type.lower()

        if model_type == 'dimenet++':
            from .dimenetplusplus import DimeNetPlusPlus
            return DimeNetPlusPlus(**ModelFactory._get_model_params(model_params))
        elif model_type == 'schnet':
            # Currently using DimeNetPlusPlus as fallback
            from .dimenetplusplus import DimeNetPlusPlus
            return DimeNetPlusPlus(**ModelFactory._get_model_params(model_params))
        elif model_type == 'egnn':
            # Currently using DimeNetPlusPlus as fallback
            from .dimenetplusplus import DimeNetPlusPlus
            return DimeNetPlusPlus(**ModelFactory._get_model_params(model_params))
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    @staticmethod
    def _get_model_params(params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract and normalize model parameters with defaults.

        Args:
            params (Dict[str, Any]): Input parameters.

        Returns:
            Dict[str, Any]: Normalized parameters with defaults.
        """
        return {
            'hidden_channels': params.get('hidden_channels', 128),
            'out_channels': params.get('out_channels', 128),
            'num_blocks': params.get('num_blocks', 4),
            'int_emb_size': params.get('int_emb_size', 64),
            'basis_emb_size': params.get('basis_emb_size', 8),
            'out_emb_channels': params.get('out_emb_channels', 256),
            'num_spherical': params.get('num_spherical', 7),
            'num_radial': params.get('num_radial', 6),
            'cutoff': params.get('cutoff', 5.0),
            'max_num_neighbors': params.get('max_num_neighbors', 32),
            'envelope_exponent': params.get('envelope_exponent', 5),
            'num_before_skip': params.get('num_before_skip', 1),
            'num_after_skip': params.get('num_after_skip', 2),
            'num_output_layers': params.get('num_output_layers', 3),
        }