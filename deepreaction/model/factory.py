import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union, Type
from .base import BaseReactionModel

class ModelRegistry:
    _models = {}
    
    @classmethod
    def register(cls, name: str, model_class: Type[BaseReactionModel]):
        cls._models[name.lower()] = model_class
    
    @classmethod
    def get_model_class(cls, model_type: str) -> Type[BaseReactionModel]:
        model_type = model_type.lower()
        if model_type == 'schnet':
            from .dimenetplusplus import DimeNetPlusPlus
            print("Warning: SchNet not implemented yet, using DimeNet++ instead.")
            return DimeNetPlusPlus
        elif model_type == 'egnn':
            from .dimenetplusplus import DimeNetPlusPlus
            print("Warning: EGNN not implemented yet, using DimeNet++ instead.")
            return DimeNetPlusPlus
        elif model_type not in cls._models:
            raise ValueError(f"Unsupported model type: {model_type}. Available models: {list(cls._models.keys())}")
        return cls._models[model_type]
    
    @classmethod
    def get_default_params(cls, model_type: str) -> Dict[str, Any]:
        model_class = cls.get_model_class(model_type)
        return model_class.get_default_params()
    
    @classmethod
    def list_available_models(cls):
        return list(cls._models.keys())

class ModelFactory:
    @staticmethod
    def create_model(model_type: str, **model_params):
        model_class = ModelRegistry.get_model_class(model_type)
        default_params = model_class.get_default_params()
        
        for key, value in default_params.items():
            if key not in model_params:
                model_params[key] = value
        
        return model_class(**model_params)

def create_model(model_type: str, **kwargs):
    return ModelFactory.create_model(model_type, **kwargs)

from .dimenetplusplus import DimeNetPlusPlus
ModelRegistry.register('dimenet++', DimeNetPlusPlus)