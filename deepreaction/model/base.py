import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional, List, Union

class BaseReactionModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, z0, z1, z2, pos0, pos1, pos2, batch=None, xtb_features=None) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("Subclasses must implement forward method")
    
    @property
    def name(self) -> str:
        return self.__class__.__name__
    
    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        return {}