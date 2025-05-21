import os
import yaml
import json
import argparse
from dataclasses import dataclass, field, asdict, fields, is_dataclass
from typing import List, Dict, Optional, Any, Union, Type, TypeVar, get_type_hints
from datetime import datetime

T = TypeVar('T')

def _convert_to_instance(cls: Type[T], data: Union[Dict, T]) -> T:
    """Convert dictionary to instance of specified class, or return if already an instance."""
    if isinstance(data, cls):
        return data
    elif isinstance(data, dict):
        # Get field names and their types from dataclass
        fieldtypes = get_type_hints(cls)
        init_kwargs = {}
        
        for key, value in data.items():
            if key in fieldtypes:
                field_type = fieldtypes[key]
                
                # Handle nested dataclasses
                if value is not None and is_dataclass(field_type) and isinstance(value, dict):
                    init_kwargs[key] = _convert_to_instance(field_type, value)
                else:
                    init_kwargs[key] = value
        
        return cls(**init_kwargs)
    else:
        raise TypeError(f"Cannot convert {type(data)} to {cls}")

@dataclass
class ReactionConfig:
    dataset_root: str = "./dataset"
    dataset_csv: str = "./dataset/data.csv"
    target_fields: List[str] = field(default_factory=lambda: ["G(TS)", "DrG"])
    file_patterns: List[str] = field(default_factory=lambda: ["*_reactant.xyz", "*_ts.xyz", "*_product.xyz"])
    input_features: List[str] = field(default_factory=lambda: ["G(TS)_xtb", "DrG_xtb"])
    use_scaler: bool = True
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    cv_folds: int = 0
    cv_test_fold: int = -1
    cv_stratify: bool = False
    cv_grouped: bool = True
    id_field: str = "ID"
    dir_field: str = "R_dir"
    reaction_field: str = "reaction"
    random_seed: int = 42234
    force_reload: bool = False
    inference_mode: bool = False
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        return _convert_to_instance(cls, config_dict)

@dataclass
class ModelConfig:
    model_type: str = "dimenet++"
    readout: str = "mean"
    hidden_channels: int = 128
    num_blocks: int = 5
    cutoff: float = 5.0
    int_emb_size: int = 64
    basis_emb_size: int = 8
    out_emb_channels: int = 256
    num_spherical: int = 7
    num_radial: int = 6
    envelope_exponent: int = 5
    num_before_skip: int = 1
    num_after_skip: int = 2
    num_output_layers: int = 3
    max_num_neighbors: int = 32
    node_dim: int = 128
    dropout: float = 0.1
    use_layer_norm: bool = False
    use_xtb_features: bool = True
    num_xtb_features: int = 2
    prediction_hidden_layers: int = 3
    prediction_hidden_dim: int = 512
    max_num_atoms: int = 100
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        return _convert_to_instance(cls, config_dict)
    
    def __post_init__(self):
        # Auto-adjust parameters if needed
        if self.num_xtb_features == 2 and not self.use_xtb_features:
            self.use_xtb_features = True

@dataclass
class TrainingConfig:
    output_dir: str = "./results/reaction_model"
    batch_size: int = 16
    learning_rate: float = 0.0005
    max_epochs: int = 100
    min_epochs: int = 0
    early_stopping_patience: int = 40
    save_best_model: bool = True
    save_last_model: bool = False
    optimizer: str = "adamw"
    weight_decay: float = 0.0001
    scheduler: str = "warmup_cosine"
    warmup_epochs: int = 10
    min_lr: float = 1e-7
    loss_function: str = "mse"
    target_weights: Optional[List[float]] = field(default_factory=lambda: [1.0, 1.0])
    gradient_clip_val: float = 0.0
    gpu: bool = True
    precision: str = "32"
    log_every_n_steps: int = 50
    save_predictions: bool = True
    num_workers: int = 4
    resume_from_checkpoint: Optional[str] = None
    mode: str = "continue"
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        return _convert_to_instance(cls, config_dict)

@dataclass
class Config:
    reaction: ReactionConfig = field(default_factory=ReactionConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    experiment_name: Optional[str] = None
    
    def __post_init__(self):
        if self.experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_name = f"{self.model.model_type}_{self.model.readout}_seed{self.reaction.random_seed}_{timestamp}"
            
        # Auto-adjust num_xtb_features based on input_features if needed
        if self.model.use_xtb_features and len(self.reaction.input_features) > 0:
            self.model.num_xtb_features = len(self.reaction.input_features)
            
        # Auto-adjust target_weights if needed
        if len(self.training.target_weights) != len(self.reaction.target_fields):
            self.training.target_weights = [1.0] * len(self.reaction.target_fields)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reaction": asdict(self.reaction),
            "model": asdict(self.model),
            "training": asdict(self.training),
            "experiment_name": self.experiment_name
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        # Process nested config objects
        reaction_config = ReactionConfig.from_dict(config_dict.get("reaction", {}))
        model_config = ModelConfig.from_dict(config_dict.get("model", {}))
        training_config = TrainingConfig.from_dict(config_dict.get("training", {}))
        experiment_name = config_dict.get("experiment_name")

        return cls(
            reaction=reaction_config,
            model=model_config,
            training=training_config,
            experiment_name=experiment_name
        )
    
    @classmethod
    def from_params(cls, params: Dict[str, Any]):
        """Create a Config object directly from a flat dictionary of parameters."""
        
        # Extract reaction config parameters
        reaction_params = {
            k: v for k, v in params.items() 
            if k in [f.name for f in fields(ReactionConfig)]
        }
        
        # Extract model config parameters
        model_params = {
            k: v for k, v in params.items() 
            if k in [f.name for f in fields(ModelConfig)]
        }
        
        # Handle derived parameters
        model_params['use_xtb_features'] = len(params.get('input_features', [])) > 0
        model_params['num_xtb_features'] = len(params.get('input_features', []))
        
        # Extract training config parameters
        training_params = {
            k: v for k, v in params.items() 
            if k in [f.name for f in fields(TrainingConfig)]
        }
        
        # Handle parameter name differences
        if 'lr' in params:
            training_params['learning_rate'] = params['lr']
        if 'epochs' in params:
            training_params['max_epochs'] = params['epochs']
        if 'early_stopping' in params:
            training_params['early_stopping_patience'] = params['early_stopping']
        if 'out_dir' in params:
            training_params['output_dir'] = params['out_dir']
        if 'cuda' in params:
            training_params['gpu'] = params['cuda']
        if 'checkpoint_path' in params:
            training_params['resume_from_checkpoint'] = params['checkpoint_path']
        if 'mode' in params:
            training_params['mode'] = params['mode']
        
        # Create the config objects
        reaction_config = ReactionConfig.from_dict(reaction_params)
        model_config = ModelConfig.from_dict(model_params)
        training_config = TrainingConfig.from_dict(training_params)
        
        return cls(
            reaction=reaction_config,
            model=model_config,
            training=training_config
        )

def load_config(config_path):
    """Load configuration from a YAML or JSON file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    file_ext = os.path.splitext(config_path)[1].lower()

    with open(config_path, 'r') as f:
        if file_ext in ['.yaml', '.yml']:
            config_dict = yaml.safe_load(f)
        elif file_ext == '.json':
            config_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {file_ext}")

    return Config.from_dict(config_dict)

def save_config(config, output_path):
    """Save configuration to a YAML and JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    config_dict = config.to_dict()

    yaml_path = f"{os.path.splitext(output_path)[0]}.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)

    json_path = f"{os.path.splitext(output_path)[0]}.json"
    with open(json_path, 'w') as f:
        json.dump(config_dict, f, indent=2)

    return yaml_path, json_path