import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict


@dataclass
class DatasetConfig:
    dataset_root: str = './dataset/DATASET_DA_F'
    dataset_csv: str = './dataset/DATASET_DA_F/dataset_xtb_final.csv'
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    target_fields: List[str] = None
    target_weights: List[float] = None
    input_features: List[str] = None
    file_keywords: List[str] = None
    readout: str = 'mean'
    id_field: str = 'ID'
    dir_field: str = 'R_dir'
    reaction_field: str = 'smiles'
    cv_folds: int = 0
    use_scaler: bool = True
    val_csv: Optional[str] = None
    test_csv: Optional[str] = None
    cv_test_fold: int = -1
    cv_stratify: bool = False
    cv_grouped: bool = True
    num_workers: int = 4
    
    def __post_init__(self):
        if self.target_fields is None:
            self.target_fields = ['DG_act', 'DrG']
        if self.target_weights is None:
            self.target_weights = [1.0] * len(self.target_fields)
        if self.input_features is None:
            self.input_features = ['DG_act_xtb', 'DrG_xtb']
        if self.file_keywords is None:
            self.file_keywords = ['reactant', 'ts', 'product']
        if len(self.target_weights) != len(self.target_fields):
            self.target_weights = [1.0] * len(self.target_fields)


@dataclass
class ModelConfig:
    model_type: str = 'dimenet++'
    node_dim: int = 128
    dropout: float = 0.1
    use_layer_norm: bool = False
    activation: str = 'silu'
    hidden_channels: int = 128
    num_blocks: int = 5
    int_emb_size: int = 64
    basis_emb_size: int = 8
    out_emb_channels: int = 256
    num_spherical: int = 7
    num_radial: int = 6
    cutoff: float = 5.0
    envelope_exponent: int = 5
    num_before_skip: int = 1
    num_after_skip: int = 2
    num_output_layers: int = 3
    max_num_neighbors: int = 32
    use_xtb_features: bool = True
    max_num_atoms: int = 100
    readout_hidden_dim: int = 128
    readout_num_heads: int = 4
    readout_num_sabs: int = 2
    prediction_hidden_layers: int = 3
    prediction_hidden_dim: int = 512


@dataclass
class TrainingConfig:
    batch_size: int = 16
    eval_batch_size: Optional[int] = None
    lr: float = 0.0005
    finetune_lr: Optional[float] = None
    max_epochs: int = 10
    min_epochs: int = 0
    early_stopping_patience: int = 40
    early_stopping_min_delta: float = 0.0001
    optimizer: str = 'adamw'
    scheduler: str = 'warmup_cosine'
    warmup_epochs: int = 10
    min_lr: float = 1e-7
    weight_decay: float = 0.0001
    random_seed: int = 42234
    loss_function: str = 'mse'
    gradient_clip_val: float = 0.0
    gradient_accumulation_steps: int = 1
    out_dir: str = './results/reaction_model'
    save_best_model: bool = True
    save_last_model: bool = False
    save_predictions: bool = True
    save_interval: int = 0
    checkpoint_path: Optional[str] = None
    mode: str = 'train'
    freeze_base_model: bool = False
    precision: str = '32'
    
    def __post_init__(self):
        if self.eval_batch_size is None:
            self.eval_batch_size = self.batch_size


@dataclass
class SystemConfig:
    cuda: bool = True
    gpu_id: int = 0
    num_workers: int = 4
    strategy: str = 'auto'
    num_nodes: int = 1
    devices: int = 1
    log_level: str = 'info'
    log_to_file: bool = False
    matmul_precision: str = 'high'


class Config:
    def __init__(self, dataset: DatasetConfig, model: ModelConfig, training: TrainingConfig, system: SystemConfig):
        self.dataset = dataset
        self.model = model
        self.training = training
        self.system = system
        self._validate_config()
    
    def _validate_config(self):
        if not os.path.exists(self.dataset.dataset_csv):
            print(f"Warning: Dataset CSV not found: {self.dataset.dataset_csv}")
        
        if not os.path.exists(self.dataset.dataset_root):
            print(f"Warning: Dataset root not found: {self.dataset.dataset_root}")
        
        if abs(self.dataset.train_ratio + self.dataset.val_ratio + self.dataset.test_ratio - 1.0) > 1e-6:
            if self.dataset.cv_folds == 0 and not (self.dataset.val_csv and self.dataset.test_csv):
                raise ValueError("Train, validation, and test ratios must sum to 1.0")
        
        if len(self.dataset.input_features) != len(self.dataset.target_fields):
            print(f"Warning: input_features ({len(self.dataset.input_features)}) and target_fields ({len(self.dataset.target_fields)}) counts don't match")
        
        if len(self.dataset.target_weights) != len(self.dataset.target_fields):
            print(f"Warning: target_weights length adjusted to match target_fields")
            self.dataset.target_weights = [1.0] * len(self.dataset.target_fields)
        
        if self.dataset.cv_folds > 0 and self.dataset.cv_test_fold >= self.dataset.cv_folds:
            raise ValueError(f"cv_test_fold ({self.dataset.cv_test_fold}) must be < cv_folds ({self.dataset.cv_folds})")
        
        if self.training.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        
        if self.training.lr <= 0:
            raise ValueError("learning rate must be positive")
    
    @classmethod
    def from_params(cls, params: Dict[str, Any]) -> 'Config':
        dataset_params = {}
        model_params = {}
        training_params = {}
        system_params = {}
        
        dataset_fields = {f.name for f in DatasetConfig.__dataclass_fields__.values()}
        model_fields = {f.name for f in ModelConfig.__dataclass_fields__.values()}
        training_fields = {f.name for f in TrainingConfig.__dataclass_fields__.values()}
        system_fields = {f.name for f in SystemConfig.__dataclass_fields__.values()}
        
        for key, value in params.items():
            if key in dataset_fields:
                dataset_params[key] = value
            elif key in model_fields:
                model_params[key] = value
            elif key in training_fields:
                training_params[key] = value
            elif key in system_fields:
                system_params[key] = value
            else:
                print(f"Warning: Unknown parameter '{key}' with value '{value}' ignored")
        
        try:
            dataset_config = DatasetConfig(**dataset_params)
            model_config = ModelConfig(**model_params)
            training_config = TrainingConfig(**training_params)
            system_config = SystemConfig(**system_params)
        except Exception as e:
            print(f"Error creating configuration: {e}")
            raise
        
        return cls(dataset_config, model_config, training_config, system_config)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'dataset': asdict(self.dataset),
            'model': asdict(self.model),
            'training': asdict(self.training),
            'system': asdict(self.system)
        }
    
    def save(self, path: str):
        import json
        try:
            with open(path, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
        except Exception as e:
            print(f"Error saving configuration to {path}: {e}")
            raise
    
    @classmethod
    def load(cls, path: str) -> 'Config':
        import json
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            
            dataset_config = DatasetConfig(**data['dataset'])
            model_config = ModelConfig(**data['model'])
            training_config = TrainingConfig(**data['training'])
            system_config = SystemConfig(**data['system'])
            
            return cls(dataset_config, model_config, training_config, system_config)
        except Exception as e:
            print(f"Error loading configuration from {path}: {e}")
            raise
    
    def print_config(self):
        print("=" * 50)
        print("CONFIGURATION")
        print("=" * 50)
        
        print("\n[DATASET CONFIG]")
        for key, value in asdict(self.dataset).items():
            print(f"  {key}: {value}")
        
        print("\n[MODEL CONFIG]")
        for key, value in asdict(self.model).items():
            print(f"  {key}: {value}")
        
        print("\n[TRAINING CONFIG]")
        for key, value in asdict(self.training).items():
            print(f"  {key}: {value}")
        
        print("\n[SYSTEM CONFIG]")
        for key, value in asdict(self.system).items():
            print(f"  {key}: {value}")
        
        print("=" * 50)