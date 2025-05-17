import os
import yaml
import json
import argparse
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any, Union
from datetime import datetime


@dataclass
class ReactionConfig:
    dataset_root: Optional[str] = None
    dataset_csv: Optional[str] = None
    target_fields: List[str] = field(default_factory=list)
    file_patterns: List[str] = field(default_factory=list)
    input_features: List[str] = field(default_factory=list)
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
    set_transformer_hidden_dim: int = 512
    set_transformer_num_heads: int = 16
    set_transformer_num_sabs: int = 2
    attention_hidden_dim: int = 256
    attention_num_heads: int = 8


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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reaction": asdict(self.reaction),
            "model": asdict(self.model),
            "training": asdict(self.training),
            "experiment_name": self.experiment_name
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        reaction_config = ReactionConfig(**config_dict.get("reaction", {}))
        model_config = ModelConfig(**config_dict.get("model", {}))
        training_config = TrainingConfig(**config_dict.get("training", {}))
        experiment_name = config_dict.get("experiment_name")

        return cls(
            reaction=reaction_config,
            model=model_config,
            training=training_config,
            experiment_name=experiment_name
        )


def parse_args_to_config(args):
    reaction_config = ReactionConfig(
        dataset_root=args.dataset_root,
        dataset_csv=args.dataset_csv,
        target_fields=args.target_fields,
        file_patterns=args.file_patterns,
        input_features=args.input_features,
        id_field=args.id_field,
        dir_field=args.dir_field,
        reaction_field=args.reaction_field,
        random_seed=args.random_seed
    )

    model_config = ModelConfig()

    training_config = TrainingConfig(
        output_dir=args.output_dir,
        max_epochs=args.epochs,
        resume_from_checkpoint=args.resume_from_checkpoint,
        gpu=args.gpu
    )

    return Config(
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