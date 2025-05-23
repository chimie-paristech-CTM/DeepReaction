import os
import json
import torch
import pandas as pd
from typing import Optional, List, Union, Dict, Any
from pathlib import Path

from .core.config import Config, DatasetConfig, ModelConfig, TrainingConfig, SystemConfig
from .core.dataset import ReactionDataset
from .core.trainer import ReactionTrainer
from .core.predictor import ReactionPredictor


class DeepReaction:
    def __init__(
            self,
            target_fields: List[str] = None,
            input_features: List[str] = None,
            target_weights: List[float] = None,
            file_keywords: List[str] = None,
            id_field: str = 'ID',
            dir_field: str = 'R_dir',
            reaction_field: str = 'smiles',
            readout: str = 'mean',
            use_scaler: bool = True,
            model_type: str = 'dimenet++',
            node_dim: int = 128,
            dropout: float = 0.1,
            use_layer_norm: bool = False,
            activation: str = 'silu',
            hidden_channels: int = 128,
            num_blocks: int = 5,
            int_emb_size: int = 64,
            basis_emb_size: int = 8,
            out_emb_channels: int = 256,
            num_spherical: int = 7,
            num_radial: int = 6,
            cutoff: float = 5.0,
            envelope_exponent: int = 5,
            num_before_skip: int = 1,
            num_after_skip: int = 2,
            num_output_layers: int = 3,
            max_num_neighbors: int = 32,
            use_xtb_features: bool = True,
            max_num_atoms: int = 100,
            readout_hidden_dim: int = 128,
            readout_num_heads: int = 4,
            readout_num_sabs: int = 2,
            prediction_hidden_layers: int = 3,
            prediction_hidden_dim: int = 512,
            batch_size: int = 16,
            eval_batch_size: Optional[int] = None,
            lr: float = 0.0005,
            optimizer: str = 'adamw',
            scheduler: str = 'warmup_cosine',
            warmup_epochs: int = 10,
            min_lr: float = 1e-7,
            weight_decay: float = 0.0001,
            loss_function: str = 'mse',
            gradient_clip_val: float = 0.0,
            gradient_accumulation_steps: int = 1,
            early_stopping_patience: int = 40,
            early_stopping_min_delta: float = 0.0001,
            random_seed: int = 42234,
            cuda: bool = True,
            gpu_id: int = 0,
            num_workers: int = 4,
            strategy: str = 'auto',
            num_nodes: int = 1,
            devices: int = 1,
            log_level: str = 'info',
            log_to_file: bool = False,
            matmul_precision: str = 'high',
            precision: str = '32',
            train_ratio: float = 0.8,
            val_ratio: float = 0.1,
            test_ratio: float = 0.1,
            cv_folds: int = 0,
            val_csv: Optional[str] = None,
            test_csv: Optional[str] = None,
            cv_test_fold: int = -1,
            cv_stratify: bool = False,
            cv_grouped: bool = True,
            **kwargs
    ):
        self.target_fields = target_fields or ['DG_act', 'DrG']
        self.input_features = input_features or ['DG_act_xtb', 'DrG_xtb']
        self.target_weights = target_weights
        self.file_keywords = file_keywords or ['reactant', 'ts', 'product']

        self.dataset_params = {
            'dataset_root': './dataset',
            'dataset_csv': './dataset/dataset.csv',
            'train_ratio': train_ratio,
            'val_ratio': val_ratio,
            'test_ratio': test_ratio,
            'target_fields': self.target_fields,
            'target_weights': self.target_weights,
            'input_features': self.input_features,
            'file_keywords': self.file_keywords,
            'readout': readout,
            'id_field': id_field,
            'dir_field': dir_field,
            'reaction_field': reaction_field,
            'cv_folds': cv_folds,
            'use_scaler': use_scaler,
            'val_csv': val_csv,
            'test_csv': test_csv,
            'cv_test_fold': cv_test_fold,
            'cv_stratify': cv_stratify,
            'cv_grouped': cv_grouped,
            'num_workers': num_workers
        }

        self.model_params = {
            'model_type': model_type,
            'node_dim': node_dim,
            'dropout': dropout,
            'use_layer_norm': use_layer_norm,
            'activation': activation,
            'hidden_channels': hidden_channels,
            'num_blocks': num_blocks,
            'int_emb_size': int_emb_size,
            'basis_emb_size': basis_emb_size,
            'out_emb_channels': out_emb_channels,
            'num_spherical': num_spherical,
            'num_radial': num_radial,
            'cutoff': cutoff,
            'envelope_exponent': envelope_exponent,
            'num_before_skip': num_before_skip,
            'num_after_skip': num_after_skip,
            'num_output_layers': num_output_layers,
            'max_num_neighbors': max_num_neighbors,
            'use_xtb_features': use_xtb_features,
            'max_num_atoms': max_num_atoms,
            'readout_hidden_dim': readout_hidden_dim,
            'readout_num_heads': readout_num_heads,
            'readout_num_sabs': readout_num_sabs,
            'prediction_hidden_layers': prediction_hidden_layers,
            'prediction_hidden_dim': prediction_hidden_dim
        }

        self.training_params = {
            'batch_size': batch_size,
            'eval_batch_size': eval_batch_size,
            'lr': lr,
            'finetune_lr': kwargs.get('finetune_lr'),
            'max_epochs': kwargs.get('max_epochs', 10),
            'min_epochs': kwargs.get('min_epochs', 0),
            'early_stopping_patience': early_stopping_patience,
            'early_stopping_min_delta': early_stopping_min_delta,
            'optimizer': optimizer,
            'scheduler': scheduler,
            'warmup_epochs': warmup_epochs,
            'min_lr': min_lr,
            'weight_decay': weight_decay,
            'random_seed': random_seed,
            'loss_function': loss_function,
            'gradient_clip_val': gradient_clip_val,
            'gradient_accumulation_steps': gradient_accumulation_steps,
            'out_dir': kwargs.get('out_dir', './results/deepreaction_model'),
            'save_best_model': kwargs.get('save_best_model', True),
            'save_last_model': kwargs.get('save_last_model', False),
            'save_predictions': kwargs.get('save_predictions', True),
            'save_interval': kwargs.get('save_interval', 0),
            'checkpoint_path': kwargs.get('checkpoint_path'),
            'mode': 'train',
            'freeze_base_model': kwargs.get('freeze_base_model', False),
            'precision': precision
        }

        self.system_params = {
            'cuda': cuda,
            'gpu_id': gpu_id,
            'num_workers': num_workers,
            'strategy': strategy,
            'num_nodes': num_nodes,
            'devices': devices,
            'log_level': log_level,
            'log_to_file': log_to_file,
            'matmul_precision': matmul_precision
        }

        for key, value in kwargs.items():
            if key not in self.dataset_params and key not in self.model_params and key not in self.training_params and key not in self.system_params:
                print(f"Warning: Unknown parameter '{key}' ignored")

        self.config = None
        self.trainer = None
        self.predictor = None
        self.best_model_path = None
        self.fitted = False

    def _create_config(self, dataset_root: Optional[str] = None, dataset_csv: Optional[str] = None):
        dataset_params = self.dataset_params.copy()
        if dataset_root:
            dataset_params['dataset_root'] = dataset_root
            if not dataset_csv:
                csv_files = list(Path(dataset_root).glob('*.csv'))
                if csv_files:
                    dataset_params['dataset_csv'] = str(csv_files[0])
                else:
                    dataset_params['dataset_csv'] = os.path.join(dataset_root, 'dataset.csv')
        if dataset_csv:
            dataset_params['dataset_csv'] = dataset_csv

        dataset_config = DatasetConfig(**dataset_params)
        model_config = ModelConfig(**self.model_params)
        training_config = TrainingConfig(**self.training_params)
        system_config = SystemConfig(**self.system_params)

        return Config(dataset_config, model_config, training_config, system_config)

    def fit(
            self,
            dataset_path: str,
            dataset_csv: Optional[str] = None,
            epochs: Optional[int] = None,
            out_dir: Optional[str] = None,
            checkpoint_path: Optional[str] = None,
            val_csv: Optional[str] = None,
            test_csv: Optional[str] = None,
            **kwargs
    ):
        if epochs is not None:
            self.training_params['max_epochs'] = epochs
        if out_dir is not None:
            self.training_params['out_dir'] = out_dir
        if checkpoint_path is not None:
            self.training_params['checkpoint_path'] = checkpoint_path
        if val_csv is not None:
            self.dataset_params['val_csv'] = val_csv
        if test_csv is not None:
            self.dataset_params['test_csv'] = test_csv

        for key, value in kwargs.items():
            if key in ['batch_size', 'lr', 'weight_decay', 'gradient_clip_val']:
                self.training_params[key] = value
            elif key in ['train_ratio', 'val_ratio', 'test_ratio']:
                self.dataset_params[key] = value

        self.config = self._create_config(dataset_path, dataset_csv)

        dataset = ReactionDataset(config=self.config)
        train_data, val_data, test_data, scalers = dataset.get_data_splits()

        self.trainer = ReactionTrainer(config=self.config)

        metrics = self.trainer.fit(
            train_dataset=train_data,
            val_dataset=val_data,
            test_dataset=test_data,
            scalers=scalers,
            checkpoint_path=self.training_params.get('checkpoint_path'),
            mode='train'
        )

        self.best_model_path = metrics.get('best_model_path')
        self.fitted = True

        model_info = {
            'config': self.config.to_dict(),
            'best_model_path': self.best_model_path,
            'training_metrics': metrics
        }

        info_path = os.path.join(self.training_params['out_dir'], 'model_info.json')
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=2)

        return metrics

    def predict(
            self,
            data_path: str,
            checkpoint_path: Optional[str] = None,
            output_dir: Optional[str] = None,
            batch_size: Optional[int] = None
    ) -> pd.DataFrame:
        if not self.fitted and not checkpoint_path:
            raise ValueError("Model not fitted. Call fit() first or provide checkpoint_path.")

        if checkpoint_path is None:
            checkpoint_path = self.best_model_path

        if checkpoint_path is None:
            raise ValueError("No checkpoint found. Train the model first or provide checkpoint_path.")

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        if output_dir is None:
            output_dir = './predictions'

        if batch_size is not None:
            self.training_params['batch_size'] = batch_size

        dataset_root = os.path.dirname(data_path)
        self.config = self._create_config(dataset_root, data_path)

        self.predictor = ReactionPredictor(config=self.config, checkpoint_path=checkpoint_path)

        results = self.predictor.predict_from_csv(data_path, output_dir=output_dir)

        return results

    def save_config(self, path: str):
        if self.config is None:
            self.config = self._create_config()
        self.config.save(path)

    def load_config(self, path: str):
        self.config = Config.load(path)

        self.dataset_params.update(self.config.dataset.__dict__)
        self.model_params.update(self.config.model.__dict__)
        self.training_params.update(self.config.training.__dict__)
        self.system_params.update(self.config.system.__dict__)

        self.target_fields = self.dataset_params.get('target_fields')
        self.input_features = self.dataset_params.get('input_features')

    @classmethod
    def from_checkpoint(
            cls,
            checkpoint_path: str,
            config_path: Optional[str] = None
    ) -> 'DeepReaction':
        if config_path is None:
            model_dir = os.path.dirname(os.path.dirname(checkpoint_path))
            config_path = os.path.join(model_dir, 'config.json')

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        config = Config.load(config_path)

        params = {}
        params.update(config.dataset.__dict__)
        params.update(config.model.__dict__)
        params.update(config.training.__dict__)
        params.update(config.system.__dict__)

        model = cls(**params)
        model.config = config
        model.best_model_path = checkpoint_path
        model.fitted = True

        return model

    def get_params(self) -> Dict[str, Any]:
        params = {}
        params.update(self.dataset_params)
        params.update(self.model_params)
        params.update(self.training_params)
        params.update(self.system_params)
        return params

    def set_params(self, **params):
        for key, value in params.items():
            if key in self.dataset_params:
                self.dataset_params[key] = value
            elif key in self.model_params:
                self.model_params[key] = value
            elif key in self.training_params:
                self.training_params[key] = value
            elif key in self.system_params:
                self.system_params[key] = value
            else:
                print(f"Warning: Unknown parameter '{key}'")

            if key == 'target_fields':
                self.target_fields = value
            elif key == 'input_features':
                self.input_features = value

        return self