import os
import time
import json
import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    TQDMProgressBar
)
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from typing import Dict, List, Optional, Tuple, Union, Any

from deepreaction.config.config import Config, ModelConfig, TrainingConfig

class ReactionTrainer:
    def __init__(
            self,
            config: Optional[Config] = None,
            **kwargs
    ):
        """
        Initialize the ReactionTrainer with a unified configuration interface.
        
        Args:
            config: A complete Config object containing all configuration
            **kwargs: Individual parameters (will override config if both provided)
        """
        # Handle configuration
        if config is None:
            # Create a config from kwargs
            from deepreaction.config.config import Config
            self.config = Config.from_params(kwargs)
        else:
            self.config = config
            
            # Override config with any explicitly provided kwargs
            for section in [self.config.model, self.config.training, self.config.reaction]:
                for key, value in kwargs.items():
                    if hasattr(section, key):
                        setattr(section, key, value)
        
        # Extract commonly used config values for convenience
        self.model_type = self.config.model.model_type
        self.readout = self.config.model.readout
        self.batch_size = self.config.training.batch_size
        self.max_epochs = self.config.training.max_epochs
        self.min_epochs = self.config.training.min_epochs
        self.learning_rate = self.config.training.learning_rate
        self.output_dir = self.config.training.output_dir
        self.early_stopping_patience = self.config.training.early_stopping_patience
        self.save_best_model = self.config.training.save_best_model
        self.save_last_model = self.config.training.save_last_model
        self.random_seed = self.config.reaction.random_seed
        self.gpu = self.config.training.gpu
        
        # Other required attributes
        self.num_targets = len(self.config.reaction.target_fields)
        self.target_field_names = self.config.reaction.target_fields
        self.use_scaler = self.config.reaction.use_scaler
        self.scalers = kwargs.get('scalers', None)
        
        # Setup model and trainer
        self.model = None
        self.lightning_model = None
        self.trainer = None

        # Set random seed
        pl.seed_everything(self.random_seed)

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

    def setup_callbacks(self, run_dir=None):
        callbacks = []

        checkpoint_dir = os.path.join(run_dir if run_dir else self.output_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)

        if self.save_best_model:
            callbacks.append(ModelCheckpoint(
                monitor='val_total_loss',
                dirpath=checkpoint_dir,
                filename='best-{epoch:04d}-{val_total_loss:.4f}',
                save_top_k=1,
                mode='min',
                save_weights_only=False,
                verbose=True
            ))

        if self.save_last_model:
            callbacks.append(ModelCheckpoint(
                dirpath=checkpoint_dir,
                filename='last-{epoch:04d}',
                save_top_k=1,
                save_last=True,
                verbose=False
            ))

        if self.early_stopping_patience > 0:
            callbacks.append(EarlyStopping(
                monitor='val_total_loss',
                patience=self.early_stopping_patience,
                min_delta=0.0001,
                mode='min',
                verbose=True
            ))

        callbacks.append(LearningRateMonitor(logging_interval='epoch'))
        callbacks.append(TQDMProgressBar(refresh_rate=10))

        return callbacks

    def setup_loggers(self, run_dir=None):
        loggers = []

        output_dir = run_dir if run_dir else self.output_dir

        loggers.append(TensorBoardLogger(
            save_dir=output_dir,
            name='tensorboard',
            default_hp_metric=False
        ))

        loggers.append(CSVLogger(
            save_dir=output_dir,
            name='csv_logs',
            flush_logs_every_n_steps=self.config.training.log_every_n_steps
        ))

        return loggers

    def create_model(self):
        from deepreaction.module.pl_wrap import Estimator

        # Create model config dictionary
        model_config = {
            'model_type': self.model_type,
            'readout': self.readout,
            'batch_size': self.batch_size,
            'lr': self.learning_rate,
            'max_num_atoms_in_mol': self.config.model.max_num_atoms,
            'scaler': self.scalers,
            'use_layer_norm': self.config.model.use_layer_norm,
            'node_latent_dim': self.config.model.node_dim,
            'dropout': self.config.model.dropout,
            'optimizer': self.config.training.optimizer,
            'weight_decay': self.config.training.weight_decay,
            'scheduler': self.config.training.scheduler,
            'warmup_epochs': self.config.training.warmup_epochs,
            'min_lr': self.config.training.min_lr,
            'loss_function': self.config.training.loss_function,
            'target_weights': self.config.training.target_weights,
            'use_xtb_features': self.config.model.use_xtb_features,
            'num_xtb_features': self.config.model.num_xtb_features,
            'prediction_hidden_layers': self.config.model.prediction_hidden_layers,
            'prediction_hidden_dim': self.config.model.prediction_hidden_dim,
            'target_field_names': self.target_field_names,
            'num_targets': self.num_targets
        }

        # Add model-specific parameters
        model_kwargs = {}
        for key in ['hidden_channels', 'num_blocks', 'int_emb_size', 'basis_emb_size',
                    'out_emb_channels', 'num_spherical', 'num_radial', 'cutoff',
                    'envelope_exponent', 'num_before_skip', 'num_after_skip',
                    'num_output_layers', 'max_num_neighbors']:
            if hasattr(self.config.model, key):
                model_kwargs[key] = getattr(self.config.model, key)

        model_config['model_kwargs'] = model_kwargs

        # Add readout-specific parameters
        readout_kwargs = {}
        readout_params = {
            'set_transformer_hidden_dim': 512,
            'set_transformer_num_heads': 16,
            'set_transformer_num_sabs': 2,
            'attention_hidden_dim': 256,
            'attention_num_heads': 8
        }
        
        for key, default_value in readout_params.items():
            readout_kwargs[key] = getattr(self.config.model, key, default_value)

        model_config['readout_kwargs'] = readout_kwargs

        self.lightning_model = Estimator(**model_config)

        return self.lightning_model

    def fit(self, train_dataset=None, val_dataset=None, test_dataset=None, checkpoint_path=None, mode=None,
            run_dir=None):
        """
        Train the model on the given datasets.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            test_dataset: Test dataset (optional)
            checkpoint_path: Path to checkpoint to resume from (overrides config value if provided)
            mode: 'continue' or 'finetune' (overrides config value if provided)
            run_dir: Directory to save outputs (overrides config.training.output_dir if provided)
        
        Returns:
            Dictionary of training metrics
        """
        output_dir = run_dir if run_dir else self.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Use provided values or fall back to config values
        checkpoint_path = checkpoint_path or self.config.training.resume_from_checkpoint
        mode = mode or self.config.training.mode

        if train_dataset is None and val_dataset is None:
            raise ValueError("At least one of train_dataset or val_dataset must be provided")

        # Create data loaders
        train_loader, val_loader, test_loader = self._create_data_loaders(train_dataset, val_dataset, test_dataset)

        # Handle checkpoint loading
        if checkpoint_path is not None:
            self._load_from_checkpoint(checkpoint_path, mode)
        else:
            self.create_model()

        # Setup trainer
        callbacks = self.setup_callbacks(run_dir=output_dir)
        loggers = self.setup_loggers(run_dir=output_dir)

        trainer_config = {
            'logger': loggers,
            'callbacks': callbacks,
            'max_epochs': self.max_epochs,
            'min_epochs': self.min_epochs,
            'log_every_n_steps': self.config.training.log_every_n_steps,
            'deterministic': True,
            'accelerator': 'gpu' if self.gpu and torch.cuda.is_available() else 'cpu',
            'devices': 1 if self.gpu and torch.cuda.is_available() else 'auto',
            'num_sanity_val_steps': 2,
        }

        # Add optional configurations
        if self.config.training.gradient_clip_val > 0:
            trainer_config['gradient_clip_val'] = self.config.training.gradient_clip_val

        if self.config.training.precision in ['16', '32', 'bf16', 'mixed']:
            trainer_config['precision'] = self.config.training.precision

        self.trainer = pl.Trainer(**trainer_config)

        # Train the model
        start_time = time.time()
        self.trainer.fit(self.lightning_model, train_loader, val_loader)
        training_time = time.time() - start_time

        # Create training metrics
        train_metrics = self._save_metrics(output_dir, training_time, mode)

        # Evaluate on test set if provided
        if test_loader is not None:
            test_results = self.trainer.test(self.lightning_model, test_loader)
            test_metrics_path = os.path.join(output_dir, 'test_metrics.json')
            with open(test_metrics_path, 'w') as f:
                json.dump(test_results[0], f, indent=2)
            train_metrics['test_metrics'] = test_results[0]

        return train_metrics
    
    def _create_data_loaders(self, train_dataset, val_dataset, test_dataset):
        """Create data loaders for the datasets."""
        from torch_geometric.loader import DataLoader
        follow_batch = ['z0', 'z1', 'z2', 'pos0', 'pos1', 'pos2']
        
        train_loader = None
        if train_dataset is not None:
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.config.training.num_workers,
                follow_batch=follow_batch
            )

        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.config.training.num_workers,
                follow_batch=follow_batch
            )

        test_loader = None
        if test_dataset is not None:
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.config.training.num_workers,
                follow_batch=follow_batch
            )
            
        return train_loader, val_loader, test_loader
    
    def _load_from_checkpoint(self, checkpoint_path, mode):
        """Load model from checkpoint."""
        from deepreaction.module.pl_wrap import Estimator
        
        if mode.lower() == 'continue':
            print(f"Loading model from checkpoint for continued training: {checkpoint_path}")
            self.lightning_model = Estimator.load_from_checkpoint(checkpoint_path)
            self.lightning_model.batch_size = self.batch_size
            self.lightning_model.lr = self.learning_rate
            
        elif mode.lower() == 'finetune':
            print(f"Loading model weights for fine-tuning: {checkpoint_path}")
            loaded_model = Estimator.load_from_checkpoint(checkpoint_path)

            self.create_model()

            self.lightning_model.net.load_state_dict(loaded_model.net.state_dict())
            self.lightning_model.readout_module.load_state_dict(loaded_model.readout_module.state_dict())

            if getattr(self.config.training, 'freeze_base_model', False):
                print("Freezing base model for fine-tuning")
                for param in self.lightning_model.net.parameters():
                    param.requires_grad = False

            finetune_lr = getattr(self.config.training, 'finetune_lr', self.learning_rate * 0.1)
            print(f"Fine-tuning with learning rate: {finetune_lr}")
            self.lightning_model.lr = finetune_lr
            
        else:
            raise ValueError(f"Invalid mode {mode}. Must be 'continue' or 'finetune'")
    
    def _save_metrics(self, output_dir, training_time, mode):
        """Save training metrics and return them."""
        train_metrics = {
            'best_model_path': self.trainer.checkpoint_callback.best_model_path if hasattr(self.trainer, 'checkpoint_callback') else None,
            'training_time': training_time,
            'epochs_completed': self.trainer.current_epoch,
            'mode': mode
        }

        metrics_path = os.path.join(output_dir, 'train_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(train_metrics, f, indent=2)
            
        return train_metrics

    def evaluate(self, test_loader, run_dir=None):
        """Evaluate model on test set."""
        output_dir = run_dir if run_dir else self.output_dir

        if self.trainer is None:
            raise ValueError("Trainer has not been initialized. Call fit() first.")

        test_results = self.trainer.test(self.lightning_model, test_loader)

        with open(os.path.join(output_dir, 'test_metrics.json'), 'w') as f:
            json.dump(test_results[0], f, indent=2)

        return test_results[0]

    def save_model(self, path=None, run_dir=None):
        """Save the model to a file."""
        output_dir = run_dir if run_dir else self.output_dir

        if self.lightning_model is None:
            raise ValueError("Model has not been initialized. Call fit() first.")

        if path is None:
            path = os.path.join(output_dir, 'model.ckpt')

        self.trainer.save_checkpoint(path)
        return path