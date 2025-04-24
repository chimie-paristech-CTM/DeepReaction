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


class ReactionTrainer:
    def __init__(
            self,
            model_type: str = 'dimenet++',
            readout: str = 'mean',
            batch_size: int = 32,
            max_epochs: int = 100,
            learning_rate: float = 0.0001,
            output_dir: str = './results',
            early_stopping_patience: int = 20,
            save_best_model: bool = True,
            save_last_model: bool = False,
            random_seed: int = 42,
            num_targets: int = 1,
            use_scaler: bool = True,
            scalers=None,
            optimizer: str = 'adam',
            weight_decay: float = 0.0,
            scheduler: str = 'cosine',
            warmup_epochs: int = 10,
            min_lr: float = 1e-6,
            gpu: bool = True,
            precision: str = '32',
            node_dim: int = 128,
            dropout: float = 0.0,
            use_layer_norm: bool = False,
            target_field_names: Optional[List[str]] = None,
            use_xtb_features: bool = False,
            num_xtb_features: int = 0,
            prediction_hidden_layers: int = 3,
            prediction_hidden_dim: int = 128,
            min_epochs: int = 10,
            gradient_clip_val: float = 0.0,
            log_every_n_steps: int = 50,
            target_weights: Optional[List[float]] = None,
            **kwargs
    ):
        self.model_type = model_type
        self.readout = readout
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.min_epochs = min_epochs
        self.learning_rate = learning_rate
        self.output_dir = output_dir
        self.early_stopping_patience = early_stopping_patience
        self.save_best_model = save_best_model
        self.save_last_model = save_last_model
        self.random_seed = random_seed
        self.num_targets = num_targets
        self.use_scaler = use_scaler
        self.scalers = scalers
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.scheduler = scheduler
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr
        self.gpu = gpu
        self.precision = precision
        self.node_dim = node_dim
        self.dropout = dropout
        self.use_layer_norm = use_layer_norm
        self.target_field_names = target_field_names
        self.use_xtb_features = use_xtb_features
        self.num_xtb_features = num_xtb_features
        self.prediction_hidden_layers = prediction_hidden_layers
        self.prediction_hidden_dim = prediction_hidden_dim
        self.gradient_clip_val = gradient_clip_val
        self.log_every_n_steps = log_every_n_steps
        self.target_weights = target_weights
        self.kwargs = kwargs

        self.model = None
        self.lightning_model = None
        self.trainer = None

        pl.seed_everything(random_seed)

        os.makedirs(output_dir, exist_ok=True)

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
            flush_logs_every_n_steps=self.log_every_n_steps
        ))

        return loggers

    def create_model(self):
        from ..module.pl_wrap import Estimator

        model_config = {
            'model_type': self.model_type,
            'readout': self.readout,
            'batch_size': self.batch_size,
            'lr': self.learning_rate,
            'max_num_atoms_in_mol': self.kwargs.get('max_num_atoms', 100),
            'scaler': self.scalers,
            'use_layer_norm': self.use_layer_norm,
            'node_latent_dim': self.node_dim,
            'dropout': self.dropout,
            'optimizer': self.optimizer,
            'weight_decay': self.weight_decay,
            'scheduler': self.scheduler,
            'warmup_epochs': self.warmup_epochs,
            'min_lr': self.min_lr,
            'loss_function': self.kwargs.get('loss_function', 'mse'),
            'target_weights': self.target_weights if self.target_weights is not None else [1.0] * self.num_targets,
            'use_xtb_features': self.use_xtb_features,
            'num_xtb_features': self.num_xtb_features,
            'prediction_hidden_layers': self.prediction_hidden_layers,
            'prediction_hidden_dim': self.prediction_hidden_dim,
            'target_field_names': self.target_field_names
        }

        model_kwargs = {}
        for key in ['hidden_channels', 'num_blocks', 'int_emb_size', 'basis_emb_size',
                    'out_emb_channels', 'num_spherical', 'num_radial', 'cutoff',
                    'envelope_exponent', 'num_before_skip', 'num_after_skip',
                    'num_output_layers', 'max_num_neighbors']:
            if key in self.kwargs:
                model_kwargs[key] = self.kwargs[key]

        model_config['model_kwargs'] = model_kwargs

        readout_kwargs = {}
        for key in ['set_transformer_hidden_dim', 'set_transformer_num_heads',
                    'set_transformer_num_sabs', 'attention_hidden_dim', 'attention_num_heads']:
            if key in self.kwargs:
                readout_kwargs[key] = self.kwargs[key]

        model_config['readout_kwargs'] = readout_kwargs

        self.lightning_model = Estimator(**model_config)

        return self.lightning_model

    def fit(self, train_dataset=None, val_dataset=None, test_dataset=None, checkpoint_path=None, mode='continue',
            run_dir=None):
        output_dir = run_dir if run_dir else self.output_dir
        os.makedirs(output_dir, exist_ok=True)

        if train_dataset is None and val_dataset is None:
            raise ValueError("At least one of train_dataset or val_dataset must be provided")

        train_loader = None
        val_loader = None
        test_loader = None

        from torch_geometric.loader import DataLoader
        follow_batch = ['z0', 'z1', 'z2', 'pos0', 'pos1', 'pos2']

        if train_dataset is not None:
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.kwargs.get('num_workers', 4),
                follow_batch=follow_batch
            )

        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.kwargs.get('num_workers', 4),
                follow_batch=follow_batch
            )

        if test_dataset is not None:
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.kwargs.get('num_workers', 4),
                follow_batch=follow_batch
            )

        if checkpoint_path is not None:
            from ..module.pl_wrap import Estimator

            if mode.lower() == 'continue':
                print(f"Loading model from checkpoint for continued training: {checkpoint_path}")
                self.lightning_model = Estimator.load_from_checkpoint(checkpoint_path)
                self.lightning_model.batch_size = self.batch_size
                self.lightning_model.lr = self.learning_rate
                trainer_config = {'resume_from_checkpoint': checkpoint_path}

            elif mode.lower() == 'finetune':
                print(f"Loading model weights for fine-tuning: {checkpoint_path}")
                loaded_model = Estimator.load_from_checkpoint(checkpoint_path)

                self.create_model()

                self.lightning_model.net.load_state_dict(loaded_model.net.state_dict())
                self.lightning_model.readout_module.load_state_dict(loaded_model.readout_module.state_dict())

                if self.kwargs.get('freeze_base_model', False):
                    print("Freezing base model for fine-tuning")
                    for param in self.lightning_model.net.parameters():
                        param.requires_grad = False

                finetune_lr = self.kwargs.get('finetune_lr', self.learning_rate * 0.1)
                print(f"Fine-tuning with learning rate: {finetune_lr}")
                self.lightning_model.lr = finetune_lr

                trainer_config = {}
            else:
                raise ValueError(f"Invalid mode {mode}. Must be 'continue' or 'finetune'")
        else:
            self.create_model()
            trainer_config = {}

        callbacks = self.setup_callbacks(run_dir=output_dir)
        loggers = self.setup_loggers(run_dir=output_dir)

        trainer_config.update({
            'logger': loggers,
            'callbacks': callbacks,
            'max_epochs': self.max_epochs,
            'min_epochs': self.min_epochs,
            'log_every_n_steps': self.log_every_n_steps,
            'deterministic': True,
            'accelerator': 'gpu' if self.gpu and torch.cuda.is_available() else 'cpu',
            'devices': 1 if self.gpu and torch.cuda.is_available() else None,
            'num_sanity_val_steps': 2,
        })

        if self.gradient_clip_val > 0:
            trainer_config['gradient_clip_val'] = self.gradient_clip_val

        if self.precision in ['16', '32', 'bf16', 'mixed']:
            trainer_config['precision'] = self.precision

        self.trainer = pl.Trainer(**trainer_config)

        start_time = time.time()
        self.trainer.fit(self.lightning_model, train_loader, val_loader)
        training_time = time.time() - start_time

        train_metrics = {
            'best_model_path': self.trainer.checkpoint_callback.best_model_path if hasattr(self.trainer,
                                                                                           'checkpoint_callback') else None,
            'training_time': training_time,
            'epochs_completed': self.trainer.current_epoch,
            'mode': mode
        }

        metrics_path = os.path.join(output_dir, 'train_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(train_metrics, f, indent=2)

        if test_loader is not None:
            test_results = self.trainer.test(self.lightning_model, test_loader)
            test_metrics_path = os.path.join(output_dir, 'test_metrics.json')
            with open(test_metrics_path, 'w') as f:
                json.dump(test_results[0], f, indent=2)
            train_metrics['test_metrics'] = test_results[0]

        return train_metrics

    def train(self, train_loader, val_loader, test_loader=None, run_dir=None):
        output_dir = run_dir if run_dir else self.output_dir
        os.makedirs(output_dir, exist_ok=True)

        if self.lightning_model is None:
            self.create_model()

        callbacks = self.setup_callbacks(run_dir=output_dir)
        loggers = self.setup_loggers(run_dir=output_dir)

        trainer_config = {
            'logger': loggers,
            'callbacks': callbacks,
            'max_epochs': self.max_epochs,
            'min_epochs': self.min_epochs,
            'log_every_n_steps': self.log_every_n_steps,
            'deterministic': True,
            'accelerator': 'gpu' if self.gpu and torch.cuda.is_available() else 'cpu',
            'devices': 1 if self.gpu and torch.cuda.is_available() else None,
            'num_sanity_val_steps': 2,
        }

        if self.gradient_clip_val > 0:
            trainer_config['gradient_clip_val'] = self.gradient_clip_val

        if self.precision in ['16', '32', 'bf16', 'mixed']:
            trainer_config['precision'] = self.precision

        self.trainer = pl.Trainer(**trainer_config)

        start_time = time.time()
        self.trainer.fit(self.lightning_model, train_loader, val_loader)
        training_time = time.time() - start_time

        train_metrics = {
            'best_model_path': self.trainer.checkpoint_callback.best_model_path if hasattr(self.trainer,
                                                                                           'checkpoint_callback') else None,
            'training_time': training_time,
            'epochs_completed': self.trainer.current_epoch,
        }

        with open(os.path.join(output_dir, 'train_metrics.json'), 'w') as f:
            json.dump(train_metrics, f, indent=2)

        if test_loader is not None:
            test_results = self.evaluate(test_loader, run_dir=output_dir)
            train_metrics['test_metrics'] = test_results

        return train_metrics

    def evaluate(self, test_loader, run_dir=None):
        output_dir = run_dir if run_dir else self.output_dir

        if self.trainer is None:
            raise ValueError("Trainer has not been initialized. Call train() or fit() first.")

        test_results = self.trainer.test(self.lightning_model, test_loader)

        with open(os.path.join(output_dir, 'test_metrics.json'), 'w') as f:
            json.dump(test_results[0], f, indent=2)

        return test_results[0]

    def save_model(self, path=None, run_dir=None):
        output_dir = run_dir if run_dir else self.output_dir

        if self.lightning_model is None:
            raise ValueError("Model has not been initialized. Call train() or fit() first.")

        if path is None:
            path = os.path.join(output_dir, 'model.ckpt')

        self.trainer.save_checkpoint(path)

        return path