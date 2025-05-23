import os
import time
import json
import torch
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple
from torch_geometric.loader import DataLoader as GeometricDataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint, EarlyStopping, LearningRateMonitor, TQDMProgressBar, Timer
)
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

from .config import Config
from ..module.pl_wrap import Estimator


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.float32):
            return float(obj)
        return json.JSONEncoder.default(self, obj)


class ReactionTrainer:
    def __init__(self, config: Config):
        self.config = config
        self.logger = self._setup_logging()
        self._setup_device()
        self._setup_precision()
        self._validate_config()

    def _setup_logging(self):
        logger = logging.getLogger('deepreaction')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(getattr(logging, self.config.system.log_level.upper()))
        return logger

    def _setup_device(self):
        if self.config.system.cuda and torch.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.config.system.gpu_id)
            self.device = torch.device(f"cuda:{self.config.system.gpu_id}")
            self.logger.info(f"Using GPU: {torch.cuda.get_device_name(self.device)}")
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            self.device = torch.device("cpu")
            self.logger.info("Using CPU")
            self.config.system.cuda = False

    def _setup_precision(self):
        if self.config.system.cuda and torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(self.device)
            if any(gpu in device_name for gpu in
                   ['V100', 'A100', 'A10', 'A30', 'A40', 'RTX 30', 'RTX 40', '3080', '3090', '4080', '4090']):
                torch.set_float32_matmul_precision(self.config.system.matmul_precision)
                self.logger.info(
                    f"Set float32 matmul precision to '{self.config.system.matmul_precision}' for better Tensor Core utilization")

    def _validate_config(self):
        self.logger.info("Validating configuration...")

        assert self.config.training.batch_size > 0, "batch_size must be positive"
        assert self.config.training.lr > 0, "learning rate must be positive"
        assert self.config.training.max_epochs > 0, "max_epochs must be positive"

        if len(self.config.dataset.target_fields) != len(self.config.dataset.input_features):
            self.logger.warning(
                f"Target fields ({len(self.config.dataset.target_fields)}) and input features ({len(self.config.dataset.input_features)}) count mismatch")

        if len(self.config.dataset.target_weights) != len(self.config.dataset.target_fields):
            self.logger.warning(
                f"Target weights length ({len(self.config.dataset.target_weights)}) doesn't match target fields ({len(self.config.dataset.target_fields)})")

        self.logger.info("Configuration validation completed")

    def _create_dataloader(self, dataset, batch_size, eval_mode=False):
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2 ** 32
            np.random.seed(worker_seed)
            torch.manual_seed(worker_seed)

        num_workers = self.config.system.num_workers if not eval_mode else min(self.config.system.num_workers, 2)

        loader_kwargs = {
            'batch_size': batch_size,
            'num_workers': num_workers,
            'worker_init_fn': None if eval_mode else seed_worker,
            'shuffle': not eval_mode,
            'follow_batch': ['z0', 'z1', 'z2', 'pos0', 'pos1', 'pos2'],
            'pin_memory': self.config.system.cuda,
        }

        self.logger.info(
            f"Creating dataloader: batch_size={batch_size}, num_workers={num_workers}, eval_mode={eval_mode}")
        return GeometricDataLoader(dataset, **loader_kwargs)

    def _create_model(self, scalers):
        scalers_to_use = scalers if self.config.dataset.use_scaler else None
        num_targets = len(self.config.dataset.target_fields)
        num_features = len(self.config.dataset.input_features)

        self.logger.info(f"Creating model with {num_targets} targets and {num_features} input features")

        model_kwargs = {
            'hidden_channels': self.config.model.hidden_channels,
            'num_blocks': self.config.model.num_blocks,
            'int_emb_size': self.config.model.int_emb_size,
            'basis_emb_size': self.config.model.basis_emb_size,
            'out_emb_channels': self.config.model.out_emb_channels,
            'num_spherical': self.config.model.num_spherical,
            'num_radial': self.config.model.num_radial,
            'cutoff': self.config.model.cutoff,
            'max_num_neighbors': self.config.model.max_num_neighbors,
            'envelope_exponent': self.config.model.envelope_exponent,
            'num_before_skip': self.config.model.num_before_skip,
            'num_after_skip': self.config.model.num_after_skip,
            'num_output_layers': self.config.model.num_output_layers,
        }

        readout_kwargs = {
            'readout_hidden_dim': self.config.model.readout_hidden_dim,
            'readout_num_heads': self.config.model.readout_num_heads,
            'readout_num_sabs': self.config.model.readout_num_sabs,
            'readout_layer_norm': self.config.model.use_layer_norm
        }

        model_config = {
            'model_type': self.config.model.model_type,
            'readout': self.config.dataset.readout,
            'batch_size': self.config.training.batch_size,
            'lr': self.config.training.lr,
            'max_num_atoms_in_mol': self.config.model.max_num_atoms,
            'scaler': scalers_to_use,
            'use_layer_norm': self.config.model.use_layer_norm,
            'node_latent_dim': self.config.model.node_dim,
            'edge_latent_dim': self.config.model.node_dim,
            'dropout': self.config.model.dropout,
            'model_kwargs': model_kwargs,
            'readout_kwargs': readout_kwargs,
            'optimizer': self.config.training.optimizer,
            'weight_decay': self.config.training.weight_decay,
            'scheduler': self.config.training.scheduler,
            'scheduler_patience': self.config.training.early_stopping_patience // 4,
            'scheduler_factor': 0.5,
            'warmup_epochs': self.config.training.warmup_epochs,
            'min_lr': self.config.training.min_lr,
            'loss_function': self.config.training.loss_function,
            'target_weights': self.config.dataset.target_weights,
            'uncertainty_method': None,
            'gradient_clip_val': self.config.training.gradient_clip_val,
            'use_xtb_features': self.config.model.use_xtb_features,
            'num_xtb_features': num_features,
            'prediction_hidden_layers': self.config.model.prediction_hidden_layers,
            'prediction_hidden_dim': self.config.model.prediction_hidden_dim,
            'target_field_names': self.config.dataset.target_fields
        }

        self.logger.info("Model configuration:")
        for key, value in model_config.items():
            if key not in ['model_kwargs', 'readout_kwargs', 'scaler']:
                self.logger.info(f"  {key}: {value}")

        if scalers_to_use:
            self.logger.info(f"Using {len(scalers_to_use)} scalers")

        return Estimator(**model_config)

    def _prepare_output_dir(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)

        checkpoint_dir = os.path.join(output_dir, 'checkpoints')
        if os.path.exists(checkpoint_dir) and os.listdir(checkpoint_dir):
            self.logger.warning(f"Checkpoint directory {checkpoint_dir} exists and is not empty")
            existing_files = os.listdir(checkpoint_dir)
            self.logger.info(f"Found {len(existing_files)} existing checkpoint files")
            backup_dir = os.path.join(output_dir, f'checkpoints_backup_{int(time.time())}')
            os.makedirs(backup_dir, exist_ok=True)
            for file in existing_files:
                if file.endswith('.ckpt'):
                    os.rename(
                        os.path.join(checkpoint_dir, file),
                        os.path.join(backup_dir, file)
                    )
            self.logger.info(f"Moved existing checkpoints to {backup_dir}")
        else:
            os.makedirs(checkpoint_dir, exist_ok=True)

    def _setup_callbacks(self, output_dir):
        callbacks = []

        if self.config.training.save_best_model:
            callbacks.append(ModelCheckpoint(
                monitor='val_total_loss',
                dirpath=os.path.join(output_dir, 'checkpoints'),
                filename='best-{epoch:04d}-{val_total_loss:.4f}',
                save_top_k=1,
                mode='min',
                save_weights_only=False,
                verbose=True
            ))
            self.logger.info("Added best model checkpoint callback")

        if self.config.training.save_last_model:
            callbacks.append(ModelCheckpoint(
                dirpath=os.path.join(output_dir, 'checkpoints'),
                filename='last-{epoch:04d}',
                save_top_k=1,
                save_last=True,
                verbose=False
            ))
            self.logger.info("Added last model checkpoint callback")

        if self.config.training.save_interval > 0:
            callbacks.append(ModelCheckpoint(
                dirpath=os.path.join(output_dir, 'checkpoints'),
                filename='epoch-{epoch:04d}',
                save_top_k=-1,
                every_n_epochs=self.config.training.save_interval,
                verbose=False
            ))
            self.logger.info(f"Added interval checkpoint callback (every {self.config.training.save_interval} epochs)")

        if self.config.training.early_stopping_patience > 0:
            callbacks.append(EarlyStopping(
                monitor='val_total_loss',
                patience=self.config.training.early_stopping_patience,
                min_delta=self.config.training.early_stopping_min_delta,
                mode='min',
                verbose=True
            ))
            self.logger.info(
                f"Added early stopping callback (patience: {self.config.training.early_stopping_patience})")

        callbacks.extend([
            LearningRateMonitor(logging_interval='epoch'),
            Timer(duration=None),
            TQDMProgressBar(refresh_rate=10)
        ])

        return callbacks

    def _setup_loggers(self, output_dir):
        loggers = [
            TensorBoardLogger(
                save_dir=output_dir,
                name='tensorboard',
                default_hp_metric=False
            ),
            CSVLogger(
                save_dir=output_dir,
                name='csv_logs',
                flush_logs_every_n_steps=50
            )
        ]
        self.logger.info(f"Setup loggers in {output_dir}")
        return loggers

    def fit(self, train_dataset, val_dataset, test_dataset=None, scalers=None, checkpoint_path=None, mode='train') -> \
    Dict[str, Any]:
        pl.seed_everything(self.config.training.random_seed)
        self.logger.info(f"Set random seed to {self.config.training.random_seed}")

        output_dir = self.config.training.out_dir
        self._prepare_output_dir(output_dir)

        self.config.save(os.path.join(output_dir, 'config.json'))
        self.logger.info(f"Configuration saved to {output_dir}/config.json")

        if self.config.system.log_level.lower() == 'debug':
            self.config.print_config()

        train_loader = self._create_dataloader(
            train_dataset,
            self.config.training.batch_size,
            eval_mode=False
        )
        val_loader = self._create_dataloader(
            val_dataset,
            self.config.training.eval_batch_size,
            eval_mode=True
        )

        self.logger.info(f"Created dataloaders: train={len(train_loader)}, val={len(val_loader)} batches")

        model = self._create_model(scalers)

        checkpoint_to_use = checkpoint_path or self.config.training.checkpoint_path
        if checkpoint_to_use and os.path.exists(checkpoint_to_use):
            self.logger.info(f"Loading from checkpoint: {checkpoint_to_use}")
            try:
                model = Estimator.load_from_checkpoint(checkpoint_to_use)
            except Exception as e:
                self.logger.error(f"Failed to load checkpoint: {e}")
                raise
        elif checkpoint_to_use:
            self.logger.warning(f"Checkpoint path specified but file not found: {checkpoint_to_use}")

        callbacks = self._setup_callbacks(output_dir)
        loggers = self._setup_loggers(output_dir)

        trainer_config = {
            'logger': loggers,
            'callbacks': callbacks,
            'max_epochs': self.config.training.max_epochs,
            'min_epochs': self.config.training.min_epochs,
            'log_every_n_steps': min(10, len(train_loader)),
            'deterministic': True,
            'accelerator': 'gpu' if self.config.system.cuda else 'cpu',
            'num_sanity_val_steps': 2,
            'gradient_clip_val': self.config.training.gradient_clip_val if self.config.training.gradient_clip_val > 0 else None,
            'accumulate_grad_batches': self.config.training.gradient_accumulation_steps,
            'precision': self.config.training.precision,
            'strategy': self.config.system.strategy,
            'num_nodes': self.config.system.num_nodes,
            'devices': self.config.system.devices if self.config.system.devices > 0 else "auto"
        }

        self.logger.info("Trainer configuration:")
        for key, value in trainer_config.items():
            if key not in ['logger', 'callbacks']:
                self.logger.info(f"  {key}: {value}")

        trainer = pl.Trainer(**trainer_config)

        start_time = time.time()
        self.logger.info(f"Starting training for {self.config.training.max_epochs} epochs")
        self.logger.info(f"Training dataset size: {len(train_dataset)}")
        self.logger.info(f"Validation dataset size: {len(val_dataset)}")

        try:
            trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise

        training_time = time.time() - start_time

        metrics = {
            'training_time': training_time,
            'epochs_completed': trainer.current_epoch,
            'final_epoch': trainer.current_epoch,
            'max_epochs': self.config.training.max_epochs,
            'config': self.config.to_dict()
        }

        if hasattr(trainer, 'checkpoint_callback') and trainer.checkpoint_callback:
            if hasattr(trainer.checkpoint_callback, 'best_model_path'):
                metrics['best_model_path'] = trainer.checkpoint_callback.best_model_path
                self.logger.info(f"Best model saved to: {trainer.checkpoint_callback.best_model_path}")

        if test_dataset is not None:
            test_loader = self._create_dataloader(
                test_dataset,
                self.config.training.eval_batch_size,
                eval_mode=True
            )
            self.logger.info("Running test evaluation")
            try:
                test_results = trainer.test(model, dataloaders=test_loader)
                metrics['test_results'] = test_results[0] if test_results else {}
                self.logger.info(f"Test results: {metrics['test_results']}")
            except Exception as e:
                self.logger.error(f"Test evaluation failed: {e}")
                metrics['test_results'] = {}

        metrics_path = os.path.join(output_dir, 'metrics.json')
        try:
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2, cls=NumpyEncoder)
        except Exception as e:
            self.logger.error(f"Failed to save metrics: {e}")

        self.logger.info(f"Training completed in {training_time:.2f} seconds")
        self.logger.info(f"Training metrics saved to {metrics_path}")

        return metrics