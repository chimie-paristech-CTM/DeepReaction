import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, Any, List, Optional, Tuple, Union
from collections import defaultdict
from torch_geometric.utils import to_dense_batch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.model import MoleculePredictionModel
from utils.metrics import compute_regression_metrics


class Estimator(pl.LightningModule):
    def __init__(
            self,
            model_type: str,
            readout: str,
            batch_size: int,
            lr: float,
            max_num_atoms_in_mol: int,
            scaler=None,
            use_layer_norm: bool = False,
            node_latent_dim: int = 128,
            edge_latent_dim: int = None,
            dropout: float = 0.0,
            model_kwargs: Optional[Dict[str, Any]] = None,
            readout_kwargs: Optional[Dict[str, Any]] = None,
            optimizer: str = 'adam',
            weight_decay: float = 0.0,
            scheduler: str = 'cosine',
            scheduler_patience: int = 10,
            scheduler_factor: float = 0.5,
            warmup_epochs: int = 0,
            min_lr: float = 1e-6,
            loss_function: str = 'mse',
            target_weights: List[float] = None,
            uncertainty_method: str = None,
            gradient_clip_val: float = 0.0,
            monitor_loss: str = 'val_total_loss',
            name: str = None,
            use_xtb_features: bool = True,
            num_xtb_features: int = 2,
            prediction_hidden_layers: int = 3,
            prediction_hidden_dim: int = 128,
            target_field_names: List[str] = None,
            **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        # Basic configuration
        self.model_type = model_type
        self.readout = readout
        self.batch_size = batch_size
        self.lr = lr
        self.max_num_atoms_in_mol = max_num_atoms_in_mol
        self.scaler = scaler
        self.use_layer_norm = use_layer_norm
        self.node_latent_dim = node_latent_dim
        self.edge_latent_dim = edge_latent_dim if edge_latent_dim is not None else node_latent_dim
        self.dropout = dropout
        self.monitor_loss = monitor_loss
        self.name = name
        self.num_targets = len(self.scaler) if isinstance(self.scaler, list) else 1
        self.use_xtb_features = use_xtb_features
        self.num_xtb_features = num_xtb_features
        self.prediction_hidden_layers = prediction_hidden_layers
        self.prediction_hidden_dim = prediction_hidden_dim

        # Optimization settings
        self.optimizer_type = optimizer
        self.weight_decay = weight_decay
        self.scheduler_type = scheduler
        self.scheduler_patience = scheduler_patience
        self.scheduler_factor = scheduler_factor
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr
        self.loss_function = loss_function
        self.target_weights = target_weights if target_weights is not None else [1.0] * self.num_targets
        self.uncertainty_method = uncertainty_method
        self.gradient_clip_val = gradient_clip_val
        self.model_kwargs = model_kwargs if model_kwargs is not None else {}
        self.readout_kwargs = readout_kwargs if readout_kwargs is not None else {}

        # Track outputs and metrics
        self.train_output = defaultdict(list)
        self.val_output = defaultdict(list)
        self.test_output = defaultdict(list)
        self.test_true = defaultdict(list)
        self.val_true = defaultdict(list)
        self.num_called_test = 1
        self.train_metrics = {}
        self.val_metrics = {}
        self.test_metrics = {}
        self.target_field_names = target_field_names

        # Initialize model
        self._init_model()

    def _init_model(self):
        self.model = MoleculePredictionModel(
            model_type=self.model_type,
            readout_type=self.readout,
            max_num_atoms=self.max_num_atoms_in_mol,
            node_dim=self.node_latent_dim,
            output_dim=self.num_targets,
            dropout=self.dropout,
            use_layer_norm=self.use_layer_norm,
            readout_kwargs=self.readout_kwargs,
            model_kwargs=self.model_kwargs,
            use_xtb_features=self.use_xtb_features,
            prediction_hidden_layers=self.prediction_hidden_layers,
            prediction_hidden_dim=self.prediction_hidden_dim,
            num_xtb_features=self.num_xtb_features
        )

        self.net = self.model.base_model
        self.readout_module = self.model.readout
        self.regr_or_cls_nn = self.model.prediction_mlp

    def forward(self, pos0, pos1, pos2, z0, z1, z2, batch_mapping, xtb_features=None):
        return self.model(pos0, pos1, pos2, z0, z1, z2, batch_mapping, xtb_features)

    def configure_optimizers(self):
        # Select optimizer
        if self.optimizer_type == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer_type == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer_type == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.weight_decay)
        elif self.optimizer_type == 'rmsprop':
            optimizer = torch.optim.RMSprop(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Return optimizer only if no scheduler
        if self.scheduler_type == 'none' or not self.scheduler_type:
            return {'optimizer': optimizer, 'monitor': self.monitor_loss}

        # Setup scheduler
        if self.scheduler_type == 'cosine':
            from torch.optim.lr_scheduler import CosineAnnealingLR
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=max(1, self.trainer.max_epochs if self.trainer else 100),
                eta_min=self.min_lr
            )
        elif self.scheduler_type == 'step':
            from torch.optim.lr_scheduler import StepLR
            scheduler = StepLR(optimizer, step_size=20, gamma=0.5)
        elif self.scheduler_type == 'plateau':
            from torch.optim.lr_scheduler import ReduceLROnPlateau
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=self.scheduler_factor,
                patience=self.scheduler_patience,
                min_lr=self.min_lr,
                verbose=True
            )
        elif self.scheduler_type == 'exponential':
            from torch.optim.lr_scheduler import ExponentialLR
            scheduler = ExponentialLR(optimizer, gamma=0.95)
        elif self.scheduler_type == 'warmup_cosine':
            from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

            # Create warmup scheduler
            warmup_scheduler = LinearLR(
                optimizer, start_factor=0.1, end_factor=1.0, total_iters=self.warmup_epochs
            )

            # Calculate remaining epochs after warmup
            remaining_epochs = max(1, (self.trainer.max_epochs - self.warmup_epochs) if self.trainer else 100)

            # Create main scheduler
            cosine_scheduler = CosineAnnealingLR(optimizer, T_max=remaining_epochs, eta_min=self.min_lr)

            # Combine schedulers if using warmup
            if self.warmup_epochs > 0:
                scheduler = SequentialLR(
                    optimizer,
                    schedulers=[warmup_scheduler, cosine_scheduler],
                    milestones=[self.warmup_epochs]
                )
            else:
                scheduler = cosine_scheduler
        else:
            # Default to no scheduler
            return {'optimizer': optimizer, 'monitor': self.monitor_loss}

        scheduler_config = {
            'scheduler': scheduler,
            'interval': 'epoch',
            'name': 'lr',
        }

        # Add monitoring for plateau scheduler
        if self.scheduler_type == 'plateau':
            scheduler_config['monitor'] = self.monitor_loss
            scheduler_config['frequency'] = 1

        return {'optimizer': optimizer, 'lr_scheduler': scheduler_config, 'monitor': self.monitor_loss}

    def _batch_loss(self, pos0, pos1, pos2, y, z0, z1, z2, batch_mapping, xtb_features=None):
        _, graph_embeddings, predictions = self.forward(
            pos0, pos1, pos2, z0, z1, z2, batch_mapping, xtb_features
        )

        # Calculate loss for each target with weighting
        total_loss = 0.0
        for i in range(self.num_targets):
            target_weight = self.target_weights[i]

            if self.loss_function == 'mse':
                loss = F.mse_loss(predictions[:, i], y[:, i])
            elif self.loss_function == 'mae' or self.loss_function == 'l1':
                loss = F.l1_loss(predictions[:, i], y[:, i])
            elif self.loss_function == 'huber':
                loss = F.huber_loss(predictions[:, i], y[:, i])
            elif self.loss_function == 'smooth_l1':
                loss = F.smooth_l1_loss(predictions[:, i], y[:, i])
            else:
                loss = F.l1_loss(predictions[:, i], y[:, i])

            total_loss += target_weight * loss

        return total_loss, graph_embeddings, predictions

    def _step(self, batch, step_type: str):
        # Extract data from batch
        pos0, pos1, pos2, y = batch.pos0, batch.pos1, batch.pos2, batch.y
        z0, z1, z2, batch_mapping = batch.z0, batch.z1, batch.z2, batch.batch
        xtb_features = getattr(batch, 'xtb_features', None)

        # Calculate loss and get predictions
        total_loss, _, predictions = self._batch_loss(
            pos0, pos1, pos2, y, z0, z1, z2, batch_mapping, xtb_features
        )

        # Store outputs for later evaluation
        output = (predictions.detach(), y.detach())
        if step_type == 'train':
            self.train_output[self.current_epoch].append(output)
        elif step_type == 'valid':
            self.val_output[self.current_epoch].append(output)
        elif step_type == 'test':
            self.test_output[self.num_called_test].append(output)

        return total_loss

    def training_step(self, batch, batch_idx):
        train_total_loss = self._step(batch, 'train')
        self.log('train_total_loss', train_total_loss, batch_size=self.batch_size, on_step=True, on_epoch=True,
                 prog_bar=True)
        return train_total_loss

    def validation_step(self, batch, batch_idx):
        val_total_loss = self._step(batch, 'valid')
        self.log('val_total_loss', val_total_loss, batch_size=self.batch_size, on_step=True, on_epoch=True,
                 prog_bar=True)
        return val_total_loss

    def test_step(self, batch, batch_idx):
        test_total_loss = self._step(batch, 'test')
        self.log('test_total_loss', test_total_loss, batch_size=self.batch_size)
        return test_total_loss

    def _epoch_end_report(self, epoch_outputs, epoch_type):
        # Concatenate all batch outputs
        preds = torch.cat([out[0] for out in epoch_outputs], dim=0)
        trues = torch.cat([out[1] for out in epoch_outputs], dim=0)

        y_pred_np = preds.cpu().numpy()
        y_true_np = trues.cpu().numpy()

        all_metrics = []
        target_metrics = {}

        # Calculate metrics for each target
        for i in range(self.num_targets):
            y_pred_target = y_pred_np[:, i].reshape(-1, 1)
            y_true_target = y_true_np[:, i].reshape(-1, 1)

            # Apply inverse scaling if available
            if isinstance(self.scaler, list) and len(self.scaler) > i:
                y_pred_target = self.scaler[i].inverse_transform(y_pred_target)
                y_true_target = self.scaler[i].inverse_transform(y_true_target)

            # Calculate regression metrics
            metrics_dict = compute_regression_metrics(y_true_target, y_pred_target, metrics=['mae', 'rmse', 'r2'])

            # Get target name
            target_name = self.target_field_names[i] if self.target_field_names and i < len(
                self.target_field_names) else f"Target {i}"

            # Store metrics
            target_metrics[f"target_{i}"] = metrics_dict
            all_metrics.append([metrics_dict['mae'], metrics_dict['rmse'], metrics_dict['r2']])

            # Log individual target metrics
            self.log(f'{epoch_type} MAE {target_name}', metrics_dict['mae'], batch_size=self.batch_size)
            self.log(f'{epoch_type} RMSE {target_name}', metrics_dict['rmse'], batch_size=self.batch_size)
            self.log(f'{epoch_type} R2 {target_name}', metrics_dict['r2'], batch_size=self.batch_size)

        # Calculate and log average metrics
        avg_mae = np.mean([m[0] for m in all_metrics])
        avg_rmse = np.mean([m[1] for m in all_metrics])
        avg_r2 = np.mean([m[2] for m in all_metrics])

        self.log(f'{epoch_type} Avg MAE', avg_mae, batch_size=self.batch_size)
        self.log(f'{epoch_type} Avg RMSE', avg_rmse, batch_size=self.batch_size)
        self.log(f'{epoch_type} Avg R2', avg_r2, batch_size=self.batch_size)

        return target_metrics, y_pred_np, y_true_np

    def on_train_epoch_end(self):
        if self.current_epoch in self.train_output and len(self.train_output[self.current_epoch]) > 0:
            train_metrics, _, _ = self._epoch_end_report(self.train_output[self.current_epoch], epoch_type='Train')
            self.train_metrics[self.current_epoch] = train_metrics
            del self.train_output[self.current_epoch]

    # def on_validation_epoch_end(self):
    #     if self.current_epoch in self.val_output and len(self.val_output[self.current_epoch]) > 0:
    #         val_metrics, y_pred, y_true = self._epoch_end_report(self.val_output[self.current_epoch], epoch_type='Validation')
    #         self.val_metrics[self.current_epoch] = val_metrics
    #         self.val_true[self.current_epoch] = y_true
    #         del self.val_output[self.current_epoch]

    def on_validation_epoch_end(self):
        if self.current_epoch in self.val_output and len(self.val_output[self.current_epoch]) > 0:
            val_metrics, y_pred, y_true = self._epoch_end_report(self.val_output[self.current_epoch],
                                                                 epoch_type='Validation')
            self.val_metrics[self.current_epoch] = val_metrics

            # Store metrics in a more accessible format for hyperopt
            if 'target_0' in val_metrics:
                for target_idx, target_data in val_metrics.items():
                    if isinstance(target_data, dict) and 'mae' in target_data:
                        # Store direct metrics for easier retrieval
                        self.val_metrics[f'val_mae'] = target_data.get('mae', float('inf'))
                        self.val_metrics[f'val_rmse'] = target_data.get('rmse', float('inf'))
                        self.val_metrics[f'val_r2'] = target_data.get('r2', 0.0)
                        break  # Just use the first target's metrics for now
            else:
                # If no target-specific metrics, use average metrics directly
                avg_mae = np.mean([m[0] for m in self.all_metrics]) if hasattr(self,
                                                                               'all_metrics') and self.all_metrics else float(
                    'inf')
                avg_rmse = np.mean([m[1] for m in self.all_metrics]) if hasattr(self,
                                                                                'all_metrics') and self.all_metrics else float(
                    'inf')
                avg_r2 = np.mean([m[2] for m in self.all_metrics]) if hasattr(self,
                                                                              'all_metrics') and self.all_metrics else 0.0

                self.val_metrics['val_mae'] = avg_mae
                self.val_metrics['val_rmse'] = avg_rmse
                self.val_metrics['val_r2'] = avg_r2

            self.val_true[self.current_epoch] = y_true
            del self.val_output[self.current_epoch]

    def on_test_epoch_end(self):
        if self.num_called_test in self.test_output and len(self.test_output[self.num_called_test]) > 0:
            metrics, y_pred, y_true = self._epoch_end_report(self.test_output[self.num_called_test], epoch_type='Test')
            self.test_output[self.num_called_test] = y_pred
            self.test_true[self.num_called_test] = y_true
            self.test_metrics[self.num_called_test] = metrics
            self.num_called_test += 1