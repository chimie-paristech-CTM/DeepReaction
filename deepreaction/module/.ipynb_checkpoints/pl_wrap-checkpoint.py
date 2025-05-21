import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, Any, List, Optional, Tuple, Union
from collections import defaultdict
from torch_geometric.utils import to_dense_batch

from ..model.factory import ModelFactory
from ..utils.metrics import compute_regression_metrics


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
            use_xtb_features: bool = False,
            num_xtb_features: int = 0,
            prediction_hidden_layers: int = 3,
            prediction_hidden_dim: int = 128,
            target_field_names: List[str] = None,
            num_targets: int = None,
            **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

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
        self.use_xtb_features = use_xtb_features
        self.num_xtb_features = num_xtb_features
        self.prediction_hidden_layers = prediction_hidden_layers
        self.prediction_hidden_dim = prediction_hidden_dim
        self.target_field_names = target_field_names or []
        self.is_training = False

        if num_targets is not None:
            self.num_targets = num_targets
            print(f"Using explicitly provided num_targets: {self.num_targets}")
        elif target_field_names and len(target_field_names) > 0:
            self.num_targets = len(target_field_names)
            print(f"Using num_targets from target_field_names: {self.num_targets}")
        else:
            self.num_targets = len(self.scaler) if isinstance(self.scaler, list) else 1
            print(f"Using num_targets from scaler: {self.num_targets}")

        self.optimizer_type = optimizer
        self.weight_decay = weight_decay
        self.scheduler_type = scheduler
        self.scheduler_patience = scheduler_patience
        self.scheduler_factor = scheduler_factor
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr
        self.loss_function = loss_function
        
        if target_weights is None or len(target_weights) != self.num_targets:
            self.target_weights = [1.0] * self.num_targets
            print(f"Setting default target weights: {self.target_weights}")
        else:
            self.target_weights = target_weights
            print(f"Using provided target weights: {self.target_weights}")
            
        self.uncertainty_method = uncertainty_method
        self.gradient_clip_val = gradient_clip_val
        self.model_kwargs = model_kwargs if model_kwargs is not None else {}
        self.readout_kwargs = readout_kwargs if readout_kwargs is not None else {}

        self.train_output = defaultdict(list)
        self.val_output = defaultdict(list)
        self.test_output = defaultdict(list)
        self.test_true = defaultdict(list)
        self.val_true = defaultdict(list)
        self.num_called_test = 1
        self.train_metrics = {}
        self.val_metrics = {}
        self.test_metrics = {}

        if not self.target_field_names or len(self.target_field_names) != self.num_targets:
            print(f"Warning: target_field_names length ({len(self.target_field_names) if self.target_field_names else 0}) "
                  f"doesn't match num_targets ({self.num_targets})")
            self.target_field_names = [f"target_{i}" for i in range(self.num_targets)]
            print(f"Using default target field names: {self.target_field_names}")
        else:
            print(f"Using target field names: {self.target_field_names}")

        print(f"Final model configuration: num_targets={self.num_targets}, "
              f"target_field_names={self.target_field_names}, "
              f"target_weights={self.target_weights}")

        self._init_model()

    def _init_model(self):
        from ..model.model import MoleculePredictionModel

        print(f"Initializing model with output_dim={self.num_targets}")
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
        is_training = hasattr(self, 'trainer') and self.trainer is not None
                   
        return self.model(pos0, pos1, pos2, z0, z1, z2, batch_mapping, xtb_features)

    def configure_optimizers(self):
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

        if self.scheduler_type == 'none' or not self.scheduler_type:
            return {'optimizer': optimizer, 'monitor': self.monitor_loss}

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

            warmup_scheduler = LinearLR(
                optimizer, start_factor=0.1, end_factor=1.0, total_iters=self.warmup_epochs
            )

            remaining_epochs = max(1, (self.trainer.max_epochs - self.warmup_epochs) if self.trainer else 100)

            cosine_scheduler = CosineAnnealingLR(optimizer, T_max=remaining_epochs, eta_min=self.min_lr)

            if self.warmup_epochs > 0:
                scheduler = SequentialLR(
                    optimizer,
                    schedulers=[warmup_scheduler, cosine_scheduler],
                    milestones=[self.warmup_epochs]
                )
            else:
                scheduler = cosine_scheduler
        else:
            return {'optimizer': optimizer, 'monitor': self.monitor_loss}

        scheduler_config = {
            'scheduler': scheduler,
            'interval': 'epoch',
            'name': 'lr',
        }

        if self.scheduler_type == 'plateau':
            scheduler_config['monitor'] = self.monitor_loss
            scheduler_config['frequency'] = 1

        return {'optimizer': optimizer, 'lr_scheduler': scheduler_config, 'monitor': self.monitor_loss}

    def _batch_loss(self, pos0, pos1, pos2, y, z0, z1, z2, batch_mapping, xtb_features=None):
        is_training = hasattr(self, 'trainer') and self.trainer is not None
        
        if not hasattr(self, 'batch_loss_debug') and is_training and self.trainer.current_epoch == 0:
            self.batch_loss_debug = True
            
        _, graph_embeddings, predictions = self.forward(
            pos0, pos1, pos2, z0, z1, z2, batch_mapping, xtb_features
        )

        if not hasattr(self, 'predictions_debug') and is_training and self.trainer.current_epoch == 0:
            if predictions.shape[1] != self.num_targets:
                print(f"WARNING: predictions shape {predictions.shape[1]} doesn't match num_targets {self.num_targets}")
            self.predictions_debug = True

        total_loss = 0.0
        individual_losses = []
        
        effective_targets = min(predictions.shape[1], y.shape[1], self.num_targets)
        if effective_targets != self.num_targets and is_training and not hasattr(self, 'targets_warning_printed'):
            print(f"WARNING: Using {effective_targets} targets instead of configured {self.num_targets}")
            self.targets_warning_printed = True
            
        for i in range(effective_targets):
            target_weight = self.target_weights[i] if i < len(self.target_weights) else 1.0

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

            individual_losses.append(loss.item())
            total_loss += target_weight * loss

        if not hasattr(self, 'loss_debug') and is_training and self.trainer.current_epoch == 0:
            self.loss_debug = True

        return total_loss, graph_embeddings, predictions

    def _step(self, batch, step_type: str):
        pos0, pos1, pos2, y = batch.pos0, batch.pos1, batch.pos2, batch.y
        z0, z1, z2, batch_mapping = batch.z0, batch.z1, batch.z2, batch.batch
        xtb_features = getattr(batch, 'xtb_features', None)

        is_training = hasattr(self, 'trainer') and self.trainer is not None
        
        if not hasattr(self, f'{step_type}_debug') and is_training and self.trainer.current_epoch == 0:
            if y.shape[1] != self.num_targets:
                print(f"WARNING: y shape {y.shape} doesn't match num_targets {self.num_targets}")
            setattr(self, f'{step_type}_debug', True)

        total_loss, _, predictions = self._batch_loss(
            pos0, pos1, pos2, y, z0, z1, z2, batch_mapping, xtb_features
        )

        if not hasattr(self, f'{step_type}_pred_debug') and is_training and self.trainer.current_epoch == 0:
            setattr(self, f'{step_type}_pred_debug', True)

        output = (predictions.detach(), y.detach())
        if step_type == 'train':
            self.train_output[self.current_epoch].append(output)
        elif step_type == 'valid':
            self.val_output[self.current_epoch].append(output)
        elif step_type == 'test':
            self.test_output[self.num_called_test].append(output)
            self.test_true[self.num_called_test].append(y.detach())

        return total_loss

    def training_step(self, batch, batch_idx):
        self.is_training = True
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
        is_training = hasattr(self, 'trainer') and self.trainer is not None
        
        preds = torch.cat([out[0] for out in epoch_outputs], dim=0)
        trues = torch.cat([out[1] for out in epoch_outputs], dim=0)

        y_pred_np = preds.cpu().numpy()
        y_true_np = trues.cpu().numpy()

        effective_targets = min(y_pred_np.shape[1], y_true_np.shape[1], self.num_targets)
        if effective_targets != self.num_targets and not hasattr(self, 'metrics_warning_printed'):
            print(f"WARNING: Using {effective_targets} targets for metrics instead of configured {self.num_targets}")
            self.metrics_warning_printed = True

        all_metrics = []
        target_metrics = {}

        metrics_to_compute = ['mae', 'rmse', 'r2', 'mpae', 'max_ae', 'median_ae']

        for i in range(effective_targets):
            y_pred_target = y_pred_np[:, i].reshape(-1, 1)
            y_true_target = y_true_np[:, i].reshape(-1, 1)

            if self.scaler is not None and isinstance(self.scaler, list) and len(self.scaler) > i:
                y_pred_target = self.scaler[i].inverse_transform(y_pred_target)
                y_true_target = self.scaler[i].inverse_transform(y_true_target)

            metrics_dict = compute_regression_metrics(y_true_target, y_pred_target, metrics=metrics_to_compute)

            target_name = self.target_field_names[i] if self.target_field_names and i < len(
                self.target_field_names) else f"Target {i}"

            target_metrics[f"target_{i}"] = metrics_dict

            metric_values = [metrics_dict.get(metric, 0) for metric in metrics_to_compute]
            all_metrics.append(metric_values)

            self.log(f'{epoch_type} MAE {target_name}', metrics_dict['mae'], batch_size=self.batch_size)
            self.log(f'{epoch_type} RMSE {target_name}', metrics_dict['rmse'], batch_size=self.batch_size)
            self.log(f'{epoch_type} R2 {target_name}', metrics_dict['r2'], batch_size=self.batch_size)

            if 'mpae' in metrics_dict and not np.isnan(metrics_dict['mpae']):
                self.log(f'{epoch_type} MPAE {target_name}', metrics_dict['mpae'], batch_size=self.batch_size)

            if 'max_ae' in metrics_dict:
                self.log(f'{epoch_type} MAX_AE {target_name}', metrics_dict['max_ae'], batch_size=self.batch_size)

            if 'median_ae' in metrics_dict:
                self.log(f'{epoch_type} MEDIAN_AE {target_name}', metrics_dict['median_ae'], batch_size=self.batch_size)

        avg_mae = np.mean([m[0] for m in all_metrics])
        avg_rmse = np.mean([m[1] for m in all_metrics])
        avg_r2 = np.mean([m[2] for m in all_metrics])

        avg_mpae = np.mean([m[3] for m in all_metrics if not np.isnan(m[3])])
        avg_max_ae = np.mean([m[4] for m in all_metrics])
        avg_median_ae = np.mean([m[5] for m in all_metrics])

        self.log(f'{epoch_type} Avg MAE', avg_mae, batch_size=self.batch_size)
        self.log(f'{epoch_type} Avg RMSE', avg_rmse, batch_size=self.batch_size)
        self.log(f'{epoch_type} Avg R2', avg_r2, batch_size=self.batch_size)

        if not np.isnan(avg_mpae):
            self.log(f'{epoch_type} Avg MPAE', avg_mpae, batch_size=self.batch_size)
        self.log(f'{epoch_type} Avg MAX_AE', avg_max_ae, batch_size=self.batch_size)
        self.log(f'{epoch_type} Avg MEDIAN_AE', avg_median_ae, batch_size=self.batch_size)

        return target_metrics, y_pred_np, y_true_np

    def on_train_epoch_end(self):
        if self.current_epoch in self.train_output and len(self.train_output[self.current_epoch]) > 0:
            metrics, _, _ = self._epoch_end_report(self.train_output[self.current_epoch], epoch_type='Train')
            self.train_metrics[self.current_epoch] = metrics

    def on_validation_epoch_end(self):
        if self.current_epoch in self.val_output and len(self.val_output[self.current_epoch]) > 0:
            metrics, _, _ = self._epoch_end_report(self.val_output[self.current_epoch], epoch_type='Val')
            self.val_metrics[self.current_epoch] = metrics

    def on_test_epoch_end(self):
        if self.num_called_test in self.test_output and len(self.test_output[self.num_called_test]) > 0:
            metrics, y_pred, y_true = self._epoch_end_report(self.test_output[self.num_called_test], epoch_type='Test')
            self.test_output[self.num_called_test] = y_pred
            self.test_true[self.num_called_test] = y_true
            self.test_metrics[self.num_called_test] = metrics
            self.num_called_test += 1