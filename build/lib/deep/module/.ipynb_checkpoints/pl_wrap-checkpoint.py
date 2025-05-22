"""
PyTorch Lightning module wrapper for molecular property prediction models.

This module contains the PyTorch Lightning wrapper for the DimeNet++ model and other components,
providing a standardized interface for training, validation, and testing.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW, SGD, RMSprop
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, 
    StepLR, 
    ReduceLROnPlateau, 
    ExponentialLR,
    CyclicLR,
    OneCycleLR
)
import pytorch_lightning as pl
from typing import Dict, Any, List, Optional, Tuple, Union
import logging
from collections import defaultdict
from torch_geometric.utils import to_dense_batch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.model import MoleculeModel, MoleculePredictionModel
from model.readout import ReadoutFactory
from model.mlp import PredictionMLP
from model.activations import Swish, swish

# Import utilities
from utils.metrics import compute_regression_metrics


class Estimator(pl.LightningModule):
    """
    PyTorch Lightning module for molecular property estimation using DimeNet++.
    """
    
    def __init__(
        self,
        readout: str,
        batch_size: int,
        lr: float,
        max_num_atoms_in_mol: int,
        scaler=None,
        use_layer_norm: bool = False,
        node_latent_dim: int = 128,
        edge_latent_dim: int = None,
        dropout: float = 0.0,
        
        # DimeNet++ parameters
        dimenet_hidden_channels: int = 128,
        dimenet_num_blocks: int = 4,
        dimenet_int_emb_size: int = 64,
        dimenet_basis_emb_size: int = 8,
        dimenet_out_emb_channels: int = 256,
        dimenet_num_spherical: int = 7,
        dimenet_num_radial: int = 6,
        dimenet_cutoff: float = 5.0,
        dimenet_envelope_exponent: int = 5,
        
        # Set Transformer parameters
        set_transformer_hidden_dim: int = 1024,
        set_transformer_num_heads: int = 16,
        set_transformer_num_sabs: int = 2,
        
        # Attention parameters
        attention_hidden_dim: int = 128,
        attention_num_heads: int = 4,
        
        # Optimization parameters
        optimizer: str = 'adam',
        weight_decay: float = 0.0,
        scheduler: str = 'cosine',
        scheduler_patience: int = 10,
        scheduler_factor: float = 0.5,
        warmup_epochs: int = 0,
        min_lr: float = 1e-6,
        loss_function: str = 'mse',
        uncertainty_method: str = None,
        gradient_clip_val: float = 0.0,
        monitor_loss: str = 'val_total_loss',
        name: str = None,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Store parameters
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
        self.linear_output_size = 1
        
        # Optimizer parameters
        self.optimizer_type = optimizer
        self.weight_decay = weight_decay
        self.scheduler_type = scheduler
        self.scheduler_patience = scheduler_patience
        self.scheduler_factor = scheduler_factor
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr
        self.loss_function = loss_function
        self.uncertainty_method = uncertainty_method
        self.gradient_clip_val = gradient_clip_val
        
        # DimeNet++ parameters
        self.dimenet_hidden_channels = dimenet_hidden_channels
        self.dimenet_num_blocks = dimenet_num_blocks
        self.dimenet_int_emb_size = dimenet_int_emb_size
        self.dimenet_basis_emb_size = dimenet_basis_emb_size
        self.dimenet_out_emb_channels = dimenet_out_emb_channels
        self.dimenet_num_spherical = dimenet_num_spherical
        self.dimenet_num_radial = dimenet_num_radial
        self.dimenet_cutoff = dimenet_cutoff
        self.dimenet_envelope_exponent = dimenet_envelope_exponent
        
        # Set Transformer parameters
        self.set_transformer_hidden_dim = set_transformer_hidden_dim
        self.set_transformer_num_heads = set_transformer_num_heads
        self.set_transformer_num_sabs = set_transformer_num_sabs
        
        # Attention parameters
        self.attention_hidden_dim = attention_hidden_dim
        self.attention_num_heads = attention_num_heads
        
        # Storage for outputs
        self.train_output = defaultdict(list)
        self.val_output = defaultdict(list)
        self.test_output = defaultdict(list)
        
        self.test_true = defaultdict(list)
        self.val_true = defaultdict(list)
        
        self.num_called_test = 1
        
        self.train_metrics = {}
        self.val_metrics = {}
        self.test_metrics = {}
        
        self.test_graph_embeddings = defaultdict(list)
        self.val_graph_embeddings = defaultdict(list)
        self.train_graph_embeddings = defaultdict(list)
        
        # Initialize model components
        self._init_model()
    
    def _init_model(self) -> None:
        """
        Initialize the model using refactored components.
        """
        # Prepare base model kwargs
        base_model_kwargs = {
            'hidden_channels': self.dimenet_hidden_channels,
            'out_channels': self.node_latent_dim,
            'num_blocks': self.dimenet_num_blocks,
            'int_emb_size': self.dimenet_int_emb_size,
            'basis_emb_size': self.dimenet_basis_emb_size,
            'out_emb_channels': self.dimenet_out_emb_channels,
            'num_spherical': self.dimenet_num_spherical,
            'num_radial': self.dimenet_num_radial,
            'cutoff': self.dimenet_cutoff,
            'max_num_neighbors': 32,
            'envelope_exponent': self.dimenet_envelope_exponent,
            'num_before_skip': 1,
            'num_after_skip': 2,
            'num_output_layers': 3,
        }
        
        # Prepare readout kwargs
        readout_kwargs = {
            'node_dim': self.node_latent_dim,
            'hidden_dim': self.attention_hidden_dim if self.readout == 'attention' else self.set_transformer_hidden_dim,
            'num_heads': self.attention_num_heads if self.readout == 'attention' else self.set_transformer_num_heads,
            'num_sabs': self.set_transformer_num_sabs,
            'layer_norm': self.use_layer_norm
        }
        
        # Initialize model using the MoleculePredictionModel
        self.model = MoleculePredictionModel(
            base_model_name='dimenet++',
            readout_type=self.readout,
            max_num_atoms=self.max_num_atoms_in_mol,
            node_dim=self.node_latent_dim,
            output_dim=self.linear_output_size,
            dropout=self.dropout,
            use_layer_norm=self.use_layer_norm,
            readout_kwargs=readout_kwargs,
            base_model_kwargs=base_model_kwargs
        )
        
        # For backward compatibility
        self.net = self.model.base_model
        self.readout_module = self.model.readout
        self.regr_or_cls_nn = self.model.prediction_mlp
    
    def forward(self, pos0, pos1, pos2, atom_z, batch_mapping):
        """
        Forward pass through the model.
        
        Args:
            pos0: First position tensor
            pos1: Second position tensor
            pos2: Third position tensor
            atom_z: Atomic numbers
            batch_mapping: Batch mapping indices
            
        Returns:
            Tuple: (node_embeddings, graph_embeddings, predictions)
        """
        return self.model(pos0, pos1, pos2, atom_z, batch_mapping)
    
    def configure_optimizers(self):
        """
        Configure optimizers and learning rate schedulers.
        
        Returns:
            Dict: Optimizer and scheduler configuration
        """
        # Set up optimizer based on type
        if self.optimizer_type == 'adam':
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_type == 'adamw':
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_type == 'sgd':
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.lr,
                momentum=0.9,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_type == 'rmsprop':
            optimizer = torch.optim.RMSprop(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay
            )
        else:
            # Default to Adam
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay
            )
        
        # Set up scheduler if requested
        if self.scheduler_type == 'none' or not self.scheduler_type:
            return {
                'optimizer': optimizer,
                'monitor': self.monitor_loss
            }
        
        # Set up scheduler based on type
        if self.scheduler_type == 'cosine':
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_epochs if self.trainer else 100,
                eta_min=self.min_lr
            )
            scheduler_config = {
                'scheduler': scheduler,
                'interval': 'epoch',
                'name': 'lr'
            }
        elif self.scheduler_type == 'step':
            scheduler = StepLR(
                optimizer,
                step_size=20,
                gamma=0.5
            )
            scheduler_config = {
                'scheduler': scheduler,
                'interval': 'epoch',
                'name': 'lr'
            }
        elif self.scheduler_type == 'plateau':
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=self.scheduler_factor,
                patience=self.scheduler_patience,
                min_lr=self.min_lr,
                verbose=True
            )
            scheduler_config = {
                'scheduler': scheduler,
                'interval': 'epoch',
                'monitor': self.monitor_loss,
                'name': 'lr'
            }
        elif self.scheduler_type == 'exponential':
            scheduler = ExponentialLR(
                optimizer,
                gamma=0.95
            )
            scheduler_config = {
                'scheduler': scheduler,
                'interval': 'epoch',
                'name': 'lr'
            }
        else:
            # Default to no scheduler
            return {
                'optimizer': optimizer,
                'monitor': self.monitor_loss
            }
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler_config,
            'monitor': self.monitor_loss
        }
    
    def _batch_loss(self, pos0, pos1, pos2, y, atom_z, batch_mapping):
        """
        Compute loss for a batch.
        
        Args:
            pos0: First position tensor
            pos1: Second position tensor
            pos2: Third position tensor
            y: Target values
            atom_z: Atomic numbers
            batch_mapping: Batch mapping indices
            
        Returns:
            Tuple: (loss, graph_embeddings, predictions)
        """
        _, graph_embeddings, predictions = self.forward(pos0, pos1, pos2, atom_z, batch_mapping)
        
        # Select loss function based on setting
        if self.loss_function == 'mse':
            loss = F.mse_loss(torch.flatten(predictions), torch.flatten(y).float())
        elif self.loss_function == 'mae' or self.loss_function == 'l1':
            loss = F.l1_loss(torch.flatten(predictions), torch.flatten(y).float())
        elif self.loss_function == 'huber':
            loss = F.huber_loss(torch.flatten(predictions), torch.flatten(y).float())
        elif self.loss_function == 'smooth_l1':
            loss = F.smooth_l1_loss(torch.flatten(predictions), torch.flatten(y).float())
        else:
            # Default to L1 loss as in the original code
            loss = F.l1_loss(torch.flatten(predictions), torch.flatten(y).float())
        
        return loss, graph_embeddings, predictions
    
    def _step(self, batch, step_type: str):
        """
        Common step function for training, validation, and testing.
        
        Args:
            batch: Batch data
            step_type: Step type ('train', 'valid', or 'test')
            
        Returns:
            torch.Tensor: Loss value
        """
        # Extract data from batch
        pos0, pos1, pos2, y, atom_z, batch_mapping = batch.pos0, batch.pos1, batch.pos2, batch.y, batch.z, batch.batch
        
        # Compute loss and predictions
        total_loss, _, predictions = self._batch_loss(pos0, pos1, pos2, y, atom_z, batch_mapping)
        
        # Store outputs
        output = (torch.flatten(predictions.detach()), torch.flatten(y.detach()))
        if step_type == 'train':
            self.train_output[self.current_epoch].append(output)
        elif step_type == 'valid':
            self.val_output[self.current_epoch].append(output)
        elif step_type == 'test':
            self.test_output[self.num_called_test].append(output)
        
        return total_loss
    
    def training_step(self, batch, batch_idx):
        """
        Training step.
        
        Args:
            batch: Batch data
            batch_idx: Batch index
            
        Returns:
            torch.Tensor: Loss value
        """
        train_total_loss = self._step(batch, 'train')
        self.log('train_total_loss', train_total_loss, batch_size=self.batch_size, on_step=True, on_epoch=True, prog_bar=True)
        return train_total_loss
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step.
        
        Args:
            batch: Batch data
            batch_idx: Batch index
            
        Returns:
            torch.Tensor: Loss value
        """
        val_total_loss = self._step(batch, 'valid')
        self.log('val_total_loss', val_total_loss, batch_size=self.batch_size, on_step=True, on_epoch=True, prog_bar=True)
        return val_total_loss
    
    def test_step(self, batch, batch_idx):
        """
        Test step.
        
        Args:
            batch: Batch data
            batch_idx: Batch index
            
        Returns:
            torch.Tensor: Loss value
        """
        test_total_loss = self._step(batch, 'test')
        self.log('test_total_loss', test_total_loss, batch_size=self.batch_size)
        return test_total_loss
    
    def _epoch_end_report(self, epoch_outputs, epoch_type):
        """
        Generate report at the end of each epoch.
        
        Args:
            epoch_outputs: Outputs collected during the epoch
            epoch_type: Epoch type ('Train', 'Validation', or 'Test')
            
        Returns:
            Tuple: (metrics, y_pred, y_true)
        """
        def flatten_list_of_tensors(lst):
            return np.array([item.cpu().numpy() for sublist in lst for item in sublist])
        
        # Extract predictions and true values
        y_pred = flatten_list_of_tensors([item[0] for item in epoch_outputs])
        y_true = flatten_list_of_tensors([item[1] for item in epoch_outputs])
        
        # Reshape if needed
        y_pred = y_pred.reshape(-1, 1)
        y_true = y_true.reshape(-1, 1)
        
        # Apply inverse transform if scaler is provided
        if self.scaler:
            y_pred = self.scaler.inverse_transform(y_pred).squeeze()
            y_true = self.scaler.inverse_transform(y_true).squeeze()
        
        # Compute metrics
        metrics_dict = compute_regression_metrics(y_true, y_pred, metrics=['mae', 'rmse', 'r2'])
        
        # Convert to list format for backward compatibility
        metrics = [
            metrics_dict['mae'],  # MAE
            metrics_dict['rmse'],  # RMSE
            metrics_dict['r2']    # R2
        ]
        
        # Log metrics
        self.log(f'{epoch_type} MAE', metrics[0], batch_size=self.batch_size)
        self.log(f'{epoch_type} RMSE', metrics[1], batch_size=self.batch_size)
        self.log(f'{epoch_type} R2', metrics[2], batch_size=self.batch_size)
        
        return metrics, y_pred, y_true
    
    def on_train_epoch_end(self):
        """
        Called at the end of the training epoch.
        """
        if self.current_epoch in self.train_output and len(self.train_output[self.current_epoch]) > 0:
            train_metrics, y_pred, y_true = self._epoch_end_report(
                self.train_output[self.current_epoch], epoch_type='Train'
            )
            self.train_metrics[self.current_epoch] = train_metrics
            del self.train_output[self.current_epoch]
    
    def on_validation_epoch_end(self):
        """
        Called at the end of the validation epoch.
        """
        if self.current_epoch in self.val_output and len(self.val_output[self.current_epoch]) > 0:
            val_metrics, y_pred, y_true = self._epoch_end_report(
                self.val_output[self.current_epoch], epoch_type='Validation'
            )
            self.val_metrics[self.current_epoch] = val_metrics
            self.val_true[self.current_epoch] = y_true
            del self.val_output[self.current_epoch]
    
    def on_test_epoch_end(self):
        """
        Called at the end of the test epoch.
        """
        if self.num_called_test in self.test_output and len(self.test_output[self.num_called_test]) > 0:
            test_outputs_per_epoch = self.test_output[self.num_called_test]
            metrics, y_pred, y_true = self._epoch_end_report(
                test_outputs_per_epoch, epoch_type='Test'
            )
            self.test_output[self.num_called_test] = y_pred
            self.test_true[self.num_called_test] = y_true
            self.test_metrics[self.num_called_test] = metrics
            self.num_called_test += 1