"""
Model utility functions for optimizers, schedulers, and loss functions.

This module provides helper functions for creating optimizers, learning rate schedulers,
and loss functions for different machine learning models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, 
    StepLR, 
    ReduceLROnPlateau, 
    ExponentialLR,
    CyclicLR,
    OneCycleLR,
    LambdaLR
)
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
import numpy as np
import math


def get_optimizer(
    model: nn.Module,
    optimizer_type: str = 'adam',
    lr: float = 0.001,
    weight_decay: float = 0.0,
    **kwargs
) -> torch.optim.Optimizer:
    """
    Get optimizer for model training.
    
    Args:
        model (nn.Module): PyTorch model
        optimizer_type (str, optional): Type of optimizer. Defaults to 'adam'.
        lr (float, optional): Learning rate. Defaults to 0.001.
        weight_decay (float, optional): Weight decay (L2 penalty). Defaults to 0.0.
        **kwargs: Additional optimizer-specific parameters
        
    Returns:
        torch.optim.Optimizer: PyTorch optimizer
        
    Raises:
        ValueError: If optimizer type is not supported
    """
    # Get model parameters
    params = model.parameters()
    
    # Create optimizer based on type
    optimizer_type = optimizer_type.lower()
    
    if optimizer_type == 'adam':
        optimizer = optim.Adam(
            params,
            lr=lr,
            weight_decay=weight_decay,
            betas=kwargs.get('betas', (0.9, 0.999)),
            eps=kwargs.get('eps', 1e-8)
        )
    elif optimizer_type == 'adamw':
        optimizer = optim.AdamW(
            params,
            lr=lr,
            weight_decay=weight_decay,
            betas=kwargs.get('betas', (0.9, 0.999)),
            eps=kwargs.get('eps', 1e-8)
        )
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD(
            params,
            lr=lr,
            momentum=kwargs.get('momentum', 0.9),
            weight_decay=weight_decay,
            nesterov=kwargs.get('nesterov', False)
        )
    elif optimizer_type == 'rmsprop':
        optimizer = optim.RMSprop(
            params,
            lr=lr,
            weight_decay=weight_decay,
            momentum=kwargs.get('momentum', 0.0),
            alpha=kwargs.get('alpha', 0.99),
            eps=kwargs.get('eps', 1e-8),
            centered=kwargs.get('centered', False)
        )
    elif optimizer_type == 'adagrad':
        optimizer = optim.Adagrad(
            params,
            lr=lr,
            weight_decay=weight_decay,
            lr_decay=kwargs.get('lr_decay', 0.0),
            eps=kwargs.get('eps', 1e-10)
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    
    return optimizer


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = 'cosine',
    max_epochs: int = 100,
    min_lr: float = 1e-6,
    steps_per_epoch: int = None,
    warmup_epochs: int = 0,
    **kwargs
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    Get learning rate scheduler.
    
    Args:
        optimizer (torch.optim.Optimizer): PyTorch optimizer
        scheduler_type (str, optional): Type of scheduler. Defaults to 'cosine'.
        max_epochs (int, optional): Maximum number of epochs. Defaults to 100.
        min_lr (float, optional): Minimum learning rate. Defaults to 1e-6.
        steps_per_epoch (int, optional): Number of steps per epoch. Required for some schedulers.
        warmup_epochs (int, optional): Number of warmup epochs. Defaults to 0.
        **kwargs: Additional scheduler-specific parameters
        
    Returns:
        Optional[torch.optim.lr_scheduler._LRScheduler]: PyTorch scheduler or None if no scheduling
        
    Raises:
        ValueError: If scheduler type is not supported
    """
    scheduler_type = scheduler_type.lower()
    
    # Handle case where no scheduler is requested
    if scheduler_type == 'none' or scheduler_type == 'constant':
        return None
    
    # Define base scheduler based on type
    if scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max_epochs - warmup_epochs,
            eta_min=min_lr
        )
    elif scheduler_type == 'step':
        step_size = kwargs.get('step_size', max_epochs // 3)
        gamma = kwargs.get('gamma', 0.1)
        scheduler = StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma
        )
    elif scheduler_type == 'multistep':
        milestones = kwargs.get('milestones', [max_epochs // 3, 2 * max_epochs // 3])
        gamma = kwargs.get('gamma', 0.1)
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=gamma
        )
    elif scheduler_type == 'exponential':
        gamma = kwargs.get('gamma', 0.95)
        scheduler = ExponentialLR(
            optimizer,
            gamma=gamma
        )
    elif scheduler_type == 'plateau':
        patience = kwargs.get('patience', 10)
        factor = kwargs.get('factor', 0.5)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=factor,
            patience=patience,
            min_lr=min_lr,
            verbose=True
        )
    elif scheduler_type == 'cyclic':
        if steps_per_epoch is None:
            raise ValueError("steps_per_epoch must be provided for cyclic scheduler")
        
        base_lr = min_lr
        max_lr = optimizer.param_groups[0]['lr']
        step_size = kwargs.get('step_size_up', steps_per_epoch * 2)
        
        scheduler = CyclicLR(
            optimizer,
            base_lr=base_lr,
            max_lr=max_lr,
            step_size_up=step_size,
            mode=kwargs.get('mode', 'triangular2'),
            cycle_momentum=False
        )
    elif scheduler_type == 'one_cycle':
        if steps_per_epoch is None:
            raise ValueError("steps_per_epoch must be provided for one_cycle scheduler")
        
        scheduler = OneCycleLR(
            optimizer,
            max_lr=optimizer.param_groups[0]['lr'],
            total_steps=steps_per_epoch * max_epochs,
            pct_start=kwargs.get('pct_start', 0.3),
            div_factor=kwargs.get('div_factor', 25.0),
            final_div_factor=kwargs.get('final_div_factor', 10000.0)
        )
    elif scheduler_type == 'warmup_cosine':
        # Create cosine annealing scheduler
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max_epochs - warmup_epochs,
            eta_min=min_lr
        )
        
        # Create a lambda function for linear warmup
        if warmup_epochs > 0:
            def lr_lambda(epoch):
                if epoch < warmup_epochs:
                    # Linear warmup
                    return float(epoch) / float(max(1, warmup_epochs))
                else:
                    # Use cosine scheduler after warmup
                    return cosine_scheduler.get_lr()[0] / optimizer.param_groups[0]['initial_lr']
            
            scheduler = LambdaLR(optimizer, lr_lambda)
        else:
            # No warmup, just return cosine scheduler
            scheduler = cosine_scheduler
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
    
    return scheduler


def get_loss_function(
    loss_type: str = 'mse',
    **kwargs
) -> Callable:
    """
    Get loss function for model training.
    
    Args:
        loss_type (str, optional): Type of loss function. Defaults to 'mse'.
        **kwargs: Additional loss-specific parameters
        
    Returns:
        Callable: Loss function
        
    Raises:
        ValueError: If loss function type is not supported
    """
    loss_type = loss_type.lower()
    
    if loss_type == 'mse':
        return nn.MSELoss(reduction=kwargs.get('reduction', 'mean'))
    elif loss_type == 'mae' or loss_type == 'l1':
        return nn.L1Loss(reduction=kwargs.get('reduction', 'mean'))
    elif loss_type == 'huber':
        delta = kwargs.get('delta', 1.0)
        return nn.HuberLoss(delta=delta, reduction=kwargs.get('reduction', 'mean'))
    elif loss_type == 'smooth_l1':
        beta = kwargs.get('beta', 1.0)
        return nn.SmoothL1Loss(beta=beta, reduction=kwargs.get('reduction', 'mean'))
    elif loss_type == 'cross_entropy':
        weight = kwargs.get('weight', None)
        return nn.CrossEntropyLoss(weight=weight, reduction=kwargs.get('reduction', 'mean'))
    elif loss_type == 'binary_cross_entropy':
        weight = kwargs.get('weight', None)
        return nn.BCELoss(weight=weight, reduction=kwargs.get('reduction', 'mean'))
    elif loss_type == 'binary_cross_entropy_with_logits':
        weight = kwargs.get('weight', None)
        pos_weight = kwargs.get('pos_weight', None)
        return nn.BCEWithLogitsLoss(weight=weight, pos_weight=pos_weight, reduction=kwargs.get('reduction', 'mean'))
    elif loss_type == 'evidence_lower_bound':
        # Evidential regression loss
        def evidential_loss(pred, target):
            # Assuming pred has 4 outputs: [mean, var, alpha, beta]
            mean = pred[:, 0].unsqueeze(1)
            var = torch.nn.functional.softplus(pred[:, 1]).unsqueeze(1)  # Ensure positive
            alpha = torch.nn.functional.softplus(pred[:, 2]).unsqueeze(1) + 1.0  # alpha > 1
            beta = torch.nn.functional.softplus(pred[:, 3]).unsqueeze(1)  # beta > 0
            
            # Compute NLL (Negative Log Likelihood) term
            nll_loss = 0.5 * torch.log(np.pi / var) - alpha * torch.log(beta) \
                     + (alpha + 0.5) * torch.log(beta + (target - mean)**2 / (2 * var)) \
                     + torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)
            
            # Compute KL divergence term
            kl_div = alpha * torch.log(beta) - 0.5 * torch.log(var) \
                   - alpha + 0.5 \
                   + torch.lgamma(alpha) - 0.5 * torch.log(2 * np.pi)
            
            # Regularization term (optional)
            reg_term = kwargs.get('reg_weight', 0.01) * (torch.abs(mean - target) / (2 * beta * (1 + var)))
            
            # Total loss
            loss = nll_loss + kl_div + reg_term
            return torch.mean(loss)
        
        return evidential_loss
    else:
        raise ValueError(f"Unsupported loss function type: {loss_type}")


def count_parameters(model: nn.Module) -> Dict[str, Any]:
    """
    Count parameters in a model.
    
    Args:
        model (nn.Module): PyTorch model
        
    Returns:
        Dict[str, Any]: Dictionary with parameter counts
    """
    # Count total parameters
    total_params = sum(p.numel() for p in model.parameters())
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Count parameters by layer type
    params_by_layer = {}
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf module
            num_params = sum(p.numel() for p in module.parameters())
            if num_params > 0:
                layer_type = module.__class__.__name__
                if layer_type not in params_by_layer:
                    params_by_layer[layer_type] = 0
                params_by_layer[layer_type] += num_params
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params,
        'by_layer': params_by_layer
    }


def model_to_fp16(model: nn.Module) -> nn.Module:
    """
    Convert model parameters to fp16.
    
    Args:
        model (nn.Module): PyTorch model
        
    Returns:
        nn.Module: Model with fp16 parameters
    """
    for param in model.parameters():
        param.data = param.data.half()
    
    return model


def model_to_fp32(model: nn.Module) -> nn.Module:
    """
    Convert model parameters to fp32.
    
    Args:
        model (nn.Module): PyTorch model
        
    Returns:
        nn.Module: Model with fp32 parameters
    """
    for param in model.parameters():
        param.data = param.data.float()
    
    return model


def weight_init(m: nn.Module) -> None:
    """
    Initialize weights for a model.
    
    Args:
        m (nn.Module): PyTorch module
    """
    if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def freeze_layers(model: nn.Module, freeze_names: List[str]) -> nn.Module:
    """
    Freeze layers in a model.
    
    Args:
        model (nn.Module): PyTorch model
        freeze_names (List[str]): Names of layers to freeze
        
    Returns:
        nn.Module: Model with frozen layers
    """
    for name, param in model.named_parameters():
        for freeze_name in freeze_names:
            if freeze_name in name:
                param.requires_grad = False
    
    return model


def unfreeze_layers(model: nn.Module, unfreeze_names: List[str]) -> nn.Module:
    """
    Unfreeze layers in a model.
    
    Args:
        model (nn.Module): PyTorch model
        unfreeze_names (List[str]): Names of layers to unfreeze
        
    Returns:
        nn.Module: Model with unfrozen layers
    """
    for name, param in model.named_parameters():
        for unfreeze_name in unfreeze_names:
            if unfreeze_name in name:
                param.requires_grad = True
    
    return model


def create_warmup_scheduler_function(warmup_epochs: int) -> Callable:
    """
    Create a learning rate warmup function.
    
    Args:
        warmup_epochs (int): Number of warmup epochs
        
    Returns:
        Callable: Learning rate scaling function
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs + 1)
        else:
            return 1.0
    
    return lr_lambda


class WarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Learning rate scheduler with linear warmup.
    
    Args:
        optimizer (torch.optim.Optimizer): Optimizer
        warmup_epochs (int): Number of warmup epochs
        scheduler (torch.optim.lr_scheduler._LRScheduler): Scheduler to use after warmup
        last_epoch (int, optional): The index of the last epoch. Default: -1.
    """
    
    def __init__(
        self, 
        optimizer: torch.optim.Optimizer, 
        warmup_epochs: int, 
        scheduler: torch.optim.lr_scheduler._LRScheduler, 
        last_epoch: int = -1
    ):
        self.warmup_epochs = warmup_epochs
        self.scheduler = scheduler
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        """
        Get learning rate based on current epoch.
        
        Returns:
            List[float]: Learning rates for each parameter group
        """
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            return [
                base_lr * (self.last_epoch + 1) / self.warmup_epochs
                for base_lr in self.base_lrs
            ]
        else:
            # Use the main scheduler
            return self.scheduler.get_lr()
    
    def step(self, epoch=None) -> None:
        """
        Update learning rate.
        
        Args:
            epoch (int, optional): Epoch index. Default: None.
        """
        if epoch is None:
            epoch = self.last_epoch + 1
        
        self.last_epoch = epoch
        
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
                param_group['lr'] = lr
        else:
            # Use the main scheduler
            if isinstance(self.scheduler, ReduceLROnPlateau):
                # ReduceLROnPlateau requires a validation metric
                # We'll update it in the calling code
                pass
            else:
                self.scheduler.step()


class EMA:
    """
    Exponential Moving Average for model parameters.
    
    Args:
        model (nn.Module): PyTorch model
        decay (float, optional): EMA decay rate. Defaults to 0.999.
    """
    
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Register parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self) -> None:
        """
        Update EMA parameters.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self) -> None:
        """
        Apply EMA parameters to model.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self) -> None:
        """
        Restore original parameters.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

def seed_worker(worker_id: int) -> None:
    """
    Initialize random seeds for data loader workers to ensure reproducibility.
    
    PyTorch DataLoader creates workers that need separate seeding to ensure 
    reproducibility. This function should be passed to DataLoader's 
    worker_init_fn parameter.
    
    Args:
        worker_id (int): Worker ID assigned by DataLoader
    """
    import random
    import numpy as np
    import torch
    
    # Get base seed from torch's global seed
    worker_seed = torch.initial_seed() % 2**32
    
    # Seed Python's random module
    random.seed(worker_seed)
    
    # Seed NumPy's random module
    np.random.seed(worker_seed)
    
    # Note: PyTorch's RNG for this worker is already seeded by initial_seed()