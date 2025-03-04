#!/usr/bin/env python
"""
Training script for molecular property prediction models.

This script handles the training pipeline, including:
1. Argument parsing and configuration
2. Dataset loading and preprocessing
3. Model creation and training
4. Evaluation and result saving
5. Ensemble model training and evaluation
"""

# import os
# import numpy as np
# import pandas as pd
# from sklearn.metrics import mean_absolute_error, mean_squared_error
# import os
# from collections import defaultdict
# import os
# import numpy as np
# import pandas as pd
# from collections import defaultdict
# from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import sys
import time
import json
import numpy as np
import logging
import random
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as GeometricDataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint, 
    EarlyStopping, 
    LearningRateMonitor,
    TQDMProgressBar,
    Timer
)
from pytorch_lightning.loggers import (
    TensorBoardLogger, 
    CSVLogger,
    WandbLogger
)
from pytorch_lightning.strategies import (
    DDPStrategy,
    DeepSpeedStrategy,
    FSDPStrategy
)

# Add parent directory to path for imports
parent_path = str(Path(Path(__file__).parent.absolute()).parent.absolute())
sys.path.insert(0, parent_path)

# Import project modules
from config import (
    process_args, 
    save_config, 
    get_model_name,
    print_args_summary,
    setup_logging,
    get_experiment_config
)

from data.load_QM7 import load_QM7
from data.load_QM8 import load_QM8
from data.load_QM9 import load_QM9
from data.load_QMugs import load_QMugs
from data.load_MD17 import load_MD17
from data.load_Reaction import load_reaction

from module.pl_wrap import Estimator
from utils.metrics import (
    compute_regression_metrics,
    compute_classification_metrics,
    RegressionMetrics,
    ClassificationMetrics
)
from utils.visualization import (
    plot_predictions, 
    plot_loss_curves, 
    plot_feature_importance,
    plot_attention_weights,
    plot_learning_rate
)
from utils.model_utils import (
    get_optimizer,
    get_scheduler,
    get_loss_function
)


def seed_worker(worker_id: int) -> None:
    """
    Initialize random seeds for data loader workers to ensure reproducibility.
    
    Args:
        worker_id (int): Worker ID
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def setup_distributed_training(args) -> Dict[str, Any]:
    """
    Set up distributed training configuration.
    
    Args:
        args: Command line arguments
        
    Returns:
        Dict[str, Any]: Configuration for PyTorch Lightning Trainer
    """
    strategy_config = {}
    
    if args.strategy == 'auto':
        # Let Lightning decide the best strategy
        return strategy_config
    
    if args.strategy == 'none':
        strategy_config['strategy'] = None
        return strategy_config
    
    if args.strategy == 'ddp':
        strategy = DDPStrategy(
            find_unused_parameters=False,
            static_graph=True
        )
        strategy_config['strategy'] = strategy
    
    elif args.strategy == 'deepspeed':
        strategy = DeepSpeedStrategy(
            stage=2,  # Optimizer state and gradients partitioned, but model parameters are replicated
            offload_optimizer=False,
            offload_parameters=False,
            allgather_bucket_size=5e8,
            reduce_bucket_size=5e8
        )
        strategy_config['strategy'] = strategy
    
    elif args.strategy == 'fsdp':
        strategy = FSDPStrategy(
            auto_wrap_policy=None,
            activation_checkpointing=False,
            cpu_offload=False
        )
        strategy_config['strategy'] = strategy
    
    # Configure other distributed parameters
    strategy_config['num_nodes'] = args.num_nodes
    strategy_config['devices'] = args.devices if args.devices > 0 else "auto"
    
    return strategy_config


def setup_loggers(args) -> List:
    """
    Set up experiment loggers for tracking metrics.
    
    Args:
        args: Command line arguments
        
    Returns:
        List: List of configured loggers
    """
    loggers = []
    
    # Create experiment directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Always add TensorBoard logger
    if args.logger_type in ['tensorboard', 'all']:
        tb_logger = TensorBoardLogger(
            save_dir=args.output_dir,
            name='tensorboard',
            default_hp_metric=False
        )
        loggers.append(tb_logger)
    
    # Add CSV logger for easy data export
    if args.logger_type in ['csv', 'all']:
        csv_logger = CSVLogger(
            save_dir=args.output_dir,
            name='csv_logs',
            flush_logs_every_n_steps=args.log_every_n_steps
        )
        loggers.append(csv_logger)
    
    # Add Weights & Biases logger if requested
    if args.logger_type in ['wandb', 'all']:
        # Try to import wandb
        try:
            import wandb
            
            # Configure wandb logger
            wandb_logger = WandbLogger(
                project=args.wandb_project,
                name=args.experiment_name,
                save_dir=args.output_dir,
                log_model=True
            )
            
            # Log configuration
            wandb_logger.experiment.config.update(get_experiment_config(args))
            
            loggers.append(wandb_logger)
        except ImportError:
            print("Warning: wandb not installed. Skipping wandb logging.")
    
    return loggers


def setup_callbacks(args) -> List:
    """
    Set up training callbacks.
    
    Args:
        args: Command line arguments
        
    Returns:
        List: List of configured callbacks
    """
    callbacks = []
    
    # Checkpoint callback for saving the best model
    if args.save_best_model:
        checkpoint_callback = ModelCheckpoint(
            monitor='val_total_loss',
            dirpath=os.path.join(args.output_dir, 'checkpoints'),
            filename='best-{epoch:04d}-{val_total_loss:.4f}',
            save_top_k=1,
            mode='min',
            save_weights_only=False,
            verbose=True
        )
        callbacks.append(checkpoint_callback)
    
    # Checkpoint callback for saving the last model
    if args.save_last_model:
        last_checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(args.output_dir, 'checkpoints'),
            filename='last-{epoch:04d}',
            save_top_k=1,
            save_last=True,
            verbose=False
        )
        callbacks.append(last_checkpoint_callback)
    
    # Checkpoint callback for saving models at regular intervals
    if args.save_interval > 0:
        interval_checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(args.output_dir, 'checkpoints'),
            filename='epoch-{epoch:04d}',
            save_top_k=-1,
            every_n_epochs=args.save_interval,
            verbose=False
        )
        callbacks.append(interval_checkpoint_callback)
    
    # Early stopping callback
    if args.early_stopping_patience > 0:
        early_stopping_callback = EarlyStopping(
            monitor='val_total_loss',
            patience=args.early_stopping_patience,
            min_delta=args.early_stopping_min_delta,
            mode='min',
            verbose=True
        )
        callbacks.append(early_stopping_callback)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)
    
    # Timer callback
    timer_callback = Timer(duration=None)
    callbacks.append(timer_callback)
    
    # Progress bar
    if args.progress_bar:
        progress_bar = TQDMProgressBar(refresh_rate=10)
        callbacks.append(progress_bar)
    
    return callbacks


def load_datasets(args, logger) -> Tuple[DataLoader, DataLoader, DataLoader, Any]:
    """
    Load and prepare datasets based on command line arguments.
    
    Args:
        args: Command line arguments
        logger: Logger instance
        
    Returns:
        Tuple[DataLoader, DataLoader, DataLoader, Any]: Train, validation, and test dataloaders, and scaler
    """
    logger.info(f"Loading dataset: {args.dataset}")
    
    # Set number of workers based on dataset
    num_workers = args.num_workers
    if args.dataset in ['benzene', 'aspirin', 'malonaldehyde', 'ethanol', 'toluene']:
        num_workers = 0
    
    # Load the appropriate dataset
    if args.dataset == 'QM7':
        train, val, test, scaler = load_QM7(args.random_seed, args.target_id)
    elif args.dataset == 'QM8':
        train, val, test, scaler = load_QM8(args.random_seed, args.target_id)
    elif args.dataset == 'QM9':
        train, val, test, scaler = load_QM9(
            args.random_seed, 
            args.target_id, 
            download_dir=args.dataset_download_dir
        )
    elif args.dataset == 'QMugs':
        train, val, test, scaler = load_QMugs(args.random_seed, args.target_id)
    elif args.dataset == 'XTB':
        # Use the configurable paths for reaction dataset
        train, val, test, scaler = load_reaction(
            args.random_seed,
            root=args.reaction_dataset_root,
            csv_file=args.reaction_dataset_csv,
            use_scaler = args.use_scaler
        )
    elif args.dataset in ['benzene', 'aspirin', 'malonaldehyde', 'ethanol', 'toluene']:
        train, val, test, scaler = load_MD17(ds=args.dataset, download_dir=args.dataset_download_dir)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    
    # Configure data loader parameters
    dataloader_kwargs = {
        'batch_size': args.batch_size,
        'num_workers': num_workers,
        'worker_init_fn': seed_worker
    }
    
    eval_dataloader_kwargs = {
        'batch_size': args.eval_batch_size,
        'num_workers': num_workers
    }
    
    # Create data loaders with appropriate batch size and worker settings
    if args.dataset == 'XTB':
        dataloader_kwargs['follow_batch'] = ['pos0', 'pos1', 'pos2']
        eval_dataloader_kwargs['follow_batch'] = ['pos0', 'pos1', 'pos2']
        
        train_loader = GeometricDataLoader(
            train, 
            shuffle=True, 
            **dataloader_kwargs
        )
        val_loader = GeometricDataLoader(
            val, 
            shuffle=False, 
            **eval_dataloader_kwargs
        )
        test_loader = GeometricDataLoader(
            test, 
            shuffle=False, 
            **eval_dataloader_kwargs
        )
    else:
        train_loader = GeometricDataLoader(
            train, 
            shuffle=True, 
            **dataloader_kwargs
        )
        val_loader = GeometricDataLoader(
            val, 
            shuffle=False, 
            **eval_dataloader_kwargs
        )
        test_loader = GeometricDataLoader(
            test, 
            shuffle=False, 
            **eval_dataloader_kwargs
        )
    
    logger.info(f"Dataset loaded successfully: {len(train)} train, {len(val)} validation, {len(test)} test samples")
    
    return train_loader, val_loader, test_loader, scaler


def setup_cross_validation_dataloaders(args, logger) -> List[Tuple[DataLoader, DataLoader]]:
    """
    Set up cross-validation dataloaders.
    
    Args:
        args: Command line arguments
        logger: Logger instance
        
    Returns:
        List[Tuple[DataLoader, DataLoader]]: List of (train, val) dataloaders for cross-validation
    """
    logger.info(f"Setting up {args.cross_validation_folds}-fold cross-validation")
    
    # Set number of workers based on dataset
    num_workers = args.num_workers
    if args.dataset in ['benzene', 'aspirin', 'malonaldehyde', 'ethanol', 'toluene']:
        num_workers = 0
    
    # Load the appropriate dataset
    if args.dataset == 'QM7':
        cv_folds, scaler = load_QM7(
            args.random_seed, 
            args.target_id, 
            cv_folds=args.cross_validation_folds
        )
    elif args.dataset == 'QM8':
        cv_folds, scaler = load_QM8(
            args.random_seed, 
            args.target_id, 
            cv_folds=args.cross_validation_folds
        )
    elif args.dataset == 'QM9':
        cv_folds, scaler = load_QM9(
            args.random_seed, 
            args.target_id, 
            download_dir=args.dataset_download_dir,
            cv_folds=args.cross_validation_folds
        )
    elif args.dataset == 'QMugs':
        cv_folds, scaler = load_QMugs(
            args.random_seed, 
            args.target_id,
            cv_folds=args.cross_validation_folds
        )

    elif args.dataset == 'XTB':
        # Use the configurable paths for reaction dataset
        cv_folds, scaler = load_reaction(
            args.random_seed,
            root=args.reaction_dataset_root,
            csv_file=args.reaction_dataset_csv,
            cv_folds=args.cross_validation_folds,
            use_scaler = args.use_scaler
        )
    
    elif args.dataset in ['benzene', 'aspirin', 'malonaldehyde', 'ethanol', 'toluene']:
        cv_folds, scaler = load_MD17(
            ds=args.dataset, 
            download_dir=args.dataset_download_dir,
            cv_folds=args.cross_validation_folds
        )
    else:
        raise ValueError(f"Cross-validation not supported for dataset: {args.dataset}")
    
    # List to store dataloaders for each fold
    cv_dataloaders = []
    
    for fold_idx, (train_data, val_data) in enumerate(cv_folds):
        logger.info(f"Preparing fold {fold_idx+1}/{args.cross_validation_folds}")
        
        # Configure data loader parameters
        dataloader_kwargs = {
            'batch_size': args.batch_size,
            'num_workers': num_workers,
            'worker_init_fn': seed_worker
        }
        
        eval_dataloader_kwargs = {
            'batch_size': args.eval_batch_size,
            'num_workers': num_workers
        }
        
        # Create data loaders with appropriate batch size and worker settings
        if args.dataset == 'XTB':
            dataloader_kwargs['follow_batch'] = ['pos0', 'pos1', 'pos2']
            eval_dataloader_kwargs['follow_batch'] = ['pos0', 'pos1', 'pos2']
        
        train_loader = GeometricDataLoader(
            train_data, 
            shuffle=True, 
            **dataloader_kwargs
        )
        
        val_loader = GeometricDataLoader(
            val_data, 
            shuffle=False, 
            **eval_dataloader_kwargs
        )
        
        cv_dataloaders.append((train_loader, val_loader))
    
    return cv_dataloaders, scaler


def create_model(args, scaler) -> pl.LightningModule:
    """
    Create model instance based on command line arguments.
    
    Args:
        args: Command line arguments
        scaler: Feature scaler
        
    Returns:
        pl.LightningModule: Model instance
    """
    # print(f"args.use_scaler:{args.use_scaler}:::::scaler:{scaler}\n\n\n")
    # Define model configuration
    scaler = scaler if args.use_scaler else None

    model_config = {
        'readout': args.readout,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'max_num_atoms_in_mol': args.max_num_atoms,
        'scaler': scaler,
        'use_layer_norm': args.use_layer_norm,
        'node_latent_dim': args.node_latent_dim,
        'edge_latent_dim': args.edge_latent_dim,
        'dropout': args.dropout,
        'dimenet_hidden_channels': args.dimenet_hidden_channels,
        'dimenet_num_blocks': args.dimenet_num_blocks,
        'dimenet_int_emb_size': args.dimenet_int_emb_size,
        'dimenet_basis_emb_size': args.dimenet_basis_emb_size,
        'dimenet_out_emb_channels': args.dimenet_out_emb_channels,
        'dimenet_num_spherical': args.dimenet_num_spherical,
        'dimenet_num_radial': args.dimenet_num_radial,
        'dimenet_cutoff': args.dimenet_cutoff,
        'dimenet_envelope_exponent': args.dimenet_envelope_exponent,
        'set_transformer_hidden_dim': args.set_transformer_hidden_dim,
        'set_transformer_num_heads': args.set_transformer_num_heads,
        'set_transformer_num_sabs': args.set_transformer_num_sabs,
        'attention_hidden_dim': args.attention_hidden_dim,
        'attention_num_heads': args.attention_num_heads,
        # Optimization parameters
        'optimizer': args.optimizer,
        'weight_decay': args.weight_decay,
        'scheduler': args.scheduler,
        'scheduler_patience': args.scheduler_patience,
        'scheduler_factor': args.scheduler_factor,
        'warmup_epochs': args.warmup_epochs,
        'min_lr': args.min_lr,
        'loss_function': args.loss_function,
        'uncertainty_method': args.uncertainty_method,
        'gradient_clip_val': args.gradient_clip_val,
    }
    
    # Create model
    model = Estimator(**model_config)
    
    return model


def train_model(
    args, 
    model: pl.LightningModule, 
    train_loader: DataLoader, 
    val_loader: DataLoader, 
    loggers: List, 
    callbacks: List,
    logger: logging.Logger
) -> Tuple[pl.Trainer, Dict[str, Any]]:
    """
    Train a model with the given dataloaders.
    
    Args:
        args: Command line arguments
        model: Model instance
        train_loader: Training dataloader
        val_loader: Validation dataloader
        loggers: List of experiment loggers
        callbacks: List of training callbacks
        logger: Logger instance
        
    Returns:
        Tuple[pl.Trainer, Dict[str, Any]]: Trainer instance and training metrics
    """
    logger.info("Initializing training")
    
    # Set up distributed training configuration
    distributed_config = setup_distributed_training(args)
    
    # Configure trainer parameters
    trainer_config = {
        'logger': loggers,
        'callbacks': callbacks,
        'max_epochs': args.max_epochs,
        'min_epochs': args.min_epochs,
        'log_every_n_steps': args.log_every_n_steps,
        'deterministic': True,
        'accelerator': 'gpu' if args.cuda and torch.cuda.is_available() else 'cpu',
        'num_sanity_val_steps': 2,
        'gradient_clip_val': args.gradient_clip_val if args.gradient_clip_val > 0 else None,
        'accumulate_grad_batches': args.gradient_accumulation_steps,
        **distributed_config
    }
    
    # Set precision for mixed/half precision training
    if args.precision in ['16', '32', 'bf16', 'mixed']:
        trainer_config['precision'] = args.precision
    
    # Add checkpoint path if resuming training
    if args.ckpt_path:
        trainer_config['resume_from_checkpoint'] = args.ckpt_path
    
    # Create trainer
    trainer = pl.Trainer(**trainer_config)
    
    # Train model
    start_time = time.time()
    logger.info("Starting training")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    training_time = time.time() - start_time
    
    metrics = {
        'best_epoch': trainer.checkpoint_callback.best_model_path if hasattr(trainer, 'checkpoint_callback') else None,
        'training_time': training_time,
        'epochs_completed': trainer.current_epoch,
    }
    
    logger.info(f"Training completed in {training_time:.2f} seconds")
    logger.info(f"Best epoch: {metrics['best_epoch']}")
    
    return trainer, metrics


def evaluate_model(
    trainer: pl.Trainer, 
    model: pl.LightningModule, 
    test_loader: DataLoader,
    output_dir: str,
    save_predictions: bool,
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    Evaluate a trained model on the test set.
    
    Args:
        trainer: Trainer instance
        model: Trained model
        test_loader: Test dataloader
        output_dir: Output directory
        save_predictions: Whether to save predictions
        logger: Logger instance
        
    Returns:
        Dict[str, Any]: Evaluation metrics
    """
    logger.info("Evaluating model on test set")
    
    # Test model
    test_results = trainer.test(model, dataloaders=test_loader)
    import numpy as np
    import pandas as pd
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import os
    from collections import defaultdict
    import os
    import numpy as np
    import pandas as pd
    from collections import defaultdict
    from sklearn.metrics import mean_absolute_error, mean_squared_error
        
    # 预测和真实值
    y_pred = model.test_output
    y_true = model.test_true
    
    # # 从 defaultdict 中提取数组
    # if isinstance(y_true, defaultdict):
    #     # 假设 defaultdict 中只有一个键，或者我们只关心第一个值
    #     y_true = list(y_true.values())[0]
    
    # if isinstance(y_pred, defaultdict):
    #     y_pred = list(y_pred.values())[0]
    
    # # 确保转换为 NumPy 数组
    # y_true = np.asarray(y_true)
    # y_pred = np.asarray(y_pred)
    
    # # 确保数据类型一致（通常使用 float32 或 float64）
    # y_true = y_true.astype(np.float64)
    # y_pred = y_pred.astype(np.float64)
    
    # # 计算评估指标
    # mae = mean_absolute_error(y_true, y_pred)
    # mse = mean_squared_error(y_true, y_pred)
    # rmse = np.sqrt(mse)
    
    # # 打印详细的评估信息
    # print("\n--- 测试集评估指标 ---")
    # print(f"Mean Absolute Error (MAE): {mae:.4f}")
    # print(f"Mean Squared Error (MSE):  {mse:.4f}")
    # print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    
    # 保存数据
    if save_predictions:
        # 转换为 pandas DataFrame
        y_pred_df = pd.DataFrame(y_pred)
        y_true_df = pd.DataFrame(y_true)
        
        # 保存为 CSV 文件（双重保险）
        y_pred_df.to_csv(os.path.join(output_dir, 'test_y_pred.csv'), index=False, header=False)
        y_true_df.to_csv(os.path.join(output_dir, 'test_y_true.csv'), index=False, header=False)
        
        # 同时保留 NumPy 文件
        np.save(os.path.join(output_dir, 'test_y_pred.npy'), y_pred)
        np.save(os.path.join(output_dir, 'test_y_true.npy'), y_true)

    
    # Get predictions and ground truth
    # y_pred = model.test_output
    # y_true = model.test_true

    # import pandas as pd
    # # Replace the existing saving code with:
    # if save_predictions:
    #     # Convert NumPy arrays to pandas DataFrame
    #     y_pred_df = pd.DataFrame(y_pred)
    #     y_true_df = pd.DataFrame(y_true)
        
    #     # Save as CSV files
    #     y_pred_df.to_csv(os.path.join(output_dir, 'test_y_pred.csv'), index=False, header=False)
    #     y_true_df.to_csv(os.path.join(output_dir, 'test_y_true.csv'), index=False, header=False)
    
    # # Optional: If you want to keep the original .npy files as well
    # # np.save(os.path.join(output_dir, 'test_y_pred.npy'), y_pred)
    # # np.save(os.path.join(output_dir, 'test_y_true.npy'), y_true)
    # # # Save predictions if requested
    # # if save_predictions:
    # #     np.save(os.path.join(output_dir, 'test_y_pred.npy'), y_pred)
    # #     np.save(os.path.join(output_dir, 'test_y_true.npy'), y_true)
    #     # np.save(os.path.join(output_dir, 'test_y_pred.csv'), y_pred)
    #     # np.save(os.path.join(output_dir, 'test_y_true.csv'), y_true)
    
        # Create prediction visualization
    try:
        plot_predictions(y_true, y_pred, output_dir)
    except Exception as e:
        logger.warning(f"Failed to create prediction visualization: {e}")
    
    # Save test metrics
    np.save(os.path.join(output_dir, 'test_metrics.npy'), model.test_metrics)
    
    # Log metrics
    logger.info(f"Test metrics: {test_results[0]}")
    
    return test_results[0]


def train_and_evaluate_ensemble(args, logger) -> Dict[str, Any]:
    """
    Train an ensemble of models.
    
    Args:
        args: Command line arguments
        logger: Logger instance
        
    Returns:
        Dict[str, Any]: Ensemble evaluation metrics
    """
    logger.info(f"Training ensemble of {args.ensemble_size} models")
    
    # Set up output directories for each ensemble member
    ensemble_dirs = []
    for i in range(args.ensemble_size):
        ensemble_dir = os.path.join(args.output_dir, f'ensemble_{i}')
        os.makedirs(ensemble_dir, exist_ok=True)
        ensemble_dirs.append(ensemble_dir)
    
    # Initialize list to store models and metrics
    ensemble_models = []
    ensemble_metrics = []
    
    # Train and evaluate each ensemble member
    for i, (seed, ensemble_dir) in enumerate(zip(args.ensemble_seeds, ensemble_dirs)):
        logger.info(f"Training ensemble member {i+1}/{args.ensemble_size} with seed {seed}")
        
        # Set seed for this ensemble member
        args.random_seed = seed
        pl.seed_everything(seed)
        
        # Load dataset
        train_loader, val_loader, test_loader, scaler = load_datasets(args, logger)
        
        # Create model
        model = create_model(args, scaler)
        
        # Set up loggers
        loggers = setup_loggers(args)
        
        # Set up callbacks
        callbacks = setup_callbacks(args)
        
        # Train model
        trainer, train_metrics = train_model(
            args, model, train_loader, val_loader, loggers, callbacks, logger
        )
        
        # Evaluate model
        test_metrics = evaluate_model(
            trainer, model, test_loader, ensemble_dir, args.save_predictions, logger
        )
        
        # Save metrics
        with open(os.path.join(ensemble_dir, 'metrics.json'), 'w') as f:
            json.dump({'train': train_metrics, 'test': test_metrics}, f, indent=2)
        
        # Store model and metrics
        ensemble_models.append(model)
        ensemble_metrics.append(test_metrics)
    
    # Combine ensemble predictions
    logger.info("Combining ensemble predictions")
    ensemble_output_dir = os.path.join(args.output_dir, 'ensemble')
    os.makedirs(ensemble_output_dir, exist_ok=True)
    
    # Load all predictions
    y_preds = []
    y_true = None
    for i in range(args.ensemble_size):
        ensemble_dir = ensemble_dirs[i]
        y_pred = np.load(os.path.join(ensemble_dir, 'test_y_pred.npy'))
        y_preds.append(y_pred)
        
        if y_true is None:
            y_true = np.load(os.path.join(ensemble_dir, 'test_y_true.npy'))
    
    # Combine predictions based on ensemble method
    if args.ensemble_method == 'mean':
        ensemble_pred = np.mean(y_preds, axis=0)
    elif args.ensemble_method == 'median':
        ensemble_pred = np.median(y_preds, axis=0)
    else:
        # Default to mean
        ensemble_pred = np.mean(y_preds, axis=0)
    
    # Compute ensemble metrics
    ensemble_metrics = compute_regression_metrics(y_true, ensemble_pred)
    
    # Save ensemble predictions and metrics
    np.save(os.path.join(ensemble_output_dir, 'test_y_pred.npy'), ensemble_pred)
    np.save(os.path.join(ensemble_output_dir, 'test_y_true.npy'), y_true)
    
    # Create prediction visualization
    try:
        plot_predictions(y_true, ensemble_pred, ensemble_output_dir)
    except Exception as e:
        logger.warning(f"Failed to create ensemble prediction visualization: {e}")
    
    # Save ensemble metrics
    with open(os.path.join(ensemble_output_dir, 'ensemble_metrics.json'), 'w') as f:
        json.dump(ensemble_metrics, f, indent=2)
    
    logger.info(f"Ensemble metrics: {ensemble_metrics}")
    
    return ensemble_metrics


def perform_cross_validation(args, logger) -> Dict[str, Any]:
    """
    Perform k-fold cross-validation.
    
    Args:
        args: Command line arguments
        logger: Logger instance
        
    Returns:
        Dict[str, Any]: Cross-validation metrics
    """
    logger.info(f"Performing {args.cross_validation_folds}-fold cross-validation")
    
    # Set up cross-validation dataloaders
    cv_dataloaders, scaler = setup_cross_validation_dataloaders(args, logger)
    
    # Initialize list to store metrics for each fold
    fold_metrics = []
    
    # Perform cross-validation
    for fold_idx, (train_loader, val_loader) in enumerate(cv_dataloaders):
        logger.info(f"Training fold {fold_idx+1}/{args.cross_validation_folds}")
        
        # Set up output directory for this fold
        fold_output_dir = os.path.join(args.output_dir, f'fold_{fold_idx+1}')
        os.makedirs(fold_output_dir, exist_ok=True)
        
        # Update output directory for this fold
        fold_args = args
        fold_args.output_dir = fold_output_dir
        
        # Create model
        model = create_model(fold_args, scaler)
        
        # Set up loggers
        loggers = setup_loggers(fold_args)
        
        # Set up callbacks
        callbacks = setup_callbacks(fold_args)
        
        # Train model
        trainer, train_metrics = train_model(
            fold_args, model, train_loader, val_loader, loggers, callbacks, logger
        )
        
        # Evaluate model on validation set (since test set is not available in cross-validation)
        val_metrics = trainer.validate(model, dataloaders=val_loader)
        
        # Save metrics
        with open(os.path.join(fold_output_dir, 'metrics.json'), 'w') as f:
            json.dump({'train': train_metrics, 'val': val_metrics[0]}, f, indent=2)
        
        # Store metrics
        fold_metrics.append(val_metrics[0])
    
    # Compute average metrics across folds
    avg_metrics = {}
    for key in fold_metrics[0].keys():
        avg_metrics[key] = np.mean([fold[key] for fold in fold_metrics])
    
    # Compute standard deviation of metrics
    std_metrics = {}
    for key in fold_metrics[0].keys():
        std_metrics[key] = np.std([fold[key] for fold in fold_metrics])
    
    # Save cross-validation metrics
    cv_metrics = {
        'avg': avg_metrics,
        'std': std_metrics,
        'folds': fold_metrics
    }
    
    with open(os.path.join(args.output_dir, 'cv_metrics.json'), 'w') as f:
        json.dump(cv_metrics, f, indent=2)
    
    logger.info(f"Cross-validation average metrics: {avg_metrics}")
    logger.info(f"Cross-validation standard deviation: {std_metrics}")
    
    return cv_metrics


def visualize_results(output_dir: str, logger: logging.Logger) -> None:
    """
    Generate visualizations of training results.
    
    Args:
        output_dir: Output directory
        logger: Logger instance
    """
    logger.info("Generating result visualizations")
    
    # Load TensorBoard log data
    tensorboard_dir = os.path.join(output_dir, 'tensorboard')
    
    # Check if TensorBoard directory exists
    if not os.path.exists(tensorboard_dir):
        logger.warning(f"TensorBoard directory not found: {tensorboard_dir}")
        return
    
    # Create visualization output directory
    viz_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Plot loss curves
    try:
        plot_loss_curves(tensorboard_dir, viz_dir)
        logger.info("Loss curves plotted successfully")
    except Exception as e:
        logger.warning(f"Failed to plot loss curves: {e}")
    
    # Plot learning rate curve
    try:
        plot_learning_rate(tensorboard_dir, viz_dir)
        logger.info("Learning rate curve plotted successfully")
    except Exception as e:
        logger.warning(f"Failed to plot learning rate curve: {e}")
    
    # Plot attention weights if available

    try:
        plot_attention_weights(output_dir, plot_dir=viz_dir)
        logger.info("Attention weights plotted successfully")
    except Exception as e:
        logger.warning(f"Failed to plot attention weights: {e}")


def parse_command_line_args():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    # Import here to avoid circular imports
    from config import process_args
    
    return process_args()


def main() -> None:
    """
    Main training function.
    """
    # Set number of threads for better reproducibility
    torch.set_num_threads(1)
    
    # Parse command line arguments
    args = parse_command_line_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save configuration
    config_path = save_config(args, args.output_dir)
    
    # Set up logging
    logger = setup_logging(args)
    logger.info(f"Configuration saved to {config_path}")
    
    # Print configuration summary
    print_args_summary(args)
    
    # Set seed for reproducibility
    pl.seed_everything(args.random_seed)
    
    # Perform cross-validation if requested
    if args.cross_validation_folds > 0:
        cv_metrics = perform_cross_validation(args, logger)
        logger.info("Cross-validation completed successfully")
        return
    
    # Train ensemble if requested
    if args.ensemble_size > 1:
        ensemble_metrics = train_and_evaluate_ensemble(args, logger)
        logger.info("Ensemble training completed successfully")
        return
    
    # Regular training and evaluation
    # Load datasets
    train_loader, val_loader, test_loader, scaler = load_datasets(args, logger)
    
    # Create model
    model = create_model(args, scaler)
    logger.info(f"Created model: {model.__class__.__name__}")
    
    # Set up loggers
    loggers = setup_loggers(args)
    
    # Set up callbacks
    callbacks = setup_callbacks(args)
    
    # Train model
    trainer, train_metrics = train_model(
        args, model, train_loader, val_loader, loggers, callbacks, logger
    )
    
    # Evaluate model
    test_metrics = evaluate_model(
        trainer, model, test_loader, args.output_dir, args.save_predictions, logger
    )
    
    # Generate visualizations
    if args.save_visualizations:
        visualize_results(args.output_dir, logger)
    
    # Save final metrics
    metrics = {
        'train': train_metrics,
        'test': test_metrics
    }
    
    with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info("Training and evaluation completed successfully")
    logger.info(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()