#!/usr/bin/env python
"""
Hyperparameter optimization script for molecular property prediction models.

This script performs hyperparameter search using Optuna for finding optimal
model configurations. It supports:
1. Various search spaces for hyperparameters
2. Multiple optimization objectives (validation loss, metrics)
3. Cross-validation during optimization
4. Parallel trials using multiprocessing

Example usage:
    python -m cli.hyperopt --dataset QM9 
                          --target_id 0 
                          --n_trials 50 
                          --optimization_metric val_total_loss
"""

import os
import sys
import time
import json
import argparse
import logging
import random
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime
import multiprocessing

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint, 
    EarlyStopping, 
    LearningRateMonitor,
    TQDMProgressBar
)
from pytorch_lightning.loggers import (
    TensorBoardLogger, 
    CSVLogger
)
from torch_geometric.loader import DataLoader as GeometricDataLoader

# Import optuna for hyperparameter optimization
try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
except ImportError:
    print("Optuna is required for hyperparameter optimization. Install it with: pip install optuna")
    sys.exit(1)

# Add parent directory to path for imports
parent_path = str(Path(Path(__file__).parent.absolute()).parent.absolute())
sys.path.insert(0, parent_path)

# Import project modules
from data.load_QM7 import load_QM7
from data.load_QM8 import load_QM8
from data.load_QM9 import load_QM9
from data.load_QMugs import load_QMugs
from data.load_MD17 import load_MD17
from data.load_Reaction import load_reaction

from module.pl_wrap import Estimator
from utils.metrics import compute_regression_metrics
from utils.visualization import plot_predictions


def seed_worker(worker_id: int) -> None:
    """
    Initialize random seeds for data loader workers to ensure reproducibility.
    
    Args:
        worker_id (int): Worker ID assigned by DataLoader
    """
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def setup_hyperopt_logging(args):
    """
    Set up logging specifically for the hyperparameter optimization script.
    
    Args:
        args: Command line arguments
    
    Returns:
        logging.Logger: Configured logger
    """
    # Create log directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Configure logger
    logger = logging.getLogger("hyperopt")
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear existing handlers
    
    # Create file handler
    file_handler = logging.FileHandler(os.path.join(args.output_dir, "hyperopt.log"))
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def create_hyperopt_argparser() -> argparse.ArgumentParser:
    """
    Create an argument parser specifically for hyperparameter optimization.
    
    Returns:
        argparse.ArgumentParser: Argument parser
    """
    parser = argparse.ArgumentParser(description="Hyperparameter optimization for molecular property prediction models")
    
    # Dataset parameters
    parser.add_argument("--dataset", type=str, default="QM9",
                        choices=["QM7", "QM8", "QM9", "QMugs", "XTB", "benzene", "aspirin", "malonaldehyde", "ethanol", "toluene"],
                        help="Dataset to use for optimization")
    parser.add_argument("--target_id", type=int, default=0,
                        help="Target property index for datasets with multiple targets")
    parser.add_argument("--dataset_download_dir", type=str, default="./data",
                        help="Directory to download datasets")
    parser.add_argument("--reaction_dataset_root", type=str, default="./data/reaction",
                        help="Root directory for reaction dataset")
    parser.add_argument("--reaction_dataset_csv", type=str, default="reaction_data.csv",
                        help="CSV file containing reaction data")
    parser.add_argument("--use_scaler", action="store_true",
                        help="Whether to use a scaler for the target values")
    parser.add_argument("--max_num_atoms", type=int, default=100,
                        help="Maximum number of atoms in a molecule")
    
    # Training parameters
    parser.add_argument("--cross_validation_folds", type=int, default=0,
                        help="Number of cross-validation folds (0 to disable)")
    parser.add_argument("--eval_batch_size", type=int, default=32,
                        help="Batch size for evaluation")
    parser.add_argument("--max_epochs", type=int, default=50,
                        help="Maximum number of epochs for each trial")
    parser.add_argument("--min_epochs", type=int, default=10,
                        help="Minimum number of epochs for each trial")
    parser.add_argument("--early_stopping_patience", type=int, default=10,
                        help="Patience for early stopping")
    parser.add_argument("--early_stopping_min_delta", type=float, default=0.0001,
                        help="Minimum delta for early stopping")
    
    # Optimization parameters
    parser.add_argument("--n_trials", type=int, default=50,
                        help="Number of optimization trials")
    parser.add_argument("--optimization_metric", type=str, default="val_total_loss",
                        choices=["val_total_loss", "val_mae", "val_rmse", "val_r2"],
                        help="Metric to optimize")
    parser.add_argument("--optimization_direction", type=str, default="minimize",
                        choices=["minimize", "maximize"],
                        help="Direction of optimization")
    parser.add_argument("--n_jobs", type=int, default=1,
                        help="Number of parallel optimization jobs")
    parser.add_argument("--timeout", type=int, default=None,
                        help="Timeout for optimization in seconds")
    parser.add_argument("--pruning", action="store_true",
                        help="Whether to use pruning for early termination of unpromising trials")
    
    # Hardware and performance parameters
    parser.add_argument("--cuda", action="store_true",
                        help="Whether to use CUDA")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for data loading")
    parser.add_argument("--precision", type=str, default="32",
                        choices=["16", "32", "bf16", "mixed"],
                        help="Precision for training")
    
    # Logging and output parameters
    parser.add_argument("--output_dir", type=str, default="./outputs/hyperopt",
                        help="Output directory")
    parser.add_argument("--save_best_model", action="store_true",
                        help="Whether to save the best model from each trial")
    parser.add_argument("--random_seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--progress_bar", action="store_true",
                        help="Whether to display progress bar")
    
    return parser


def load_datasets(args, logger) -> Tuple:
    """
    Load and prepare datasets based on command line arguments.
    
    Args:
        args: Command line arguments
        logger: Logger instance
        
    Returns:
        Tuple: Dataset components and scaler
    """
    logger.info(f"Loading dataset: {args.dataset}")
    
    # Set number of workers based on dataset
    num_workers = args.num_workers
    if args.dataset in ['benzene', 'aspirin', 'malonaldehyde', 'ethanol', 'toluene']:
        num_workers = 0
    
    # Load the appropriate dataset - for hyperopt we use cross-validation if specified
    if args.cross_validation_folds > 0:
        logger.info(f"Using {args.cross_validation_folds}-fold cross-validation")
        
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
            cv_folds, scaler = load_reaction(
                args.random_seed,
                root=args.reaction_dataset_root,
                csv_file=args.reaction_dataset_csv,
                cv_folds=args.cross_validation_folds,
                use_scaler=args.use_scaler
            )
        elif args.dataset in ['benzene', 'aspirin', 'malonaldehyde', 'ethanol', 'toluene']:
            cv_folds, scaler = load_MD17(
                ds=args.dataset, 
                download_dir=args.dataset_download_dir,
                cv_folds=args.cross_validation_folds
            )
        else:
            raise ValueError(f"Unsupported dataset: {args.dataset}")
        
        return cv_folds, scaler, num_workers
    
    # If not using cross-validation, use standard train/val/test split
    else:
        logger.info("Using standard train/validation/test split")
        
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
            train, val, test, scaler = load_reaction(
                args.random_seed,
                root=args.reaction_dataset_root,
                csv_file=args.reaction_dataset_csv,
                use_scaler=args.use_scaler
            )
        elif args.dataset in ['benzene', 'aspirin', 'malonaldehyde', 'ethanol', 'toluene']:
            train, val, test, scaler = load_MD17(ds=args.dataset, download_dir=args.dataset_download_dir)
        else:
            raise ValueError(f"Unsupported dataset: {args.dataset}")
        
        return (train, val, test), scaler, num_workers


def create_dataloaders(dataset_components, num_workers, batch_size, eval_batch_size, is_reaction_dataset):
    """
    Create data loaders from dataset components.
    
    Args:
        dataset_components: Dataset components (either train/val/test or cv_folds)
        num_workers: Number of workers for data loading
        batch_size: Batch size for training
        eval_batch_size: Batch size for evaluation
        is_reaction_dataset: Whether the dataset is a reaction dataset
    
    Returns:
        Tuple or List[Tuple]: Data loaders for training, validation, and testing
    """
    # Configure data loader parameters
    dataloader_kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'worker_init_fn': seed_worker
    }
    
    eval_dataloader_kwargs = {
        'batch_size': eval_batch_size,
        'num_workers': num_workers
    }
    
    if is_reaction_dataset:
        dataloader_kwargs['follow_batch'] = ['pos0', 'pos1', 'pos2']
        eval_dataloader_kwargs['follow_batch'] = ['pos0', 'pos1', 'pos2']
    
    # Check if we have cross-validation folds or standard split
    if isinstance(dataset_components[0], tuple):
        # Cross-validation folds
        cv_dataloaders = []
        
        for train_data, val_data in dataset_components:
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
        
        return cv_dataloaders
    else:
        # Standard train/val/test split
        train, val, test = dataset_components
        
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
        
        return train_loader, val_loader, test_loader


def define_model_hyperparameters(trial):
    """
    Define the hyperparameter search space for the model.
    
    Args:
        trial: Optuna trial object
        
    Returns:
        Dict: Hyperparameter dictionary
    """
    # Select readout type
    readout = trial.suggest_categorical("readout", [
        "mean", "sum", "max", "attention", "set_transformer"
    ])
    
    # Basic model parameters
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
    node_latent_dim = trial.suggest_categorical("node_latent_dim", [64, 128, 256])
    use_layer_norm = trial.suggest_categorical("use_layer_norm", [True, False])
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    
    # Optimizer parameters
    optimizer = trial.suggest_categorical("optimizer", ["adam", "adamw"])
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    
    # Scheduler parameters
    scheduler = trial.suggest_categorical("scheduler", ["cosine", "plateau", "none"])
    
    # DimeNet++ parameters
    dimenet_hidden_channels = trial.suggest_categorical("dimenet_hidden_channels", [64, 128, 256])
    dimenet_num_blocks = trial.suggest_int("dimenet_num_blocks", 2, 6)
    
    # Readout-specific parameters
    if readout == "set_transformer":
        set_transformer_hidden_dim = trial.suggest_categorical("set_transformer_hidden_dim", [128, 256, 512])
        set_transformer_num_heads = trial.suggest_categorical("set_transformer_num_heads", [4, 8, 16])
        set_transformer_num_sabs = trial.suggest_int("set_transformer_num_sabs", 1, 3)
    else:
        # Default values for non-set-transformer readouts
        set_transformer_hidden_dim = 128
        set_transformer_num_heads = 4
        set_transformer_num_sabs = 1
    
    if readout == "attention":
        attention_hidden_dim = trial.suggest_categorical("attention_hidden_dim", [64, 128, 256])
        attention_num_heads = trial.suggest_categorical("attention_num_heads", [1, 2, 4, 8])
    else:
        # Default values for non-attention readouts
        attention_hidden_dim = 128
        attention_num_heads = 4
    
    # Loss function
    loss_function = trial.suggest_categorical("loss_function", ["mse", "mae", "huber"])
    
    # Gather all parameters
    params = {
        "readout": readout,
        "batch_size": batch_size,
        "node_latent_dim": node_latent_dim,
        "use_layer_norm": use_layer_norm,
        "dropout": dropout,
        "optimizer": optimizer,
        "lr": lr,
        "weight_decay": weight_decay,
        "scheduler": scheduler,
        "dimenet_hidden_channels": dimenet_hidden_channels,
        "dimenet_num_blocks": dimenet_num_blocks,
        "set_transformer_hidden_dim": set_transformer_hidden_dim,
        "set_transformer_num_heads": set_transformer_num_heads,
        "set_transformer_num_sabs": set_transformer_num_sabs,
        "attention_hidden_dim": attention_hidden_dim,
        "attention_num_heads": attention_num_heads,
        "loss_function": loss_function,
    }
    
    return params


def setup_model_with_hyperparams(params, scaler, max_num_atoms):
    """
    Create model instance with given hyperparameters.
    
    Args:
        params: Hyperparameter dictionary
        scaler: Scaler for target values
        max_num_atoms: Maximum number of atoms in a molecule
        
    Returns:
        pl.LightningModule: Model instance
    """
    model_config = {
        'readout': params['readout'],
        'batch_size': params['batch_size'],
        'lr': params['lr'],
        'max_num_atoms_in_mol': max_num_atoms,
        'scaler': scaler,
        'use_layer_norm': params['use_layer_norm'],
        'node_latent_dim': params['node_latent_dim'],
        'dropout': params['dropout'],
        'optimizer': params['optimizer'],
        'weight_decay': params['weight_decay'],
        'scheduler': params['scheduler'],
        'dimenet_hidden_channels': params['dimenet_hidden_channels'],
        'dimenet_num_blocks': params['dimenet_num_blocks'],
        'set_transformer_hidden_dim': params['set_transformer_hidden_dim'],
        'set_transformer_num_heads': params['set_transformer_num_heads'],
        'set_transformer_num_sabs': params['set_transformer_num_sabs'],
        'attention_hidden_dim': params['attention_hidden_dim'],
        'attention_num_heads': params['attention_num_heads'],
        'loss_function': params['loss_function'],
    }
    
    return Estimator(**model_config)


def setup_training_callbacks(args, trial_dir):
    """
    Set up training callbacks.
    
    Args:
        args: Command line arguments
        trial_dir: Directory for trial outputs
        
    Returns:
        List: List of configured callbacks
    """
    callbacks = []
    
    # Checkpoint callback for saving the best model
    if args.save_best_model:
        checkpoint_callback = ModelCheckpoint(
            monitor=args.optimization_metric,
            dirpath=os.path.join(trial_dir, 'checkpoints'),
            filename='best-{epoch:04d}-{' + args.optimization_metric + ':.4f}',
            save_top_k=1,
            mode='min' if args.optimization_direction == 'minimize' else 'max',
            save_weights_only=False,
            verbose=False
        )
        callbacks.append(checkpoint_callback)
    
    # Early stopping callback
    early_stopping_callback = EarlyStopping(
        monitor=args.optimization_metric,
        patience=args.early_stopping_patience,
        min_delta=args.early_stopping_min_delta,
        mode='min' if args.optimization_direction == 'minimize' else 'max',
        verbose=False
    )
    callbacks.append(early_stopping_callback)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)
    
    # Progress bar
    if args.progress_bar:
        progress_bar = TQDMProgressBar(refresh_rate=10)
        callbacks.append(progress_bar)
    
    return callbacks


def setup_loggers(trial_dir):
    """
    Set up experiment loggers for tracking metrics.
    
    Args:
        trial_dir: Directory for trial outputs
        
    Returns:
        List: List of configured loggers
    """
    loggers = []
    
    # Create experiment directory
    os.makedirs(trial_dir, exist_ok=True)
    
    # Add TensorBoard logger
    tb_logger = TensorBoardLogger(
        save_dir=trial_dir,
        name='tensorboard',
        default_hp_metric=False,
        version=''
    )
    loggers.append(tb_logger)
    
    # Add CSV logger for easy data export
    csv_logger = CSVLogger(
        save_dir=trial_dir,
        name='csv_logs',
        version='',
        flush_logs_every_n_steps=10
    )
    loggers.append(csv_logger)
    
    return loggers
# Create sampler
    sampler = TPESampler(seed=args.random_seed)
    
    # Create pruner
    pruner = MedianPruner() if args.pruning else None
    
    # Create study
    study = optuna.create_study(
        direction=args.optimization_direction,
        sampler=sampler,
        pruner=pruner
    )
    
    # Create objective function with fixed arguments
    objective_fixed = lambda trial: objective(trial, args, dataset_info, logger)
    
    # Run optimization
    try:
        logger.info(f"Starting optimization with {args.n_trials} trials")
        study.optimize(
            objective_fixed, 
            n_trials=args.n_trials,
            n_jobs=args.n_jobs,
            timeout=args.timeout
        )
    except KeyboardInterrupt:
        logger.info("Optimization interrupted by user")
    
    # Get best trial
    best_trial = study.best_trial
    
    # Log best trial
    logger.info(f"Best trial: {best_trial.number}")
    logger.info(f"Best {args.optimization_metric}: {best_trial.value:.6f}")
    logger.info("Best hyperparameters:")
    for key, value in best_trial.params.items():
        logger.info(f"  {key}: {value}")
    
    # Save study results
    study_results = {
        'best_trial': {
            'number': best_trial.number,
            'value': best_trial.value,
            'params': best_trial.params
        },
        'all_trials': [
            {
                'number': trial.number,
                'value': trial.value,
                'params': trial.params,
                'state': str(trial.state)
            }
            for trial in study.trials
        ]
    }
    
    # Save study results to JSON
    study_results_path = os.path.join(args.output_dir, 'study_results.json')
    with open(study_results_path, 'w') as f:
        json.dump(study_results, f, indent=2)
    logger.info(f"Study results saved to {study_results_path}")
    
    # Create importance plot
    try:
        from optuna.visualization import plot_param_importances
        import matplotlib.pyplot as plt
        
        # Create importance plot
        fig = plot_param_importances(study)
        fig_path = os.path.join(args.output_dir, 'param_importances.png')
        fig.write_image(fig_path)
        logger.info(f"Parameter importance plot saved to {fig_path}")
        
        # Create optimization history plot
        from optuna.visualization import plot_optimization_history
        fig = plot_optimization_history(study)
        fig_path = os.path.join(args.output_dir, 'optimization_history.png')
        fig.write_image(fig_path)
        logger.info(f"Optimization history plot saved to {fig_path}")
        
        # Create parallel coordinate plot
        from optuna.visualization import plot_parallel_coordinate
        fig = plot_parallel_coordinate(study)
        fig_path = os.path.join(args.output_dir, 'parallel_coordinate.png')
        fig.write_image(fig_path)
        logger.info(f"Parallel coordinate plot saved to {fig_path}")
        
        # Create slice plot
        from optuna.visualization import plot_slice
        fig = plot_slice(study)
        fig_path = os.path.join(args.output_dir, 'slice.png')
        fig.write_image(fig_path)
        logger.info(f"Slice plot saved to {fig_path}")
    except ImportError:
        logger.warning("Could not create visualization plots. Make sure plotly is installed.")
    
    # Create final model with best hyperparameters
    if args.save_best_model:
        logger.info("Training final model with best hyperparameters")
        
        # Create final model directory
        final_model_dir = os.path.join(args.output_dir, 'final_model')
        os.makedirs(final_model_dir, exist_ok=True)
        
        # Create final model
        final_model = setup_model_with_hyperparams(best_trial.params, dataset_info[1], args.max_num_atoms)
        
        # Set up callbacks
        callbacks = setup_training_callbacks(args, final_model_dir)
        
        # Set up loggers
        loggers = setup_loggers(final_model_dir)
        
        # Create final dataloaders
        if args.cross_validation_folds > 0:
            # For CV, use all data for training
            all_train_data = []
            all_val_data = []
            
            for train_data, val_data in dataset_info[0]:
                all_train_data.extend(train_data)
                all_val_data.extend(val_data)
            
            # Create dataloaders
            train_loader = GeometricDataLoader(
                all_train_data, 
                shuffle=True, 
                batch_size=best_trial.params['batch_size'],
                num_workers=dataset_info[2],
                worker_init_fn=seed_worker
            )
            
            val_loader = GeometricDataLoader(
                all_val_data, 
                shuffle=False, 
                batch_size=args.eval_batch_size,
                num_workers=dataset_info[2]
            )
        else:
            # For standard split, use train/val loaders
            train_loader, val_loader, _ = create_dataloaders(
                dataset_info[0], 
                dataset_info[2], 
                best_trial.params['batch_size'], 
                args.eval_batch_size, 
                args.dataset == 'XTB'
            )
        
        # Configure trainer
        trainer_config = {
            'logger': loggers,
            'callbacks': callbacks,
            'max_epochs': args.max_epochs * 2,  # Double epochs for final model
            'min_epochs': args.min_epochs,
            'log_every_n_steps': 10,
            'deterministic': True,
            'accelerator': 'gpu' if args.cuda and torch.cuda.is_available() else 'cpu',
            'num_sanity_val_steps': 2,
        }
        
        # Set precision
        if args.precision in ['16', '32', 'bf16', 'mixed']:
            trainer_config['precision'] = args.precision
        
        # Create trainer
        trainer = pl.Trainer(**trainer_config)
        
        # Train final model
        trainer.fit(final_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        
        # Save final model configuration
        final_config = {
            'hyperparameters': best_trial.params,
            'training': {
                'epochs': trainer.current_epoch,
                'batch_size': best_trial.params['batch_size'],
                'optimizer': best_trial.params['optimizer'],
                'learning_rate': best_trial.params['lr'],
                'weight_decay': best_trial.params['weight_decay'],
                'scheduler': best_trial.params['scheduler'],
            },
            'model': {
                'readout': best_trial.params['readout'],
                'node_latent_dim': best_trial.params['node_latent_dim'],
                'dimenet_hidden_channels': best_trial.params['dimenet_hidden_channels'],
                'dimenet_num_blocks': best_trial.params['dimenet_num_blocks'],
            },
            'dataset': {
                'name': args.dataset,
                'target_id': args.target_id,
            }
        }
        
        # Save final configuration
        with open(os.path.join(final_model_dir, 'config.json'), 'w') as f:
            json.dump(final_config, f, indent=2)
        
        logger.info(f"Final model trained and saved to {final_model_dir}")
    
    logger.info("Hyperparameter optimization completed successfully")


if __name__ == "__main__":
    main()