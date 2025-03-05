#!/usr/bin/env python
"""
Fine-tuning script for molecular property prediction models.

This script provides a command-line interface for fine-tuning pre-trained models
on new datasets or for specific tasks. It supports:
1. Loading and configuring pre-trained models
2. Fine-tuning on new datasets with flexible hyperparameter control
3. Evaluation and model saving

"""

import os
import sys
import time
import json
import argparse
import logging
import numpy as np
import random
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime

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

# Add parent directory to path for imports
parent_path = str(Path(Path(__file__).parent.absolute()).parent.absolute())
sys.path.insert(0, parent_path)

# Import project modules
from cli.config import process_args, save_config, get_model_name, print_args_summary, setup_logging, \
    get_experiment_config

from data.load_QM7 import load_QM7
from data.load_QM8 import load_QM8
from data.load_QM9 import load_QM9
from data.load_QMugs import load_QMugs
from data.load_MD17 import load_MD17
from data.load_Reaction import load_reaction

from module.pl_wrap import Estimator
from utils.metrics import compute_regression_metrics
from utils.visualization import plot_predictions


# Define seed_worker function if it's not in utils.model_utils
def seed_worker(worker_id: int) -> None:
    """
    Initialize random seeds for data loader workers to ensure reproducibility.

    Args:
        worker_id (int): Worker ID assigned by DataLoader
    """
    worker_seed = torch.initial_seed() % 2 ** 32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def setup_finetune_logging(args):
    """
    Set up logging specifically for the fine-tuning script.

    Args:
        args: Command line arguments

    Returns:
        logging.Logger: Configured logger
    """
    # Create log directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Configure logger
    logger = logging.getLogger("finetune")
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear existing handlers

    # Create file handler
    file_handler = logging.FileHandler(os.path.join(args.output_dir, "finetune.log"))
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


def create_finetune_argparser() -> argparse.ArgumentParser:
    """
    Create an argument parser specifically for fine-tuning.

    Returns:
        argparse.ArgumentParser: Argument parser
    """
    parser = argparse.ArgumentParser(description="Fine-tune a pre-trained molecular property prediction model")

    # Model loading parameters
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to pre-trained model checkpoint")
    parser.add_argument("--strict_loading", type=bool, default=True,
                        help="Whether to strictly enforce that the keys in checkpoint match the keys in model")

    # Dataset parameters
    parser.add_argument("--dataset", type=str, default="QM9",
                        choices=["QM7", "QM8", "QM9", "QMugs", "XTB", "benzene", "aspirin", "malonaldehyde", "ethanol",
                                 "toluene"],
                        help="Dataset to use for fine-tuning")
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
    parser.add_argument("--use_dataset_scaler", action="store_true", default=False,
                        help="Use a new scaler from the fine-tuning dataset instead of the pre-trained model's scaler")
    parser.add_argument("--preserve_original_scaler", action="store_true", default=True,
                        help="Preserve the original scaler from the pre-trained model (ignored if --use_dataset_scaler is set)")
    parser.add_argument("--max_num_atoms", type=int, default=100,
                        help="Maximum number of atoms in a molecule")

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for fine-tuning")
    parser.add_argument("--eval_batch_size", type=int, default=32,
                        help="Batch size for evaluation")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate for fine-tuning")
    parser.add_argument("--max_epochs", type=int, default=20,
                        help="Maximum number of epochs for fine-tuning")
    parser.add_argument("--min_epochs", type=int, default=1,
                        help="Minimum number of epochs for fine-tuning")
    parser.add_argument("--early_stopping_patience", type=int, default=10,
                        help="Patience for early stopping (0 to disable)")
    parser.add_argument("--early_stopping_min_delta", type=float, default=0.0001,
                        help="Minimum delta for early stopping")

    # Fine-tuning strategy
    parser.add_argument("--freeze_backbone", action="store_true",
                        help="Whether to freeze the backbone (feature extractor) during fine-tuning")
    parser.add_argument("--freeze_layers", type=str, default="",
                        help="Comma-separated list of layer names to freeze (e.g., 'net,readout_module')")

    # Optimizer parameters
    parser.add_argument("--optimizer", type=str, default="adam",
                        choices=["adam", "adamw", "sgd", "rmsprop"],
                        help="Optimizer to use for fine-tuning")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="Weight decay for optimizer")
    parser.add_argument("--scheduler", type=str, default="cosine",
                        choices=["none", "cosine", "step", "plateau", "exponential"],
                        help="Learning rate scheduler to use")
    parser.add_argument("--scheduler_patience", type=int, default=5,
                        help="Patience for plateau scheduler")
    parser.add_argument("--scheduler_factor", type=float, default=0.5,
                        help="Factor for plateau scheduler")
    parser.add_argument("--warmup_epochs", type=int, default=0,
                        help="Number of warmup epochs")
    parser.add_argument("--min_lr", type=float, default=1e-6,
                        help="Minimum learning rate")
    parser.add_argument("--loss_function", type=str, default="mse",
                        choices=["mse", "mae", "huber", "smooth_l1"],
                        help="Loss function to use")
    parser.add_argument("--gradient_clip_val", type=float, default=0.0,
                        help="Gradient clipping value (0 to disable)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of steps to accumulate gradients")

    # Hardware and performance parameters
    parser.add_argument("--cuda", action="store_true",
                        help="Whether to use CUDA")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for data loading")
    parser.add_argument("--precision", type=str, default="32",
                        choices=["16", "32", "bf16", "mixed"],
                        help="Precision for training")

    # Logging and output parameters
    parser.add_argument("--output_dir", type=str, default="./outputs/finetuned",
                        help="Output directory")
    parser.add_argument("--experiment_name", type=str, default=None,
                        help="Experiment name (defaults to dataset_timestamp)")
    parser.add_argument("--random_seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--log_every_n_steps", type=int, default=50,
                        help="Log every N steps")
    parser.add_argument("--save_best_model", action="store_true",
                        help="Whether to save the best model")
    parser.add_argument("--save_last_model", action="store_true",
                        help="Whether to save the last model")
    parser.add_argument("--save_interval", type=int, default=0,
                        help="Save checkpoint every N epochs (0 to disable)")
    parser.add_argument("--save_predictions", action="store_true",
                        help="Whether to save predictions")
    parser.add_argument("--save_visualizations", action="store_true",
                        help="Whether to save visualizations")
    parser.add_argument("--progress_bar", action="store_true",
                        help="Whether to display progress bar")

    return parser


def load_pretrained_model(
        args,
        checkpoint_path: str,
        logger: logging.Logger
) -> pl.LightningModule:
    """
    Load a pre-trained model from a checkpoint file.

    Args:
        args: Command line arguments
        checkpoint_path: Path to checkpoint file
        logger: Logger instance

    Returns:
        pl.LightningModule: Loaded model
    """
    logger.info(f"Loading pre-trained model from {checkpoint_path}")

    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    # Load checkpoint to get hyperparameters
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Extract hyperparameters from checkpoint
    if 'hyper_parameters' in checkpoint:
        hparams = checkpoint['hyper_parameters']
        logger.info(f"Loaded hyperparameters from checkpoint")

        # Check for scaler in hyperparameters
        if 'scaler' in hparams:
            if hparams['scaler'] is not None:
                logger.info("Found scaler in model hyperparameters")
                # Logging scaler details if available
                scaler = hparams['scaler']
                logger.info(f"Model checkpoint scaler type: {type(scaler).__name__}")
                if hasattr(scaler, 'mean_'):
                    logger.info(f"Model scaler mean: {scaler.mean_}")
                if hasattr(scaler, 'scale_'):
                    logger.info(f"Model scaler scale: {scaler.scale_}")
                if hasattr(scaler, 'var_'):
                    logger.info(f"Model scaler variance: {scaler.var_}")
            else:
                logger.info("Scaler in model hyperparameters is None")
        else:
            logger.info("No scaler found in model hyperparameters")
    else:
        # If hyperparameters not in checkpoint, use default values
        logger.warning("No hyperparameters found in checkpoint, using defaults")
        hparams = {}

    # Ensure max_num_atoms is set
    if 'max_num_atoms_in_mol' not in hparams and 'max_num_atoms' not in hparams:
        logger.info(f"Setting max_num_atoms_in_mol to {args.max_num_atoms}")
        hparams['max_num_atoms_in_mol'] = args.max_num_atoms

    # Create model with hyperparameters from checkpoint
    try:
        model = Estimator(**hparams)
    except TypeError as e:
        logger.warning(f"Error creating model with checkpoint hyperparameters: {e}")
        logger.warning("Attempting to create model with minimal hyperparameters")

        # Try with minimal hyperparameters
        minimal_hparams = {
            'readout': hparams.get('readout', 'mean'),
            'batch_size': args.batch_size,
            'lr': args.lr,
            'max_num_atoms_in_mol': hparams.get('max_num_atoms_in_mol', args.max_num_atoms),
        }
        model = Estimator(**minimal_hparams)

    # Load weights from checkpoint
    try:
        model.load_state_dict(checkpoint['state_dict'], strict=args.strict_loading)
    except Exception as e:
        logger.error(f"Error loading state dict: {e}")
        if not args.strict_loading:
            logger.warning("Continuing with partial model loading")
        else:
            raise

    # Check if scaler is available in the loaded model
    if hasattr(model, 'scaler') and model.scaler is not None:
        logger.info(f"Loaded model has a scaler of type: {type(model.scaler).__name__}")
        if hasattr(model.scaler, 'mean_'):
            logger.info(f"Original model scaler mean: {model.scaler.mean_}")
        if hasattr(model.scaler, 'scale_'):
            logger.info(f"Original model scaler scale: {model.scaler.scale_}")
    else:
        logger.warning("Loaded model does not have a scaler attached")

    # Modify model for fine-tuning
    if args.freeze_backbone:
        logger.info("Freezing backbone layers")
        for param in model.net.parameters():
            param.requires_grad = False

    # Freeze specific layers if requested
    if args.freeze_layers:
        layer_names = args.freeze_layers.split(',')
        logger.info(f"Freezing layers: {layer_names}")

        for name, param in model.named_parameters():
            for layer_name in layer_names:
                if layer_name in name:
                    param.requires_grad = False
                    break

    # Update learning rate for fine-tuning
    model.lr = args.lr

    # Update other hyperparameters for fine-tuning
    model.optimizer_type = args.optimizer
    model.weight_decay = args.weight_decay
    model.scheduler_type = args.scheduler
    model.scheduler_patience = args.scheduler_patience
    model.scheduler_factor = args.scheduler_factor
    model.warmup_epochs = args.warmup_epochs
    model.min_lr = args.min_lr
    model.loss_function = args.loss_function

    # Log trainable parameter count
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model has {trainable_params:,} trainable parameters out of {total_params:,} total parameters")

    return model


def load_datasets(args, logger) -> Tuple[GeometricDataLoader, GeometricDataLoader, GeometricDataLoader, Any]:
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
            use_scaler=args.use_scaler
        )
    elif args.dataset in ['benzene', 'aspirin', 'malonaldehyde', 'ethanol', 'toluene']:
        train, val, test, scaler = load_MD17(ds=args.dataset, download_dir=args.dataset_download_dir)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    # Log scaler information if available
    if args.use_scaler and scaler is not None:
        logger.info(f"Dataset loaded with scaler type: {type(scaler).__name__}")
        if hasattr(scaler, 'mean_'):
            logger.info(f"Dataset scaler mean: {scaler.mean_}")
        if hasattr(scaler, 'scale_'):
            logger.info(f"Dataset scaler scale: {scaler.scale_}")
    else:
        if args.use_scaler:
            logger.warning("use_scaler is True but no scaler was returned from dataset loader")
        else:
            logger.info("No scaler being used from dataset (use_scaler=False)")

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

    # Progress bar
    if args.progress_bar:
        progress_bar = TQDMProgressBar(refresh_rate=10)
        callbacks.append(progress_bar)

    return callbacks


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

    # Add TensorBoard logger
    tb_logger = TensorBoardLogger(
        save_dir=args.output_dir,
        name='tensorboard',
        default_hp_metric=False
    )
    loggers.append(tb_logger)

    # Add CSV logger for easy data export
    csv_logger = CSVLogger(
        save_dir=args.output_dir,
        name='csv_logs',
        flush_logs_every_n_steps=args.log_every_n_steps
    )
    loggers.append(csv_logger)

    return loggers


def fine_tune_model(
        args,
        model: pl.LightningModule,
        train_loader: GeometricDataLoader,
        val_loader: GeometricDataLoader,
        loggers: List,
        callbacks: List,
        logger: logging.Logger
) -> Tuple[pl.Trainer, Dict[str, Any]]:
    """
    Fine-tune a model with the given dataloaders.

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
    logger.info("Initializing fine-tuning")

    # Verify model's scaler state before fine-tuning
    if hasattr(model, 'scaler') and model.scaler is not None:
        logger.info(f"Model has scaler of type for fine-tuning: {type(model.scaler).__name__}")
        if hasattr(model.scaler, 'mean_'):
            logger.info(f"Fine-tuning model scaler mean: {model.scaler.mean_}")
        if hasattr(model.scaler, 'scale_'):
            logger.info(f"Fine-tuning model scaler scale: {model.scaler.scale_}")
    else:
        logger.info("Model does not have a scaler for fine-tuning")

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
    }

    # Set precision for mixed/half precision training
    if args.precision in ['16', '32', 'bf16', 'mixed']:
        trainer_config['precision'] = args.precision

    # Create trainer
    trainer = pl.Trainer(**trainer_config)

    # Fine-tune model
    start_time = time.time()
    logger.info("Starting fine-tuning")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    fine_tuning_time = time.time() - start_time

    metrics = {
        'best_epoch': getattr(trainer.checkpoint_callback, 'best_model_path', None) if hasattr(trainer,
                                                                                               'checkpoint_callback') else None,
        'fine_tuning_time': fine_tuning_time,
        'epochs_completed': trainer.current_epoch,
    }

    logger.info(f"Fine-tuning completed in {fine_tuning_time:.2f} seconds")
    logger.info(f"Best epoch: {metrics['best_epoch']}")

    # Verify model's scaler state after fine-tuning
    if hasattr(model, 'scaler') and model.scaler is not None:
        logger.info(f"After fine-tuning, model has scaler of type: {type(model.scaler).__name__}")
        if hasattr(model.scaler, 'mean_'):
            logger.info(f"Final model scaler mean: {model.scaler.mean_}")
        if hasattr(model.scaler, 'scale_'):
            logger.info(f"Final model scaler scale: {model.scaler.scale_}")

    return trainer, metrics


def evaluate_model(
        trainer: pl.Trainer,
        model: pl.LightningModule,
        test_loader: GeometricDataLoader,
        output_dir: str,
        save_predictions: bool,
        logger: logging.Logger
) -> Dict[str, Any]:
    """
    Evaluate a fine-tuned model on the test set.

    Args:
        trainer: Trainer instance
        model: Fine-tuned model
        test_loader: Test dataloader
        output_dir: Output directory
        save_predictions: Whether to save predictions
        logger: Logger instance

    Returns:
        Dict[str, Any]: Evaluation metrics
    """
    logger.info("Evaluating model on test set")

    # Verify model's scaler state before evaluation
    if hasattr(model, 'scaler') and model.scaler is not None:
        logger.info(f"Model has scaler of type for evaluation: {type(model.scaler).__name__}")
        if hasattr(model.scaler, 'mean_'):
            logger.info(f"Evaluation model scaler mean: {model.scaler.mean_}")
        if hasattr(model.scaler, 'scale_'):
            logger.info(f"Evaluation model scaler scale: {model.scaler.scale_}")
    else:
        logger.info("Model does not have a scaler for evaluation")

    # Test model
    test_results = trainer.test(model, dataloaders=test_loader)

    # Get predictions and ground truth
    y_pred = model.test_output
    y_true = model.test_true

    # Save predictions if requested
    if save_predictions:
        # Convert to pandas DataFrame for CSV output
        import pandas as pd

        # Handle defaultdict if needed
        if hasattr(y_pred, 'keys'):
            # Extract the first key (typically '1')
            y_pred = y_pred[list(y_pred.keys())[0]]

        if hasattr(y_true, 'keys'):
            # Extract the first key (typically '1')
            y_true = y_true[list(y_true.keys())[0]]

        # Convert to DataFrame
        y_pred_df = pd.DataFrame(y_pred)
        y_true_df = pd.DataFrame(y_true)

        # Save as CSV files
        y_pred_df.to_csv(os.path.join(output_dir, 'test_y_pred.csv'), index=False, header=False)
        y_true_df.to_csv(os.path.join(output_dir, 'test_y_true.csv'), index=False, header=False)

        # Also save as NumPy files
        np.save(os.path.join(output_dir, 'test_y_pred.npy'), y_pred)
        np.save(os.path.join(output_dir, 'test_y_true.npy'), y_true)

    # Create prediction visualization
    try:
        plot_predictions(y_true, y_pred, output_dir)
    except Exception as e:
        logger.warning(f"Failed to create prediction visualization: {e}")

    # Save test metrics
    if hasattr(model, 'test_metrics'):
        np.save(os.path.join(output_dir, 'test_metrics.npy'), model.test_metrics)

    # Log metrics
    logger.info(f"Test metrics: {test_results[0]}")

    return test_results[0]


def main():
    """
    Main fine-tuning function.
    """
    # Parse command line arguments
    parser = create_finetune_argparser()
    args = parser.parse_args()

    # Set number of threads for better reproducibility
    torch.set_num_threads(1)

    # Create output directory
    if not args.experiment_name:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.experiment_name = f"{args.dataset}_{timestamp}"

    args.output_dir = os.path.join(args.output_dir, args.experiment_name)
    os.makedirs(args.output_dir, exist_ok=True)

    # Set up logging (using our custom function)
    logger = setup_finetune_logging(args)

    # Save configuration
    config_path = os.path.join(args.output_dir, 'finetune_config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    logger.info(f"Configuration saved to {config_path}")

    # Print configuration summary
    logger.info("Fine-tuning configuration:")
    for key, value in vars(args).items():
        logger.info(f"  {key}: {value}")

    # Set seed for reproducibility
    logger.info(f"Setting random seed to {args.random_seed}")
    pl.seed_everything(args.random_seed)

    # Load pre-trained model
    model = load_pretrained_model(args, args.model_path, logger)

    # Load datasets
    train_loader, val_loader, test_loader, scaler = load_datasets(args, logger)

    # Handle scaler based on command line arguments
    if args.use_scaler:
        if args.use_dataset_scaler:
            logger.info("Using new scaler from fine-tuning dataset")
            model.scaler = scaler
        elif not args.preserve_original_scaler or not hasattr(model, 'scaler') or model.scaler is None:
            logger.info("Using dataset scaler because no model scaler is available or preservation is disabled")
            model.scaler = scaler
        else:
            logger.info("Preserving original scaler from pre-trained model")
            # Log information about the preserved scaler
            if hasattr(model.scaler, 'mean_'):
                logger.info(f"Original scaler mean: {model.scaler.mean_}")
            if hasattr(model.scaler, 'scale_'):
                logger.info(f"Original scaler scale: {model.scaler.scale_}")
    else:
        logger.info("No scaler will be used (use_scaler=False)")
        model.scaler = None

    # Set up loggers
    loggers = setup_loggers(args)

    # Set up callbacks
    callbacks = setup_callbacks(args)

    # Fine-tune model
    trainer, train_metrics = fine_tune_model(
        args, model, train_loader, val_loader, loggers, callbacks, logger
    )

    # Evaluate model
    test_metrics = evaluate_model(
        trainer, model, test_loader, args.output_dir, args.save_predictions, logger
    )

    # Save final metrics
    metrics = {
        'train': train_metrics,
        'test': test_metrics
    }

    with open(os.path.join(args.output_dir, 'finetune_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    logger.info("Fine-tuning and evaluation completed successfully")
    logger.info(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()