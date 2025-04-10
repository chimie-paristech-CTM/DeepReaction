#!/usr/bin/env python

import os
import sys
import time
import json
import numpy as np
import logging
import random
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
from datetime import datetime
import copy
import argparse
import traceback

# Import Optuna for Bayesian optimization
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

import torch
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as GeometricDataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    TQDMProgressBar
)
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

# Add parent directories to path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(script_dir))
deep_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, deep_dir)


class BayesOptEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy types."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.float32):
            return float(obj)
        if isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        return json.JSONEncoder.default(self, obj)


def clean_processed_dir(root_dir):
    """Clean potentially corrupted files in the processed directory."""
    processed_dir = os.path.join(root_dir, 'processed')
    if os.path.exists(processed_dir):
        try:
            for filename in os.listdir(processed_dir):
                file_path = os.path.join(processed_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    logging.info(f"Removed potentially corrupted file: {file_path}")
        except Exception as e:
            logging.warning(f"Error cleaning processed directory: {e}")


def create_experiment_name(params):
    """Create a descriptive name for the experiment based on parameters."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"bayesopt_cut{params['cutoff']}_blk{params['num_blocks']}_" \
           f"phl{params['prediction_hidden_layers']}_phd{params['prediction_hidden_dim']}_{timestamp}"
    return name


def get_parser():
    """Create and return the argument parser."""
    # Import the parser from config.py to maintain consistency
    from cli.config import get_parser
    parser = get_parser()

    # Add Bayesian optimization specific arguments
    bayesopt_group = parser.add_argument_group('Bayesian Optimization')
    bayesopt_group.add_argument('--cutoff_min', type=float, default=5.0,
                                help='Minimum cutoff distance value')
    bayesopt_group.add_argument('--cutoff_max', type=float, default=15.0,
                                help='Maximum cutoff distance value')
    bayesopt_group.add_argument('--num_blocks_min', type=int, default=4,
                                help='Minimum number of blocks value')
    bayesopt_group.add_argument('--num_blocks_max', type=int, default=6,
                                help='Maximum number of blocks value')
    bayesopt_group.add_argument('--prediction_hidden_layers_min', type=int, default=3,
                                help='Minimum prediction hidden layers value')
    bayesopt_group.add_argument('--prediction_hidden_layers_max', type=int, default=5,
                                help='Maximum prediction hidden layers value')
    bayesopt_group.add_argument('--prediction_hidden_dim_min', type=int, default=128,
                                help='Minimum prediction hidden dimension value')
    bayesopt_group.add_argument('--prediction_hidden_dim_max', type=int, default=512,
                                help='Maximum prediction hidden dimension value')
    bayesopt_group.add_argument('--n_trials', type=int, default=20,
                                help='Number of optimization trials')
    bayesopt_group.add_argument('--metric_for_best', type=str, default='val_mae',
                                help='Metric to use for selecting best hyperparameters')
    bayesopt_group.add_argument('--metric_mode', type=str, default='min', choices=['min', 'max'],
                                help='Whether to minimize or maximize the metric')
    bayesopt_group.add_argument('--study_name', type=str, default=None,
                                help='Name of the Optuna study')
    bayesopt_group.add_argument('--storage', type=str, default=None,
                                help='Database URL for Optuna storage')
    bayesopt_group.add_argument('--clean_data', action='store_true', default=True,
                                help='Clean processed data directory before loading')
    bayesopt_group.add_argument('--batch_size_min', type=int, default=4,
                                help='Minimum batch size to try')
    bayesopt_group.add_argument('--batch_size_max', type=int, default=32,
                                help='Maximum batch size to try')

    return parser


def parse_command_line_args():
    """Parse and process command line arguments."""
    parser = get_parser()
    args = parser.parse_args()

    # Process derived arguments
    if args.eval_batch_size is None:
        args.eval_batch_size = args.batch_size

    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"{args.dataset}_{args.model_type}_{args.readout}_seed{args.random_seed}_{timestamp}"

    args.output_dir = os.path.join(
        args.out_dir,
        args.experiment_name
    )

    args.max_num_atoms = 100

    if args.reaction_target_fields is not None and args.target_weights is None:
        args.target_weights = [1.0] * len(args.reaction_target_fields)

    if args.study_name is None:
        args.study_name = f"bayesopt_study_{args.dataset}_{args.model_type}_{args.readout}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Force cv_folds to 0 to disable cross-validation
    args.cv_folds = 0

    return args


def setup_logging(args):
    """Setup logging for the experiment."""
    os.makedirs(args.output_dir, exist_ok=True)

    log_level = getattr(logging, args.log_level.upper())

    logger = logging.getLogger('deep')
    logger.setLevel(log_level)

    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if args.log_to_file:
        log_file = os.path.join(args.output_dir, 'bayesopt.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def save_config(args, output_dir):
    """Save configuration to YAML and JSON files."""
    import yaml
    os.makedirs(output_dir, exist_ok=True)

    config_dict = vars(args)

    yaml_path = os.path.join(output_dir, 'config.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)

    json_path = os.path.join(output_dir, 'config.json')
    with open(json_path, 'w') as f:
        json.dump(config_dict, f, indent=2)

    return yaml_path


def create_dataloader(dataset, batch_size, eval_mode=False, num_workers=4, **kwargs):
    """Create a data loader for a dataset."""
    loader_kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'worker_init_fn': lambda worker_id: np.random.seed(torch.initial_seed() % 2 ** 32) if not eval_mode else None,
        'shuffle': not eval_mode,
        **kwargs
    }

    return GeometricDataLoader(dataset, **loader_kwargs)


def setup_loggers(args, output_dir=None):
    """Set up loggers for PyTorch Lightning."""
    output_dir = output_dir or args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    loggers = []

    if args.logger_type in ['tensorboard', 'all']:
        loggers.append(TensorBoardLogger(
            save_dir=output_dir,
            name='tensorboard',
            default_hp_metric=False
        ))

    if args.logger_type in ['csv', 'all']:
        loggers.append(CSVLogger(
            save_dir=output_dir,
            name='csv_logs',
            flush_logs_every_n_steps=args.log_every_n_steps
        ))

    return loggers


def setup_callbacks(args, output_dir=None):
    """Set up callbacks for PyTorch Lightning."""
    output_dir = output_dir or args.output_dir
    callbacks = []

    if args.save_best_model:
        callbacks.append(ModelCheckpoint(
            monitor='val_total_loss',
            dirpath=os.path.join(output_dir, 'checkpoints'),
            filename='best-{epoch:04d}-{val_total_loss:.4f}',
            save_top_k=1,
            mode='min',
            save_weights_only=False,
            verbose=True
        ))

    if args.save_last_model:
        callbacks.append(ModelCheckpoint(
            dirpath=os.path.join(output_dir, 'checkpoints'),
            filename='last-{epoch:04d}',
            save_top_k=1,
            save_last=True,
            verbose=False
        ))

    if args.save_interval > 0:
        callbacks.append(ModelCheckpoint(
            dirpath=os.path.join(output_dir, 'checkpoints'),
            filename='epoch-{epoch:04d}',
            save_top_k=-1,
            every_n_epochs=args.save_interval,
            verbose=False
        ))

    if args.early_stopping_patience > 0:
        callbacks.append(EarlyStopping(
            monitor='val_total_loss',
            patience=args.early_stopping_patience,
            min_delta=args.early_stopping_min_delta,
            mode='min',
            verbose=True
        ))

    callbacks.extend([
        LearningRateMonitor(logging_interval='epoch')
    ])

    if args.progress_bar:
        callbacks.append(TQDMProgressBar(refresh_rate=10))

    return callbacks


def objective(trial, args, logger):
    """Objective function for Optuna optimization using a single train/val/test split."""
    # Define hyperparameter search space
    cutoff = trial.suggest_categorical('cutoff', [5, 10, 15])
    num_blocks = trial.suggest_int('num_blocks', args.num_blocks_min, args.num_blocks_max)
    prediction_hidden_layers = trial.suggest_int('prediction_hidden_layers',
                                                 args.prediction_hidden_layers_min,
                                                 args.prediction_hidden_layers_max)
    prediction_hidden_dim = trial.suggest_categorical('prediction_hidden_dim',
                                                      [128, 256, 512])
    # Dynamically adjust batch size based on GPU memory availability
    batch_size = trial.suggest_categorical('batch_size',
                                           [16])

    # Create a copy of args with updated hyperparameters
    trial_args = copy.deepcopy(args)
    trial_args.cutoff = cutoff
    trial_args.num_blocks = num_blocks
    trial_args.prediction_hidden_layers = prediction_hidden_layers
    trial_args.prediction_hidden_dim = prediction_hidden_dim
    trial_args.batch_size = batch_size
    trial_args.eval_batch_size = batch_size

    # Create a unique experiment name for this trial
    experiment_name = create_experiment_name({
        'cutoff': cutoff,
        'num_blocks': num_blocks,
        'prediction_hidden_layers': prediction_hidden_layers,
        'prediction_hidden_dim': prediction_hidden_dim
    })

    trial_args.experiment_name = experiment_name
    trial_args.output_dir = os.path.join(args.out_dir, experiment_name)
    os.makedirs(trial_args.output_dir, exist_ok=True)

    # Save configuration
    save_config(trial_args, trial_args.output_dir)

    # Set seed for reproducibility
    pl.seed_everything(trial_args.random_seed)

    logger.info(f"Trial {trial.number}: Hyperparameters - cutoff={cutoff}, " +
                f"num_blocks={num_blocks}, prediction_hidden_layers={prediction_hidden_layers}, " +
                f"prediction_hidden_dim={prediction_hidden_dim}, batch_size={batch_size}")

    # Clean up processed directory to avoid corrupted files
    if hasattr(trial_args, 'clean_data') and trial_args.clean_data:
        clean_processed_dir(trial_args.reaction_dataset_root)

    # Import necessary modules
    from data.load_Reaction import load_reaction
    from train import create_model, train_model, evaluate_model

    try:
        # Load dataset with regular train/val/test split
        train_data, val_data, test_data, scalers = load_reaction(
            trial_args.random_seed,
            root=trial_args.reaction_dataset_root,
            dataset_csv=trial_args.dataset_csv,
            train_ratio=trial_args.train_ratio,
            val_ratio=trial_args.val_ratio,
            test_ratio=trial_args.test_ratio,
            use_scaler=trial_args.use_scaler,
            target_fields=trial_args.reaction_target_fields,
            file_suffixes=trial_args.reaction_file_suffixes,
            input_features=trial_args.input_features,
            val_csv=trial_args.val_csv,
            test_csv=trial_args.test_csv
        )
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        logger.info("Cleaning processed directory and trying again...")
        clean_processed_dir(trial_args.reaction_dataset_root)

        # Try again with force_reload=True
        train_data, val_data, test_data, scalers = load_reaction(
            trial_args.random_seed,
            root=trial_args.reaction_dataset_root,
            dataset_csv=trial_args.dataset_csv,
            train_ratio=trial_args.train_ratio,
            val_ratio=trial_args.val_ratio,
            test_ratio=trial_args.test_ratio,
            use_scaler=trial_args.use_scaler,
            target_fields=trial_args.reaction_target_fields,
            file_suffixes=trial_args.reaction_file_suffixes,
            input_features=trial_args.input_features,
            val_csv=trial_args.val_csv,
            test_csv=trial_args.test_csv,
            force_reload=True
        )

    # Create data loaders
    follow_batch = ['z0', 'z1', 'z2', 'pos0', 'pos1', 'pos2'] if trial_args.dataset == 'XTB' else None
    dataloader_kwargs = {'follow_batch': follow_batch} if follow_batch else {}

    train_loader = create_dataloader(
        train_data, trial_args.batch_size, eval_mode=False,
        num_workers=trial_args.num_workers, **dataloader_kwargs
    )

    val_loader = create_dataloader(
        val_data, trial_args.eval_batch_size, eval_mode=True,
        num_workers=trial_args.num_workers, **dataloader_kwargs
    )

    test_loader = create_dataloader(
        test_data, trial_args.eval_batch_size, eval_mode=True,
        num_workers=trial_args.num_workers, **dataloader_kwargs
    )

    # Create model, loggers, and callbacks
    model = create_model(trial_args, scalers)
    trial_loggers = setup_loggers(trial_args)
    trial_callbacks = setup_callbacks(trial_args)

    try:
        # Train the model
        trainer, train_metrics = train_model(
            trial_args, model, train_loader, val_loader, trial_loggers, trial_callbacks, logger
        )

        # Evaluate on test set
        test_metrics = evaluate_model(
            trainer, model, test_loader, trial_args.output_dir, trial_args.save_predictions,
            trial_args.reaction_target_fields, logger
        )

        # Get validation metrics from the model
        val_metrics = {}
        if hasattr(model, 'val_metrics'):
            val_metrics = model.val_metrics

        # Collect all metrics from different sources
        val_mae = val_metrics.get('val_mae', float('inf'))
        val_rmse = val_metrics.get('val_rmse', float('inf'))

        # If val metrics not found, try to get them from test metrics
        if val_mae == float('inf'):
            for k, v in test_metrics.items():
                if 'Validation Avg MAE' in k or 'Val Avg MAE' in k:
                    val_mae = float(v)
                elif 'Validation Avg RMSE' in k or 'Val Avg RMSE' in k:
                    val_rmse = float(v)

        # If still not found, use test metrics as fallback
        if val_mae == float('inf'):
            test_mae = 0.0
            test_rmse = 0.0

            for k, v in test_metrics.items():
                if 'Test Avg MAE' in k:
                    test_mae = float(v)
                    val_mae = test_mae  # Use test metric as fallback
                elif 'Test Avg RMSE' in k:
                    test_rmse = float(v)
                    val_rmse = test_rmse  # Use test metric as fallback

        # Save results
        result_summary = {
            'hyperparameters': {
                'cutoff': cutoff,
                'num_blocks': num_blocks,
                'prediction_hidden_layers': prediction_hidden_layers,
                'prediction_hidden_dim': prediction_hidden_dim,
                'batch_size': batch_size
            },
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'train_metrics': train_metrics,
            'trial_number': trial.number
        }

        with open(os.path.join(trial_args.output_dir, 'trial_summary.json'), 'w') as f:
            json.dump(result_summary, f, indent=2, cls=BayesOptEncoder)

        logger.info(f"Trial {trial.number} completed successfully")
        logger.info(f"Validation MAE: {val_mae:.6f}, Validation RMSE: {val_rmse:.6f}")

        # Return the objective metric (default is val_mae)
        if args.metric_for_best == 'val_mae':
            return val_mae
        elif args.metric_for_best == 'val_rmse':
            return val_rmse
        elif args.metric_for_best == 'test_mae':
            return test_mae if 'test_mae' in locals() else float('inf')
        elif args.metric_for_best == 'test_rmse':
            return test_rmse if 'test_rmse' in locals() else float('inf')
        else:
            return val_mae  # Default fallback

    except Exception as e:
        logger.error(f"Error in trial {trial.number}: {e}")
        logger.error(traceback.format_exc())

        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Return a poor score
        return float('inf') if args.metric_mode == 'min' else float('-inf')


def main():
    """Main function to run Bayesian hyperparameter optimization."""
    # Set CUDA environment variables for deterministic training
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)

    # Set PyTorch memory allocation configuration to reduce fragmentation
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Parse command line arguments
    args = parse_command_line_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Setup logging
    logger = setup_logging(args)
    logger.info("Starting Bayesian hyperparameter optimization with Optuna (No CV)")

    # Set up output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save configuration
    config_path = save_config(args, args.output_dir)
    logger.info(f"Configuration saved to {config_path}")

    # Clean processed data directory before starting
    if args.clean_data:
        logger.info("Cleaning processed data directory before starting hyperparameter search")
        clean_processed_dir(args.reaction_dataset_root)

    # Configure Optuna study direction
    direction = 'minimize' if args.metric_mode == 'min' else 'maximize'

    # Set up pruner and sampler
    pruner = MedianPruner(n_startup_trials=2, n_warmup_steps=0, interval_steps=1)
    sampler = TPESampler(seed=args.random_seed)

    # Create Optuna storage if specified
    storage = args.storage

    # Create or load Optuna study
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction=direction,
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True
    )

    start_time = time.time()

    # Run optimization
    try:
        study.optimize(
            lambda trial: objective(trial, args, logger),
            n_trials=args.n_trials,
            timeout=None,
            catch=(Exception,),
            show_progress_bar=True
        )
    except KeyboardInterrupt:
        logger.info("Optimization interrupted by user!")
    except Exception as e:
        logger.error(f"Optimization failed with error: {e}")
        logger.error(traceback.format_exc())

    # Calculate total optimization time
    total_time = time.time() - start_time

    # Get best trial and results
    best_trial = study.best_trial
    best_params = best_trial.params
    best_value = best_trial.value

    # Summarize results
    optimization_results = {
        'total_time': total_time,
        'num_trials': len(study.trials),
        'best_params': best_params,
        'best_value': best_value,
        'best_trial_number': best_trial.number,
        'metric_for_best': args.metric_for_best,
        'metric_direction': direction,
        'all_trials': [
            {
                'trial_number': t.number,
                'params': t.params,
                'value': t.value,
                'state': t.state.name
            }
            for t in study.trials
        ]
    }

    # Save optimization results
    results_path = os.path.join(args.output_dir, 'bayesopt_results.json')
    with open(results_path, 'w') as f:
        json.dump(optimization_results, f, indent=2, cls=BayesOptEncoder)

    # Create figures for visualization if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        from optuna.visualization import plot_optimization_history, plot_param_importances, plot_contour, plot_slice

        # Set figure size
        plt.figure(figsize=(12, 8))

        # Plot optimization history
        fig = plot_optimization_history(study)
        fig.write_image(os.path.join(args.output_dir, 'optimization_history.png'))

        # Plot parameter importances
        fig = plot_param_importances(study)
        fig.write_image(os.path.join(args.output_dir, 'param_importances.png'))

        # Plot slice plot
        fig = plot_slice(study)
        fig.write_image(os.path.join(args.output_dir, 'slice_plot.png'))

        # Plot contour plot
        fig = plot_contour(study, params=['cutoff', 'num_blocks'])
        fig.write_image(os.path.join(args.output_dir, 'contour_plot.png'))
    except ImportError:
        logger.warning("Matplotlib or plotly not available for generating figures")
    except Exception as e:
        logger.warning(f"Error generating visualization: {e}")

    # Log summary
    logger.info(f"=== Bayesian Optimization Completed ===")
    logger.info(f"Total time: {total_time:.2f} seconds")
    logger.info(f"Number of trials: {len(study.trials)}")
    logger.info(f"Best parameters: {best_params}")
    logger.info(f"Best {args.metric_for_best} value: {best_value:.6f}")
    logger.info(f"Results saved to: {results_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())