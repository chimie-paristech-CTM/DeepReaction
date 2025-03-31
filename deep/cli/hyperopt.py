#!/usr/bin/env python

import os
import sys
import time
import json
import numpy as np
import logging
import random
import itertools
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import concurrent.futures
import multiprocessing
from datetime import datetime
import argparse
import copy

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

# Add parent directory to path for imports
parent_path = str(Path(Path(__file__).parent.absolute()).parent.parent.absolute())
sys.path.insert(0, parent_path)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cli.config import (
    get_parser, process_args, save_config, setup_logging,
    get_experiment_config, get_model_name, print_args_summary
)

# Import load_reaction directly avoiding circular imports
from data.load_Reaction import load_reaction

# Import necessary functions from train.py
from train import (
    NumpyEncoder, setup_loggers, setup_callbacks,
    create_model, create_dataloader
)


class HyperoptEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.float32):
            return float(obj)
        if isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        return json.JSONEncoder.default(self, obj)


def parse_command_line_args():
    parser = get_parser()

    # Add hyperparameter search specific arguments
    hyperopt_group = parser.add_argument_group('Hyperparameter Optimization')
    hyperopt_group.add_argument('--cutoff_values', type=float, nargs='+', default=[5.0, 10.0, 15.0],
                                help='Cutoff distance values to search')
    hyperopt_group.add_argument('--num_blocks_values', type=int, nargs='+', default=[4, 5, 6],
                                help='Number of blocks values to search')
    hyperopt_group.add_argument('--prediction_hidden_layers_values', type=int, nargs='+', default=[3, 4, 5],
                                help='Prediction hidden layers values to search')
    hyperopt_group.add_argument('--prediction_hidden_dim_values', type=int, nargs='+', default=[128, 256, 512],
                                help='Prediction hidden dimension values to search')
    hyperopt_group.add_argument('--parallel_jobs', type=int, default=1,
                                help='Number of parallel jobs for hyperparameter search')
    hyperopt_group.add_argument('--metric_for_best', type=str, default='Validation Avg MAE',
                                help='Metric to use for selecting best hyperparameters')
    hyperopt_group.add_argument('--metric_mode', type=str, default='min', choices=['min', 'max'],
                                help='Whether to minimize or maximize the metric')

    args = process_args(parser)
    return args


def generate_hyperparameter_grid(args):
    """Generate grid of hyperparameters to search."""
    param_grid = {
        'cutoff': args.cutoff_values,
        'num_blocks': args.num_blocks_values,
        'prediction_hidden_layers': args.prediction_hidden_layers_values,
        'prediction_hidden_dim': args.prediction_hidden_dim_values
    }

    # Generate all combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))

    hyperparameter_sets = []
    for combo in combinations:
        param_set = {keys[i]: combo[i] for i in range(len(keys))}
        hyperparameter_sets.append(param_set)

    return hyperparameter_sets


def create_experiment_name(params):
    """Create a unique experiment name based on hyperparameters."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"hyperopt_cut{params['cutoff']}_blk{params['num_blocks']}_" \
           f"phl{params['prediction_hidden_layers']}_phd{params['prediction_hidden_dim']}_{timestamp}"
    return name


def run_single_fold(fold_idx, fold_data, args, logger):
    """Run a single fold of cross-validation with the given hyperparameters."""
    fold_dir = os.path.join(args.output_dir, f"fold_{fold_idx}")
    os.makedirs(fold_dir, exist_ok=True)

    logger.info(f"=== Processing fold {fold_idx + 1}/{args.cv_folds} ===")

    # Create dataloaders for this fold
    train_data = fold_data['train']
    val_data = fold_data['val']
    test_data = fold_data['test']
    scalers = fold_data['scalers']

    follow_batch = ['z0', 'z1', 'z2', 'pos0', 'pos1', 'pos2'] if args.dataset == 'XTB' else None
    dataloader_kwargs = {'follow_batch': follow_batch} if follow_batch else {}

    train_loader = create_dataloader(
        train_data, args.batch_size, eval_mode=False,
        num_workers=args.num_workers, **dataloader_kwargs
    )

    val_loader = create_dataloader(
        val_data, args.eval_batch_size, eval_mode=True,
        num_workers=args.num_workers, **dataloader_kwargs
    )

    test_loader = create_dataloader(
        test_data, args.eval_batch_size, eval_mode=True,
        num_workers=args.num_workers, **dataloader_kwargs
    )

    # Create model
    model = create_model(args, scalers)

    # Loggers and callbacks
    fold_loggers = setup_loggers(args, fold_dir)
    fold_callbacks = setup_callbacks(args, fold_dir)

    # Train the model
    from train import train_model, evaluate_model
    trainer, train_metrics = train_model(
        args, model, train_loader, val_loader, fold_loggers, fold_callbacks, logger
    )

    # Evaluate on test set
    test_metrics = evaluate_model(
        trainer, model, test_loader, fold_dir, args.save_predictions,
        args.reaction_target_fields, logger
    )

    # Get validation metrics
    val_metrics = {}
    for k, v in model.val_metrics.items():
        if isinstance(v, dict):
            for metric_name, metric_val in v.items():
                val_metrics[f"Validation {metric_name}"] = metric_val

    # Combine metrics
    fold_metrics = {
        'train': train_metrics,
        'validation': val_metrics,
        'test': test_metrics,
        'fold': fold_idx
    }

    # Save fold metrics
    with open(os.path.join(fold_dir, 'metrics.json'), 'w') as f:
        json.dump(fold_metrics, f, indent=2, cls=HyperoptEncoder)

    return fold_metrics


def run_cv_experiment(hyperparams, base_args, experiment_id=0):
    """Run a single CV experiment with the given hyperparameters."""
    # Create a deep copy of the base args to modify
    args = copy.deepcopy(base_args)

    # Update args with the hyperparameters to test
    for param, value in hyperparams.items():
        setattr(args, param, value)

    # Create a unique output directory for this experiment
    experiment_name = create_experiment_name(hyperparams)
    args.experiment_name = experiment_name
    args.output_dir = os.path.join(args.out_dir, experiment_name)
    os.makedirs(args.output_dir, exist_ok=True)

    # Set up logging
    logger = setup_logging(args)
    logger.info(f"Starting hyperparameter evaluation (ID: {experiment_id}): {hyperparams}")

    # Save configuration
    config_path = save_config(args, args.output_dir)
    logger.info(f"Configuration saved to {config_path}")

    # Set random seed
    pl.seed_everything(args.random_seed)

    # Load datasets for cross-validation
    logger.info(f"Loading dataset for {args.cv_folds}-fold CV")
    fold_datasets = load_reaction(
        args.random_seed,
        root=args.reaction_dataset_root,
        dataset_csv=args.dataset_csv,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        use_scaler=args.use_scaler,
        target_fields=args.reaction_target_fields,
        file_suffixes=args.reaction_file_suffixes,
        input_features=args.input_features,
        cv_folds=args.cv_folds,
        cv_test_fold=args.cv_test_fold,
        cv_stratify=args.cv_stratify,
        cv_grouped=args.cv_grouped
    )

    # Run cross-validation
    all_fold_metrics = []

    for fold_idx, fold_data in enumerate(fold_datasets):
        try:
            fold_metrics = run_single_fold(fold_idx, fold_data, args, logger)
            all_fold_metrics.append(fold_metrics)
        except Exception as e:
            logger.error(f"Error in fold {fold_idx}: {e}")
            import traceback
            logger.error(traceback.format_exc())

    # Calculate average metrics across folds
    avg_metrics = {}
    std_metrics = {}

    # Collect all metrics from different folds
    all_metrics = {}
    for fold_result in all_fold_metrics:
        for phase in ['validation', 'test']:
            if phase in fold_result:
                for metric_name, value in fold_result[phase].items():
                    if isinstance(value, (int, float, np.number)):
                        key = f"{phase}_{metric_name}"
                        if key not in all_metrics:
                            all_metrics[key] = []
                        all_metrics[key].append(float(value))

    # Calculate averages and standard deviations
    for key, values in all_metrics.items():
        if values:
            avg_metrics[key.replace('_', ' ')] = float(np.mean(values))
            std_metrics[key.replace('_', ' ').replace('Avg', 'Std')] = float(np.std(values))

    # Save summary
    result = {
        'hyperparams': hyperparams,
        'avg_metrics': avg_metrics,
        'std_metrics': std_metrics,
        'all_fold_metrics': all_fold_metrics
    }

    with open(os.path.join(args.output_dir, 'cv_summary.json'), 'w') as f:
        json.dump(result, f, indent=2, cls=HyperoptEncoder)

    logger.info(f"=== Experiment Summary for {hyperparams} ===")
    for metric, value in avg_metrics.items():
        std_key = metric.replace('Avg', 'Std')
        std_value = std_metrics.get(std_key, 0)
        logger.info(f"{metric}: {value:.6f} ± {std_value:.6f}")

    return result


def run_serial_experiments(param_grid, base_args, logger):
    """Run hyperparameter experiments in serial."""
    results = []
    for i, params in enumerate(param_grid):
        logger.info(f"Running experiment {i + 1}/{len(param_grid)}")
        try:
            result = run_cv_experiment(params, base_args, i)
            results.append(result)
        except Exception as e:
            logger.error(f"Error in experiment with params {params}: {e}")
            import traceback
            logger.error(traceback.format_exc())
    return results


def select_best_hyperparameters(results, metric_name, mode='min'):
    """Select the best hyperparameters based on the specified metric."""
    if not results:
        return None, None

    # Get metric values from all experiments
    metric_values = []
    for result in results:
        metric_val = result['avg_metrics'].get(metric_name, float('inf') if mode == 'min' else float('-inf'))
        metric_values.append(metric_val)

    # Find best experiment based on metric mode
    if mode == 'min':
        best_idx = np.argmin(metric_values)
    else:
        best_idx = np.argmax(metric_values)

    best_result = results[best_idx]
    best_hyperparams = best_result['hyperparams']
    best_metrics = best_result['avg_metrics']

    return best_hyperparams, best_metrics


def main():
    # Configure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)

    # Parse command line arguments
    args = parse_command_line_args()

    # Create main output directory
    os.makedirs(args.out_dir, exist_ok=True)

    # Setup logging
    logger = setup_logging(args)
    logger.info("Starting hyperparameter optimization")

    # Generate hyperparameter grid
    param_grid = generate_hyperparameter_grid(args)
    logger.info(f"Generated {len(param_grid)} hyperparameter combinations to evaluate")

    # Save hyperparameter grid
    with open(os.path.join(args.out_dir, 'hyperparameter_grid.json'), 'w') as f:
        json.dump(param_grid, f, indent=2, cls=HyperoptEncoder)

    start_time = time.time()

    # Run experiments serially
    logger.info("Running experiments serially")
    results = run_serial_experiments(param_grid, args, logger)

    total_time = time.time() - start_time

    # Select best hyperparameters
    best_hyperparams, best_metrics = select_best_hyperparameters(
        results, args.metric_for_best, args.metric_mode
    )

    # Save overall results
    overall_results = {
        'total_time': total_time,
        'num_combinations': len(param_grid),
        'best_hyperparams': best_hyperparams,
        'best_metrics': best_metrics,
        'all_results': results
    }

    with open(os.path.join(args.out_dir, 'hyperopt_results.json'), 'w') as f:
        json.dump(overall_results, f, indent=2, cls=HyperoptEncoder)

    # Print summary
    logger.info(f"=== Hyperparameter Optimization Completed ===")
    logger.info(f"Total time: {total_time:.2f} seconds")
    logger.info(f"Best hyperparameters: {best_hyperparams}")
    if best_metrics:
        for metric, value in best_metrics.items():
            logger.info(f"{metric}: {value:.6f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())