#!/usr/bin/env python

import os
import sys
import time
import json
import numpy as np
import logging
import random
import shutil
import copy
import importlib.util
import hashlib
import argparse
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
from datetime import datetime

os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"

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

import ray
from ray import tune
from ray.tune.search import BasicVariantGenerator

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(script_dir))
deep_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, deep_dir)


class HyperoptEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.float32):
            return float(obj)
        if isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        if isinstance(obj, argparse.Namespace):
            return vars(obj)
        return json.JSONEncoder.default(self, obj)


def clean_processed_dir(root_dir):
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


def validate_paths(args, logger):
    """Validate that all necessary paths exist and are accessible."""
    # Check dataset root
    root_path = Path(args.reaction_dataset_root)
    if not root_path.exists():
        logger.error(f"Dataset root path does not exist: {root_path}")
        return False

    # Check CSV file
    csv_path = Path(args.dataset_csv)
    if not csv_path.exists():
        logger.error(f"Dataset CSV file does not exist: {csv_path}")
        return False

    # Check output directory
    output_path = Path(args.output_dir)
    try:
        os.makedirs(output_path, exist_ok=True)
        # Test write permissions
        test_file = output_path / 'test_write.txt'
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
    except Exception as e:
        logger.error(f"Cannot write to output directory {output_path}: {e}")
        return False

    return True


def train_fold(config, checkpoint_dir=None):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(os.path.dirname(script_dir))
    deep_dir = os.path.dirname(script_dir)

    sys.path.insert(0, parent_dir)
    sys.path.insert(0, deep_dir)
    sys.path.insert(0, script_dir)

    if torch.cuda.is_available() and ray.get_gpu_ids():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(ray.get_gpu_ids()[0])
        torch.cuda.set_device(0)

    args = copy.deepcopy(config["args"])
    fold_idx = config["fold_idx"]

    fold_dir = os.path.join(args.output_dir, f"fold_{fold_idx}")
    os.makedirs(fold_dir, exist_ok=True)

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

    log_file = os.path.join(fold_dir, 'training.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info(f"Starting training for fold {fold_idx}")
    logger.info(f"GPU IDs available: {ray.get_gpu_ids()}")

    # Validate all paths before proceeding
    if not validate_paths(args, logger):
        logger.error("Path validation failed, aborting")
        return {
            "error": "Path validation failed",
            "fold": fold_idx,
            "val_mae": 999.0,
            "val_rmse": 999.0,
            "test_mae": 999.0,
            "test_rmse": 999.0
        }

    # Set seeds for reproducibility
    pl.seed_everything(args.random_seed + fold_idx)

    # Check for necessary modules
    try:
        from data.load_Reaction import load_reaction
        logger.info("Successfully imported load_reaction")
    except ImportError as e:
        logger.error(f"Failed to import load_Reaction: {e}")
        return {
            "error": f"Import error: {e}",
            "fold": fold_idx,
            "val_mae": 999.0,
            "val_rmse": 999.0,
            "test_mae": 999.0,
            "test_rmse": 999.0
        }

    # Try to load reaction data with robust error handling
    fold_datasets = None
    try:
        # First attempt
        logger.info(f"Loading reaction data from {args.reaction_dataset_root}, CSV: {args.dataset_csv}")
        logger.info(f"Target fields: {args.reaction_target_fields}, Features: {args.input_features}")

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
    except Exception as e:
        logger.error(f"First attempt to load data failed: {e}")
        logger.error(traceback.format_exc())

        try:
            # Clean up and retry
            logger.info("Cleaning processed directory and trying again...")
            clean_processed_dir(args.reaction_dataset_root)

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
                cv_grouped=args.cv_grouped,
                force_reload=True
            )
        except Exception as e2:
            logger.error(f"Second attempt to load data failed: {e2}")
            logger.error(traceback.format_exc())
            return {
                "error": f"Data loading failed: {e2}",
                "fold": fold_idx,
                "val_mae": 999.0,
                "val_rmse": 999.0,
                "test_mae": 999.0,
                "test_rmse": 999.0
            }

    if fold_datasets is None:
        logger.error("Failed to load datasets even after retry")
        return {
            "error": "Dataset loading failed",
            "fold": fold_idx,
            "val_mae": 999.0,
            "val_rmse": 999.0,
            "test_mae": 999.0,
            "test_rmse": 999.0
        }

    if fold_idx >= len(fold_datasets):
        logger.error(f"Fold index {fold_idx} out of range, only {len(fold_datasets)} folds available")
        return {
            "error": f"Fold index {fold_idx} out of range",
            "fold": fold_idx,
            "val_mae": 999.0,
            "val_rmse": 999.0,
            "test_mae": 999.0,
            "test_rmse": 999.0
        }

    # Import and run the fold training
    try:
        from train import run_fold
        logger.info("Successfully imported run_fold")
    except ImportError as e:
        logger.error(f"Failed to import run_fold: {e}")
        return {
            "error": f"Import error: {e}",
            "fold": fold_idx,
            "val_mae": 999.0,
            "val_rmse": 999.0,
            "test_mae": 999.0,
            "test_rmse": 999.0
        }

    try:
        # Run the fold and get test metrics
        logger.info(f"Starting fold {fold_idx} training with {len(fold_datasets[fold_idx]['train'])} training samples")
        test_metrics = run_fold(fold_idx, fold_datasets[fold_idx], args, logger)

        # Process metrics from the fold run
        fold_metrics = {}

        # Extract metrics from test_metrics
        for key, value in test_metrics.items():
            if isinstance(value, (int, float, np.number)):
                fold_metrics[key] = float(value)

        # Find validation metrics if available
        val_metrics_path = os.path.join(fold_dir, 'val_metrics.json')
        fold_val_metrics = {}

        if os.path.exists(val_metrics_path):
            try:
                with open(val_metrics_path, 'r') as f:
                    fold_val_metrics = json.load(f)
                logger.info(f"Loaded validation metrics from {val_metrics_path}")
            except Exception as e:
                logger.warning(f"Could not load validation metrics from {val_metrics_path}: {e}")

        # Find metrics.json for additional information
        metrics_json_path = os.path.join(fold_dir, 'metrics.json')
        if os.path.exists(metrics_json_path):
            try:
                with open(metrics_json_path, 'r') as f:
                    metrics_json = json.load(f)
                logger.info(f"Loaded additional metrics from {metrics_json_path}")

                if 'test' in metrics_json:
                    for key, value in metrics_json['test'].items():
                        if key.startswith('Test') and isinstance(value, (int, float, np.number)):
                            fold_metrics[key] = float(value)

                if 'val' in metrics_json:
                    for key, value in metrics_json['val'].items():
                        if isinstance(value, (int, float, np.number)):
                            fold_val_metrics[key] = float(value)
            except Exception as e:
                logger.warning(f"Could not load additional metrics from {metrics_json_path}: {e}")

        # Get test and validation metrics
        # First check for Test Avg MAE directly
        test_mae = 0.0
        test_rmse = 0.0
        for key, value in fold_metrics.items():
            if key == 'Test Avg MAE':
                test_mae = value
            elif key == 'Test Avg RMSE':
                test_rmse = value

        # If not found, calculate from individual MAE/RMSE values
        if test_mae == 0.0:
            mae_values = [v for k, v in fold_metrics.items() if 'MAE' in k]
            if mae_values:
                test_mae = np.mean(mae_values)
                logger.info(f"Calculated test_mae = {test_mae} from {len(mae_values)} MAE values")

        if test_rmse == 0.0:
            rmse_values = [v for k, v in fold_metrics.items() if 'RMSE' in k]
            if rmse_values:
                test_rmse = np.mean(rmse_values)
                logger.info(f"Calculated test_rmse = {test_rmse} from {len(rmse_values)} RMSE values")

        # Get validation metrics
        val_mae = fold_val_metrics.get('val_mae', 0.0)
        val_rmse = fold_val_metrics.get('val_rmse', 0.0)

        # If validation metrics are still zero, use test metrics as proxy
        if val_mae == 0.0 and test_mae > 0:
            val_mae = test_mae
            logger.info(f"Using test_mae = {test_mae} as proxy for val_mae")

        if val_rmse == 0.0 and test_rmse > 0:
            val_rmse = test_rmse
            logger.info(f"Using test_rmse = {test_rmse} as proxy for val_rmse")

        result = {
            "fold": fold_idx,
            "val_mae": val_mae,
            "val_rmse": val_rmse,
            "test_mae": test_mae,
            "test_rmse": test_rmse,
            "test_metrics": fold_metrics,
            "val_metrics": fold_val_metrics
        }

        with open(os.path.join(fold_dir, 'fold_result.json'), 'w') as f:
            json.dump(result, f, indent=2, cls=HyperoptEncoder)

        logger.info(f"Fold {fold_idx} completed successfully")
        logger.info(f"Validation MAE: {val_mae:.6f}, Validation RMSE: {val_rmse:.6f}")
        logger.info(f"Test MAE: {test_mae:.6f}, Test RMSE: {test_rmse:.6f}")

        tune.report({
            "fold": fold_idx,
            "val_mae": val_mae,
            "val_rmse": val_rmse,
            "test_mae": test_mae,
            "test_rmse": test_rmse
        })

        return result

    except Exception as e:
        logger.error(f"Error in fold {fold_idx}: {e}")
        logger.error(traceback.format_exc())

        tune.report({
            "fold": fold_idx,
            "error": str(e),
            "val_mae": 999.0,
            "val_rmse": 999.0,
            "test_mae": 999.0,
            "test_rmse": 999.0
        })

        return {
            "fold": fold_idx,
            "error": str(e),
            "val_mae": 999.0,
            "val_rmse": 999.0,
            "test_mae": 999.0,
            "test_rmse": 999.0
        }


def main():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)

    from cli.config import process_args, save_config, setup_logging

    args = process_args()
    os.makedirs(args.output_dir, exist_ok=True)
    config_path = save_config(args, args.output_dir)

    logger = setup_logging(args)
    logger.info(f"Configuration saved to {config_path}")

    if args.cv_folds <= 0:
        logger.error("This script requires cv_folds > 0. Use the regular train.py for non-CV training.")
        return 1

    # Validate paths before proceeding
    if not validate_paths(args, logger):
        logger.error("Path validation failed, exiting")
        return 1

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    available_gpus = min(num_gpus, args.cv_folds)

    logger.info(f"Starting parallel cross-validation with {args.cv_folds} folds")
    logger.info(f"Found {num_gpus} GPUs, will use {available_gpus} for parallel training")

    # Clean processed data directory before starting
    logger.info("Cleaning processed data directory before starting training")
    clean_processed_dir(args.reaction_dataset_root)

    ray.init(num_cpus=args.cv_folds, num_gpus=num_gpus, include_dashboard=False)

    storage_path = os.path.join(args.output_dir, "ray_results")
    os.makedirs(storage_path, exist_ok=True)

    # Prepare config for the trials
    config = {
        "fold_idx": tune.grid_search(list(range(args.cv_folds))),
        "args": args  # This will be passed to all trials
    }

    # Run parallel training using Ray Tune
    start_time = time.time()

    analysis = tune.run(
        train_fold,
        config=config,
        resources_per_trial={"cpu": 1, "gpu": 1 if num_gpus > 0 else 0},
        metric="val_mae",
        mode="min",
        storage_path=storage_path,
        name=f"cv_parallel_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        verbose=1,
        fail_fast=False,
        num_samples=1
    )

    total_time = time.time() - start_time

    # Aggregate results from all folds
    all_fold_metrics = []
    val_mae_values = []
    val_rmse_values = []
    test_mae_values = []
    test_rmse_values = []

    # Process trial results
    for trial in analysis.trials:
        if trial.last_result:
            # Convert to dict if it's not already
            result = dict(trial.last_result)
            fold_idx = result.get("fold", -1)
            fold_error = result.get("error", None)

            if fold_error is None:
                # This trial completed successfully
                all_fold_metrics.append(result)

                # Extract metrics
                val_mae = result.get("val_mae", None)
                val_rmse = result.get("val_rmse", None)
                test_mae = result.get("test_mae", None)
                test_rmse = result.get("test_rmse", None)

                # Only add valid metric values (not None, not 999)
                if val_mae is not None and val_mae != 999.0:
                    val_mae_values.append(val_mae)
                if val_rmse is not None and val_rmse != 999.0:
                    val_rmse_values.append(val_rmse)
                if test_mae is not None and test_mae != 999.0:
                    test_mae_values.append(test_mae)
                if test_rmse is not None and test_rmse != 999.0:
                    test_rmse_values.append(test_rmse)

                logger.info(
                    f"Fold {fold_idx} metrics - val_mae: {val_mae}, val_rmse: {val_rmse}, test_mae: {test_mae}, test_rmse: {test_rmse}")
            else:
                logger.error(f"Fold {fold_idx} failed with error: {fold_error}")

    avg_val_mae = np.mean(val_mae_values) if val_mae_values else 0.0
    avg_val_rmse = np.mean(val_rmse_values) if val_rmse_values else 0.0
    avg_test_mae = np.mean(test_mae_values) if test_mae_values else 0.0
    avg_test_rmse = np.mean(test_rmse_values) if test_rmse_values else 0.0

    std_val_mae = np.std(val_mae_values) if len(val_mae_values) > 1 else 0.0
    std_val_rmse = np.std(val_rmse_values) if len(val_rmse_values) > 1 else 0.0
    std_test_mae = np.std(test_mae_values) if len(test_mae_values) > 1 else 0.0
    std_test_rmse = np.std(test_rmse_values) if len(test_rmse_values) > 1 else 0.0

    cv_summary = {
        'fold_metrics': all_fold_metrics,
        'avg_metrics': {
            'val_mae': avg_val_mae,
            'val_rmse': avg_val_rmse,
            'test_mae': avg_test_mae,
            'test_rmse': avg_test_rmse
        },
        'std_metrics': {
            'val_mae': std_val_mae,
            'val_rmse': std_val_rmse,
            'test_mae': std_test_mae,
            'test_rmse': std_test_rmse
        },
        'total_time': total_time,
        'folds_completed': len(all_fold_metrics),
        'total_folds': args.cv_folds
    }

    cv_summary_path = os.path.join(args.output_dir, 'cv_summary.json')
    with open(cv_summary_path, 'w') as f:
        json.dump(cv_summary, f, indent=2, cls=HyperoptEncoder)

    logger.info("=== Cross-Validation Summary ===")
    logger.info(f"Completed {len(all_fold_metrics)}/{args.cv_folds} folds in {total_time:.2f} seconds")
    logger.info(f"Average validation MAE: {avg_val_mae:.6f} ± {std_val_mae:.6f}")
    logger.info(f"Average validation RMSE: {avg_val_rmse:.6f} ± {std_val_rmse:.6f}")
    logger.info(f"Average test MAE: {avg_test_mae:.6f} ± {std_test_mae:.6f}")
    logger.info(f"Average test RMSE: {avg_test_rmse:.6f} ± {std_test_rmse:.6f}")
    logger.info(f"Cross-validation results saved to {cv_summary_path}")

    ray.shutdown()

    return 0


if __name__ == "__main__":
    sys.exit(main())