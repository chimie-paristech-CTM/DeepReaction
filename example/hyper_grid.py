#!/usr/bin/env python
import os
import sys
import argparse
import json
import yaml
import numpy as np
import pandas as pd
import time
import itertools
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime
from pathlib import Path
import ray
from ray import tune
from ray.tune.search.optuna import OptunaSearch
import optuna
from optuna.samplers import TPESampler
from optuna.visualization import plot_param_importances, plot_optimization_history
import logging
import io
import contextlib
import tqdm

from deepreaction import ReactionDataset, ReactionTrainer
from deepreaction.config import ReactionConfig, ModelConfig, TrainingConfig, Config, save_config

def get_parser():
    parser = argparse.ArgumentParser(description='Hyperparameter optimization for molecular reaction prediction model')
    
    # Basic configuration
    parser.add_argument('--method', type=str, default='bayesian', choices=['grid', 'bayesian', 'random'], 
                        help='Hyperparameter optimization method')
    parser.add_argument('--n_trials', type=int, default=5, help='Number of trials for Bayesian optimization')
    parser.add_argument('--out_dir', type=str, default='./results/hyperopt', help='Output directory')
    parser.add_argument('--config_file', type=str, default=None, help='Path to hyperparameter search space config file')
    parser.add_argument('--dataset_root', type=str, default='./dataset/DATASET_DA_F', help='Dataset root directory')
    parser.add_argument('--dataset_csv', type=str, default='./dataset/DATASET_DA_F/dataset_xtb_final.csv', help='Dataset CSV file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Dataset parameters
    parser.add_argument('--target_fields', type=str, nargs='+', default=['G(TS)', 'DrG'], help='Target fields')
    parser.add_argument('--input_features', type=str, nargs='+', default=['G(TS)_xtb', 'DrG_xtb'], help='Input features')
    parser.add_argument('--file_patterns', type=str, nargs='+', default=['*_reactant.xyz', '*_ts.xyz', '*_product.xyz'], help='File patterns')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Validation set ratio')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='Test set ratio')
    parser.add_argument('--cv_folds', type=int, default=0, help='Number of cross-validation folds (0 for no CV)')
    
    # Parallelization parameters
    parser.add_argument('--num_gpus', type=int, default=None, help='Number of GPUs to use (default: all available)')
    parser.add_argument('--num_cpus', type=int, default=None, help='Number of CPUs to use (default: all available)')
    parser.add_argument('--gpus_per_trial', type=float, default=1, help='Number of GPUs per trial')
    parser.add_argument('--cpus_per_trial', type=float, default=2, help='Number of CPUs per trial')
    parser.add_argument('--parallel_trials', type=int, default=None, 
                        help='Number of trials to run in parallel (default: max based on available resources)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=1, help='Maximum number of epochs per trial')
    parser.add_argument('--batch_size', type=int, default=16, help='Default batch size if not in search space')
    parser.add_argument('--early_stopping', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    
    # Logging parameters
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--log_level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                      help='Logging level')
    
    return parser

def setup_logger(log_file, log_level=logging.INFO):
    """
    Set up a logger that writes to both file and console
    
    Args:
        log_file: Path to the log file
        log_level: Logging level (default: INFO)
        
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger(os.path.basename(log_file))
    logger.setLevel(log_level)
    logger.handlers = []  # Clear existing handlers
    
    # Create file handler
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    
    # Create formatter and add it to handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    
    return logger

def load_search_space(config_file=None):
    """
    Load hyperparameter search space from config file or return default search space.
    
    Args:
        config_file: Path to YAML or JSON configuration file defining search space
        
    Returns:
        Dictionary with hyperparameter search space definitions
    """
    default_search_space = {
        "model_type": {"values": ["dimenet++"]},
        "readout": {"values": ["sum", "mean", "attention"]},
        "batch_size": {"values": [4]},
        "learning_rate": {"range": [0.0001, 0.001], "log": True},
        "dropout": {"range": [0.0, 0.5]},
        "hidden_channels": {"values": [64, 128, 256]},
        "num_blocks": {"values": [3, 4, 5]},
        "node_dim": {"values": [64, 128, 256]},
        "prediction_hidden_dim": {"values": [128, 256, 512]},
        "weight_decay": {"range": [1e-5, 1e-3], "log": True},
        "optimizer": {"values": ["adam", "adamw"]},
    }
    
    if config_file is None:
        return default_search_space
    
    # Load from file
    file_ext = os.path.splitext(config_file)[1].lower()
    with open(config_file, 'r') as f:
        if file_ext in ['.yaml', '.yml']:
            search_space = yaml.safe_load(f)
        elif file_ext == '.json':
            search_space = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {file_ext}")
    
    return search_space

def create_optuna_space(search_space):
    """
    Create an Optuna parameter space from search space configuration.
    
    Args:
        search_space: Dictionary with hyperparameter search space definitions
        
    Returns:
        Optuna parameter creation function
    """
    def optuna_parameters(trial):
        params = {}
        
        for param_name, param_config in search_space.items():
            if "values" in param_config:
                params[param_name] = trial.suggest_categorical(param_name, param_config["values"])
            elif "range" in param_config:
                low, high = param_config["range"]
                if param_config.get("log", False):
                    params[param_name] = trial.suggest_float(param_name, low, high, log=True)
                elif param_config.get("int", False) or all(isinstance(x, int) for x in param_config["range"]):
                    params[param_name] = trial.suggest_int(param_name, low, high)
                else:
                    params[param_name] = trial.suggest_float(param_name, low, high)
        
        return params
    
    return optuna_parameters

def create_grid_space(search_space):
    """
    Create a grid search parameter space from search space configuration.
    
    Args:
        search_space: Dictionary with hyperparameter search space definitions
        
    Returns:
        List of parameter dictionaries for grid search
    """
    grid_params = {}
    
    for param_name, param_config in search_space.items():
        if "values" in param_config:
            grid_params[param_name] = param_config["values"]
        elif "range" in param_config:
            low, high = param_config["range"]
            if param_config.get("int", False) or all(isinstance(x, int) for x in param_config["range"]):
                # For integer parameters, create a discrete range
                num_points = param_config.get("num_points", 3)
                grid_params[param_name] = np.linspace(low, high, num_points, dtype=int).tolist()
            elif param_config.get("log", False):
                # For log-scale parameters, create a log-spaced range
                num_points = param_config.get("num_points", 5)
                grid_params[param_name] = np.logspace(np.log10(low), np.log10(high), num_points).tolist()
            else:
                # For continuous parameters, create a linear range
                num_points = param_config.get("num_points", 5)
                grid_params[param_name] = np.linspace(low, high, num_points).tolist()
    
    # Create all combinations
    param_names = list(grid_params.keys())
    param_values = list(grid_params.values())
    combinations = list(itertools.product(*param_values))
    
    return [dict(zip(param_names, combo)) for combo in combinations]

@ray.remote(num_gpus=1)
def train_and_evaluate_trial(trial_id, params, args_dict, dataset=None, fold_idx=None):
    """
    Train and evaluate a model with given hyperparameters.
    
    Args:
        trial_id: ID of the trial
        params: Hyperparameters for this trial
        args_dict: Dictionary of command line arguments
        dataset: Optional dataset object
        fold_idx: Optional fold index for cross-validation
        
    Returns:
        Dictionary with evaluation results
    """
    # Create trial output directory
    trial_dir = os.path.join(args_dict["out_dir"], f"trial_{trial_id}")
    os.makedirs(trial_dir, exist_ok=True)
    
    # Set up logging to file
    log_file = os.path.join(trial_dir, "trial.log")
    logger = setup_logger(log_file, getattr(logging, args_dict.get("log_level", "INFO")))
    
    # Redirect stdout and stderr to the log file
    log_stream = io.StringIO()
    sys.stdout = log_stream
    sys.stderr = log_stream
    
    # Save progress updates to the log file periodically
    def log_writer():
        while True:
            time.sleep(5)  # Write logs every 5 seconds
            with open(log_file, 'a') as f:
                f.write(log_stream.getvalue())
            log_stream.truncate(0)
            log_stream.seek(0)
    
    # Start background thread to write logs
    import threading
    log_thread = threading.Thread(target=log_writer, daemon=True)
    log_thread.start()
    
    try:
        # Log trial start
        logger.info(f"Starting trial {trial_id} with parameters: {params}")
        
        # Set up environment
        gpu_id = ray.get_gpu_ids()[0] if ray.get_gpu_ids() else 0
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
        logger.info(f"Trial {trial_id}: Using device {device} (GPU {gpu_id})")
        
        # Save hyperparameters
        with open(os.path.join(trial_dir, "params.json"), "w") as f:
            json.dump(params, f, indent=2)
        
        # Handle dataset loading or reuse provided dataset
        if dataset is None:
            # Load dataset
            logger.info(f"Loading dataset from {args_dict['dataset_root']}")
            dataset = ReactionDataset(
                root=args_dict["dataset_root"],
                csv_file=args_dict["dataset_csv"],
                target_fields=args_dict["target_fields"],
                file_patterns=args_dict["file_patterns"],
                input_features=args_dict["input_features"],
                use_scaler=True,
                random_seed=args_dict["seed"],
                train_ratio=args_dict["train_ratio"],
                val_ratio=args_dict["val_ratio"],
                test_ratio=args_dict["test_ratio"],
                cv_folds=args_dict["cv_folds"],
            )
        
        # Set fold if cross-validation is being used
        if fold_idx is not None and args_dict["cv_folds"] > 0:
            dataset.set_fold(fold_idx)
            logger.info(f"Using fold {fold_idx}")
        
        # Get dataset stats
        data_stats = dataset.get_data_stats()
        logger.info(f"Dataset stats: Train: {data_stats['train_size']}, Validation: {data_stats['val_size']}, Test: {data_stats['test_size']}")
        
        # Create trainer
        trainer_args = {
            "model_type": params.get("model_type", "dimenet++"),
            "readout": params.get("readout", "mean"),
            "batch_size": params.get("batch_size", args_dict["batch_size"]),
            "max_epochs": args_dict["epochs"],
            "learning_rate": params.get("learning_rate", 0.0005),
            "output_dir": trial_dir,
            "early_stopping_patience": args_dict["early_stopping"],
            "save_best_model": True,
            "save_last_model": False,
            "random_seed": args_dict["seed"],
            "num_targets": len(args_dict["target_fields"]),
            "use_scaler": True,
            "scalers": dataset.get_scalers(),
            "optimizer": params.get("optimizer", "adamw"),
            "weight_decay": params.get("weight_decay", 0.0001),
            "scheduler": params.get("scheduler", "cosine"),
            "gpu": torch.cuda.is_available(),
            "node_dim": params.get("node_dim", 128),
            "dropout": params.get("dropout", 0.1),
            "use_layer_norm": params.get("use_layer_norm", False),
            "target_field_names": args_dict["target_fields"],
            "use_xtb_features": len(args_dict["input_features"]) > 0,
            "num_xtb_features": len(args_dict["input_features"]),
            "prediction_hidden_layers": params.get("prediction_hidden_layers", 3),
            "prediction_hidden_dim": params.get("prediction_hidden_dim", 512),
            "num_workers": args_dict["num_workers"],
        }
        
        # Add model-specific parameters
        for key in ["hidden_channels", "num_blocks", "int_emb_size", "basis_emb_size", 
                    "out_emb_channels", "num_spherical", "num_radial", "cutoff",
                    "envelope_exponent", "num_before_skip", "num_after_skip", 
                    "num_output_layers", "max_num_neighbors"]:
            if key in params:
                trainer_args[key] = params[key]
        
        logger.info(f"Creating trainer with arguments: {trainer_args}")
        trainer = ReactionTrainer(**trainer_args)
        
        # Train model
        start_time = time.time()
        logger.info("Starting model training")
        
        # Suppress PyTorch Lightning logs to our log file
        import pytorch_lightning as pl
        pl.utilities.rank_zero_only.rank_zero_only = lambda fn: lambda *args, **kwargs: None
        
        # Train the model
        train_metrics = trainer.fit(
            train_dataset=dataset.train_data,
            val_dataset=dataset.val_data,
            test_dataset=dataset.test_data,
        )
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Extract validation metrics
        val_metrics = {}
        if hasattr(trainer, "val_metrics") and trainer.val_metrics:
            for epoch, metrics in trainer.val_metrics.items():
                for target, target_metrics in metrics.items():
                    for metric_name, value in target_metrics.items():
                        val_metrics[f"{target}_{metric_name}"] = value
        
        # Log best validation metrics
        if val_metrics:
            logger.info(f"Validation metrics: {val_metrics}")
        
        # Evaluate on test set
        test_metrics = None
        if trainer.trainer is not None and dataset.test_data is not None and len(dataset.test_data) > 0:
            from torch_geometric.loader import DataLoader
            follow_batch = ['z0', 'z1', 'z2', 'pos0', 'pos1', 'pos2']
            test_loader = DataLoader(
                dataset.test_data,
                batch_size=trainer_args["batch_size"],
                shuffle=False,
                num_workers=args_dict["num_workers"],
                follow_batch=follow_batch
            )
            
            logger.info("Evaluating model on test set")
            test_results = trainer.trainer.test(trainer.lightning_model, test_loader)
            if test_results and isinstance(test_results, list) and len(test_results) > 0:
                test_metrics = test_results[0]
                logger.info(f"Test metrics: {test_metrics}")
        
        # Get best validation loss and epoch
        best_val_loss = float('inf')
        best_epoch = -1
        if hasattr(trainer.trainer, 'callback_metrics'):
            for epoch, metrics in trainer.trainer.callback_metrics.items():
                if 'val_total_loss' in metrics and metrics['val_total_loss'] < best_val_loss:
                    best_val_loss = metrics['val_total_loss']
                    best_epoch = epoch
        
        # Prepare result
        result = {
            "trial_id": trial_id,
            "params": params,
            "training_time": training_time,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "best_model_path": train_metrics.get("best_model_path"),
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
        }
        
        # Save result
        with open(os.path.join(trial_dir, "result.json"), "w") as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"Trial {trial_id} completed with best validation loss: {best_val_loss:.6f}")
        
        # Make sure all logs are written
        with open(log_file, 'a') as f:
            f.write(log_stream.getvalue())
        
        # Free up GPU memory
        del trainer
        torch.cuda.empty_cache()
        
        return result
    
    except Exception as e:
        logger.error(f"Error in trial {trial_id}: {str(e)}", exc_info=True)
        
        # Make sure all logs are written
        with open(log_file, 'a') as f:
            f.write(log_stream.getvalue())
            f.write(f"\nERROR: {str(e)}\n")
            import traceback
            f.write(traceback.format_exc())
        
        # Return error result
        return {
            "trial_id": trial_id,
            "params": params,
            "error": str(e),
            "status": "ERROR"
        }
    
    finally:
        # Restore stdout and stderr
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

def run_bayesian_optimization(search_space, args_dict, dataset=None):
    """
    Run Bayesian hyperparameter optimization using Optuna.
    
    Args:
        search_space: Dictionary with hyperparameter search space definitions
        args_dict: Dictionary of command line arguments
        dataset: Optional dataset object to use for all trials
        
    Returns:
        Dictionary with optimization results
    """
    print("Starting Bayesian hyperparameter optimization with Optuna")
    
    # Create Optuna study
    study_name = f"reaction_model_bayesian_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        sampler=TPESampler(seed=args_dict["seed"]),
    )
    
    # Create parameter space
    optuna_params = create_optuna_space(search_space)
    
    # Determine number of parallel trials
    if args_dict["parallel_trials"] is None:
        num_gpus = ray.available_resources().get("GPU", 1) if ray.is_initialized() else 1
        args_dict["parallel_trials"] = max(1, int(num_gpus / args_dict["gpus_per_trial"]))
    
    print(f"Running with {args_dict['parallel_trials']} parallel trials")
    
    # Set up the main log file
    main_log_file = os.path.join(args_dict["out_dir"], "optimization.log")
    main_logger = setup_logger(main_log_file, getattr(logging, args_dict.get("log_level", "INFO")))
    main_logger.info(f"Starting Bayesian optimization with {args_dict['n_trials']} trials")
    main_logger.info(f"Using {args_dict['parallel_trials']} parallel trials")
    main_logger.info(f"Search space: {search_space}")
    
    # Handle cross-validation
    if args_dict["cv_folds"] > 0 and dataset is not None:
        num_folds = dataset.get_num_folds()
        main_logger.info(f"Using {num_folds}-fold cross-validation")
        
        def objective(trial):
            # Get parameters for this trial
            params = optuna_params(trial)
            trial_id = trial.number
            
            main_logger.info(f"Starting trial {trial_id} with parameters: {params}")
            
            # Run cross-validation
            cv_results = []
            for fold_idx in range(num_folds):
                dataset.set_fold(fold_idx)
                fold_result = train_and_evaluate_trial.remote(
                    trial_id=f"{trial_id}_fold_{fold_idx}",
                    params=params,
                    args_dict=args_dict,
                    dataset=dataset,
                    fold_idx=fold_idx,
                )
                cv_results.append(fold_result)
            
            # Get results
            print(f"Waiting for trial {trial_id} CV folds to complete...")
            cv_results = ray.get(cv_results)
            
            # Compute average validation loss
            avg_val_loss = np.mean([r["best_val_loss"] for r in cv_results if "best_val_loss" in r])
            
            # Save trial results
            trial_dir = os.path.join(args_dict["out_dir"], f"trial_{trial_id}")
            os.makedirs(trial_dir, exist_ok=True)
            with open(os.path.join(trial_dir, "cv_results.json"), "w") as f:
                json.dump(cv_results, f, indent=2)
            
            main_logger.info(f"Trial {trial_id} completed with avg val loss: {avg_val_loss:.6f}")
            print(f"Trial {trial_id}: {params}")
            print(f"Average validation loss: {avg_val_loss:.6f}")
            
            return avg_val_loss
    else:
        # No cross-validation
        def objective(trial):
            # Get parameters for this trial
            params = optuna_params(trial)
            trial_id = trial.number
            
            main_logger.info(f"Starting trial {trial_id} with parameters: {params}")
            
            # Run single training
            result_future = train_and_evaluate_trial.remote(
                trial_id=trial_id,
                params=params,
                args_dict=args_dict,
                dataset=dataset,
            )
            
            # Get result
            print(f"Waiting for trial {trial_id} to complete...")
            result = ray.get(result_future)
            
            if "error" in result:
                main_logger.error(f"Trial {trial_id} failed with error: {result['error']}")
                # Return a high loss value to discourage this region
                return float('inf')
            
            main_logger.info(f"Trial {trial_id} completed with val loss: {result['best_val_loss']:.6f}")
            print(f"Trial {trial_id}: {params}")
            print(f"Validation loss: {result['best_val_loss']:.6f}")
            
            return result["best_val_loss"]
    
    # Create progress bar
    pbar = tqdm.tqdm(total=args_dict["n_trials"], desc="Optimization progress")
    
    # Create callback to update progress bar
    def callback(study, trial):
        pbar.update(1)
        
        # Log best trial so far
        best_trial = study.best_trial
        main_logger.info(f"Current best: Trial {best_trial.number} with value {best_trial.value:.6f}")
        print(f"\nCurrent best: Trial {best_trial.number} with value {best_trial.value:.6f}")
        print(f"Parameters: {best_trial.params}")
    
    # Run optimization
    try:
        study.optimize(objective, n_trials=args_dict["n_trials"], n_jobs=args_dict["parallel_trials"], callbacks=[callback])
    except KeyboardInterrupt:
        main_logger.info("Optimization stopped manually")
        print("Optimization stopped manually")
    finally:
        pbar.close()
    
    # Get results
    best_trial = study.best_trial
    best_params = best_trial.params
    best_value = best_trial.value
    
    main_logger.info(f"Optimization completed. Best trial: {best_trial.number} with value {best_value:.6f}")
    main_logger.info(f"Best parameters: {best_params}")
    
    print("\n" + "="*50)
    print(f"Optimization completed!")
    print(f"Best trial: {best_trial.number} with validation loss {best_value:.6f}")
    print(f"Best parameters: {best_params}")
    print("="*50)
    
    # Save results
    study_results = {
        "best_params": best_params,
        "best_value": best_value,
        "best_trial": best_trial.number,
        "n_trials": len(study.trials),
        "study_name": study_name,
    }
    
    results_path = os.path.join(args_dict["out_dir"], "bayesian_results.json")
    with open(results_path, "w") as f:
        json.dump(study_results, f, indent=2)
    
    # Create result visualizations
    try:
        # Parameter importance plot
        fig1 = plot_param_importances(study)
        fig1.write_image(os.path.join(args_dict["out_dir"], "param_importances.png"))
        
        # Optimization history plot
        fig2 = plot_optimization_history(study)
        fig2.write_image(os.path.join(args_dict["out_dir"], "optimization_history.png"))
    except Exception as e:
        main_logger.error(f"Error creating visualizations: {e}")
        print(f"Error creating visualizations: {e}")
    
    return study_results

def run_grid_search(search_space, args_dict, dataset=None):
    """
    Run grid search hyperparameter optimization.
    
    Args:
        search_space: Dictionary with hyperparameter search space definitions
        args_dict: Dictionary of command line arguments
        dataset: Optional dataset object to use for all trials
        
    Returns:
        Dictionary with optimization results
    """
    # Create grid parameter combinations
    grid_params = create_grid_space(search_space)
    print(f"Starting grid search with {len(grid_params)} parameter combinations")
    
    # Set up the main log file
    main_log_file = os.path.join(args_dict["out_dir"], "optimization.log")
    main_logger = setup_logger(main_log_file, getattr(logging, args_dict.get("log_level", "INFO")))
    main_logger.info(f"Starting grid search with {len(grid_params)} combinations")
    
    # Determine number of parallel trials
    if args_dict["parallel_trials"] is None:
        num_gpus = ray.available_resources().get("GPU", 1) if ray.is_initialized() else 1
        args_dict["parallel_trials"] = max(1, int(num_gpus / args_dict["gpus_per_trial"]))
    
    main_logger.info(f"Running with {args_dict['parallel_trials']} parallel trials")
    print(f"Running with {args_dict['parallel_trials']} parallel trials")
    
    # Handle cross-validation
    if args_dict["cv_folds"] > 0 and dataset is not None:
        num_folds = dataset.get_num_folds()
        main_logger.info(f"Using {num_folds}-fold cross-validation")
        
        # Initialize tasks and results
        tasks = []
        for trial_id, params in enumerate(grid_params):
            main_logger.info(f"Scheduling trial {trial_id} with parameters: {params}")
            for fold_idx in range(num_folds):
                task = train_and_evaluate_trial.remote(
                    trial_id=f"{trial_id}_fold_{fold_idx}",
                    params=params,
                    args_dict=args_dict,
                    dataset=dataset,
                    fold_idx=fold_idx,
                )
                tasks.append((trial_id, fold_idx, task))
        
        # Create progress bar
        pbar = tqdm.tqdm(total=len(tasks), desc="Grid search progress")
        
        # Process results as they complete
        all_results = []
        pending_tasks = list(tasks)
        while pending_tasks:
            # Wait for a task to complete
            done_id, pending_ids = ray.wait([task for _, _, task in pending_tasks], num_returns=1)
            
            # Find the completed task
            for i, (trial_id, fold_idx, task) in enumerate(pending_tasks):
                if task in done_id:
                    # Get the result
                    result = ray.get(task)
                    all_results.append((trial_id, fold_idx, result))
                    
                    # Remove from pending tasks
                    pending_tasks.pop(i)
                    
                    # Update progress bar and log
                    pbar.update(1)
                    if "best_val_loss" in result:
                        main_logger.info(f"Trial {trial_id} fold {fold_idx} completed with val loss: {result['best_val_loss']:.6f}")
                        print(f"\nTrial {trial_id} fold {fold_idx}: val loss {result['best_val_loss']:.6f}")
                    else:
                        main_logger.warning(f"Trial {trial_id} fold {fold_idx} completed with error: {result.get('error', 'unknown error')}")
                        print(f"\nTrial {trial_id} fold {fold_idx} failed!")
                    
                    break
        
        pbar.close()
        
        # Group results by trial
        trial_results = {}
        for trial_id, fold_idx, result in all_results:
            if trial_id not in trial_results:
                trial_results[trial_id] = []
            trial_results[trial_id].append(result)
        
        # Compute average metrics for each trial
        grid_results = []
        for trial_id, results in trial_results.items():
            valid_results = [r for r in results if "best_val_loss" in r]
            if valid_results:
                avg_val_loss = np.mean([r["best_val_loss"] for r in valid_results])
                grid_results.append({
                    "trial_id": trial_id,
                    "params": grid_params[trial_id],
                    "avg_val_loss": avg_val_loss,
                    "fold_results": results,
                })
                main_logger.info(f"Trial {trial_id} average val loss: {avg_val_loss:.6f}")
                print(f"Trial {trial_id} average val loss: {avg_val_loss:.6f}")
            else:
                main_logger.warning(f"Trial {trial_id} has no valid results")
                print(f"Trial {trial_id} has no valid results")
    else:
        # No cross-validation
        tasks = []
        for trial_id, params in enumerate(grid_params):
            main_logger.info(f"Scheduling trial {trial_id} with parameters: {params}")
            task = train_and_evaluate_trial.remote(
                trial_id=trial_id,
                params=params,
                args_dict=args_dict,
                dataset=dataset,
            )
            tasks.append((trial_id, task))
        
        # Create progress bar
        pbar = tqdm.tqdm(total=len(tasks), desc="Grid search progress")
        
        # Process results as they complete
        grid_results = []
        pending_tasks = list(tasks)
        while pending_tasks:
            # Wait for a task to complete
            done_id, pending_ids = ray.wait([task for _, task in pending_tasks], num_returns=1)
            
            # Find the completed task
            for i, (trial_id, task) in enumerate(pending_tasks):
                if task in done_id:
                    # Get the result
                    result = ray.get(task)
                    result["trial_id"] = trial_id
                    result["params"] = grid_params[trial_id]
                    grid_results.append(result)
                    
                    # Remove from pending tasks
                    pending_tasks.pop(i)
                    
                    # Update progress bar and log
                    pbar.update(1)
                    if "best_val_loss" in result:
                        main_logger.info(f"Trial {trial_id} completed with val loss: {result['best_val_loss']:.6f}")
                        print(f"\nTrial {trial_id}: val loss {result['best_val_loss']:.6f}")
                    else:
                        main_logger.warning(f"Trial {trial_id} completed with error: {result.get('error', 'unknown error')}")
                        print(f"\nTrial {trial_id} failed!")
                    
                    break
        
        pbar.close()
    
    # Sort results by validation loss
    grid_results = sorted(grid_results, key=lambda x: x.get("avg_val_loss", x.get("best_val_loss", float("inf"))))
    
    # Extract best parameters
    if grid_results:
        best_result = grid_results[0]
        best_params = best_result["params"]
        best_value = best_result.get("avg_val_loss", best_result.get("best_val_loss"))
        
        main_logger.info(f"Grid search completed. Best trial: {best_result['trial_id']} with value {best_value:.6f}")
        main_logger.info(f"Best parameters: {best_params}")
        
        print("\n" + "="*50)
        print(f"Grid search completed!")
        print(f"Best trial: {best_result['trial_id']} with validation loss {best_value:.6f}")
        print(f"Best parameters: {best_params}")
        print("="*50)
        
        # Save results
        results = {
            "best_params": best_params,
            "best_value": best_value,
            "best_trial": best_result["trial_id"],
            "n_trials": len(grid_params),
            "all_results": grid_results,
        }
    else:
        main_logger.error("No valid results found in grid search")
        print("No valid results found in grid search")
        
        # Save empty results
        results = {
            "best_params": None,
            "best_value": None,
            "best_trial": None,
            "n_trials": len(grid_params),
            "all_results": [],
        }
    
    results_path = os.path.join(args_dict["out_dir"], "grid_search_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    return results

def visualize_results(results, method, out_dir):
    """
    Create visualizations of hyperparameter optimization results.
    
    Args:
        results: Results dictionary from optimization
        method: Optimization method ('grid' or 'bayesian')
        out_dir: Output directory for visualizations
    """
    print("Creating result visualizations")
    
    # Prepare visualization directory
    vis_dir = os.path.join(out_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    try:
        # Extract parameter names and create summary DataFrame
        if method == "grid" and "all_results" in results and results["all_results"]:
            data = []
            for result in results["all_results"]:
                if "params" not in result or "best_val_loss" not in result and "avg_val_loss" not in result:
                    continue
                
                row = result["params"].copy()
                row["best_val_loss"] = result.get("avg_val_loss", result.get("best_val_loss"))
                row["training_time"] = result.get("training_time", 0)
                data.append(row)
            
            if not data:
                print("No valid results to visualize")
                return
            
            df = pd.DataFrame(data)
            
            # Save summary CSV
            df.to_csv(os.path.join(vis_dir, "grid_search_summary.csv"), index=False)
            
            # Plot validation loss distribution
            plt.figure(figsize=(10, 6))
            sns.histplot(df["best_val_loss"], kde=True)
            plt.title("Distribution of Validation Loss")
            plt.xlabel("Validation Loss")
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, "loss_distribution.png"))
            plt.close()
            
            # Plot correlation between parameters and validation loss
            plt.figure(figsize=(12, 10))
            corr = df.corr()
            sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)
            plt.title("Correlation between Parameters and Validation Loss")
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, "parameter_correlation.png"))
            plt.close()
            
            # Create pairplots for most important parameters
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            if len(numeric_cols) > 2:
                # Choose parameters most correlated with validation loss
                loss_corr = abs(corr["best_val_loss"]).sort_values(ascending=False)
                top_params = loss_corr.index[1:min(5, len(loss_corr))].tolist()  # Skip the first one which is best_val_loss itself
                
                plot_cols = top_params + ["best_val_loss"]
                plt.figure(figsize=(15, 15))
                sns.pairplot(df[plot_cols], diag_kind="kde", plot_kws={"alpha": 0.6})
                plt.suptitle("Pairwise Relationships between Top Parameters and Validation Loss", y=1.02)
                plt.tight_layout()
                plt.savefig(os.path.join(vis_dir, "parameter_pairplot.png"))
                plt.close()
    except Exception as e:
        print(f"Error creating visualizations: {e}")

def main():
    parser = get_parser()
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Configure logging
    log_level = getattr(logging, args.log_level)
    main_log_file = os.path.join(args.out_dir, "main.log")
    main_logger = setup_logger(main_log_file, log_level)
    
    # Log start of execution
    main_logger.info(f"Starting hyperparameter optimization with method: {args.method}")
    main_logger.info(f"Arguments: {vars(args)}")
    
    # Save command line arguments
    args_dict = vars(args)
    with open(os.path.join(args.out_dir, "args.json"), "w") as f:
        json.dump(args_dict, f, indent=2)
    
    # Load hyperparameter search space
    search_space = load_search_space(args.config_file)
    main_logger.info(f"Search space: {search_space}")
    
    # Initialize Ray
    ray_resources = {}
    if args.num_gpus is not None:
        ray_resources["num_gpus"] = args.num_gpus
    if args.num_cpus is not None:
        ray_resources["num_cpus"] = args.num_cpus
    
    ray.init(**ray_resources)
    ray_resources = ray.available_resources()
    main_logger.info(f"Ray initialized with resources: {ray_resources}")
    print(f"Ray initialized with resources: {ray_resources}")
    
    # Save hyperparameter search space
    with open(os.path.join(args.out_dir, "search_space.json"), "w") as f:
        json.dump(search_space, f, indent=2)
    
    # Load dataset once if it will be reused across trials
    dataset = None
    if args.cv_folds > 0:
        main_logger.info(f"Loading dataset from {args.dataset_root}")
        print(f"Loading dataset from {args.dataset_root}")
        dataset = ReactionDataset(
            root=args.dataset_root,
            csv_file=args.dataset_csv,
            target_fields=args.target_fields,
            file_patterns=args.file_patterns,
            input_features=args.input_features,
            use_scaler=True,
            random_seed=args.seed,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            cv_folds=args.cv_folds,
        )
        main_logger.info(f"Dataset loaded with {args.cv_folds} folds")
        print(f"Dataset loaded with {args.cv_folds} folds")
    
    # Run hyperparameter optimization
    start_time = time.time()
    main_logger.info("Starting optimization")
    
    try:
        if args.method == "bayesian":
            results = run_bayesian_optimization(search_space, args_dict, dataset)
        elif args.method == "grid":
            results = run_grid_search(search_space, args_dict, dataset)
        elif args.method == "random":
            # Random search is just Bayesian with a different sampler
            args_dict["n_trials"] = len(create_grid_space(search_space)) if args.n_trials is None else args.n_trials
            args_dict["sampler"] = "random"
            results = run_bayesian_optimization(search_space, args_dict, dataset)
    except Exception as e:
        main_logger.error(f"Error during optimization: {str(e)}", exc_info=True)
        print(f"Error during optimization: {str(e)}")
        results = {"error": str(e)}
    
    total_time = time.time() - start_time
    main_logger.info(f"Optimization completed in {total_time:.2f} seconds")
    
    # Print and save results
    if "best_params" in results and results["best_params"] is not None:
        main_logger.info(f"Best parameters: {results['best_params']}")
        main_logger.info(f"Best validation loss: {results['best_value']}")
        
        print("\n" + "="*50)
        print(f"Hyperparameter optimization completed in {total_time:.2f} seconds")
        print(f"Best parameters: {results['best_params']}")
        print(f"Best validation loss: {results['best_value']}")
        print("="*50)
        
        # Create visualizations
        visualize_results(results, args.method, args.out_dir)
    else:
        main_logger.warning("No valid results found")
        print("No valid results found")
    
    # Shut down Ray
    ray.shutdown()
    main_logger.info("Ray shutdown complete")
    print("Ray shutdown complete")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())