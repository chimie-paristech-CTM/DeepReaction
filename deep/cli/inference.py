#!/usr/bin/env python
"""
Inference script for molecular property prediction models.

This script loads a trained model and performs inference on new data. It supports:
1. Loading pre-trained models from checkpoints
2. Batch inference on datasets
3. Single molecule inference
4. Uncertainty estimation
5. Result visualization and export
"""

import os
import sys
import json
import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime

import torch
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader as GeometricDataLoader
import matplotlib.pyplot as plt

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


def setup_inference_logging(args):
    """
    Set up logging specifically for the inference script.

    Args:
        args: Command line arguments

    Returns:
        logging.Logger: Configured logger
    """
    # Create log directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Configure logger
    logger = logging.getLogger("inference")
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear existing handlers

    # Create file handler
    file_handler = logging.FileHandler(os.path.join(args.output_dir, "inference.log"))
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


def create_inference_argparser() -> argparse.ArgumentParser:
    """
    Create an argument parser specifically for inference.

    Returns:
        argparse.ArgumentParser: Argument parser
    """
    parser = argparse.ArgumentParser(description="Inference for molecular property prediction models")

    # Model loading parameters
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--strict_loading", type=bool, default=True,
                        help="Whether to strictly enforce that the keys in checkpoint match the keys in model")

    # Dataset parameters
    parser.add_argument("--dataset", type=str, default=None,
                        choices=[None, "QM7", "QM8", "QM9", "QMugs", "XTB", "benzene", "aspirin", "malonaldehyde",
                                 "ethanol", "toluene"],
                        help="Dataset to use for inference (None for custom input)")
    parser.add_argument("--target_id", type=int, default=0,
                        help="Target property index for datasets with multiple targets")
    parser.add_argument("--dataset_download_dir", type=str, default="./data",
                        help="Directory to download datasets")
    parser.add_argument("--reaction_dataset_root", type=str, default="./data/reaction",
                        help="Root directory for reaction dataset")
    parser.add_argument("--reaction_dataset_csv", type=str, default="reaction_data.csv",
                        help="CSV file containing reaction data")
    parser.add_argument("--reaction_energy_field", type=str, default=None,
                        help="Name of the energy field in the reaction dataset CSV (default: autodetect from common names)")
    parser.add_argument("--reaction_file_suffixes", type=str, nargs=3,
                        default=['_reactant.xyz', '_ts.xyz', '_product.xyz'],
                        help="Suffixes for the three XYZ files (reactant, transition state, product)")
    parser.add_argument("--custom_input_file", type=str, default=None,
                        help="Path to custom input file (CSV, SDF, XYZ, or JSON)")
    parser.add_argument("--use_scaler", action="store_true", default=True,
                        help="Whether to use a scaler for the target values")
    parser.add_argument("--force_model_scaler", action="store_true", default=True,
                        help="Always use the scaler from the model checkpoint, never overwrite with dataset scaler")
    parser.add_argument("--max_num_atoms", type=int, default=100,
                        help="Maximum number of atoms in a molecule")

    # Inference parameters
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for inference")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test", "all"],
                        help="Dataset split to use for inference")
    parser.add_argument("--ensemble", action="store_true",
                        help="Whether to perform ensemble inference with multiple model checkpoints")
    parser.add_argument("--ensemble_dir", type=str, default=None,
                        help="Directory containing ensemble model checkpoints")
    parser.add_argument("--uncertainty", action="store_true",
                        help="Whether to estimate prediction uncertainty")
    parser.add_argument("--monte_carlo_dropout", type=int, default=10,
                        help="Number of Monte Carlo dropout samples for uncertainty estimation")

    # Hardware parameters
    parser.add_argument("--cuda", action="store_true",
                        help="Whether to use CUDA")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for data loading")
    parser.add_argument("--precision", type=str, default="16-mixed",
                        choices=["16-mixed", "32", "bf16", "mixed"],
                        help="Precision for inference")

    # Output parameters
    parser.add_argument("--output_dir", type=str, default="./outputs/inference",
                        help="Output directory")
    parser.add_argument("--save_predictions", action="store_true",
                        help="Whether to save predictions")
    parser.add_argument("--save_embeddings", action="store_true",
                        help="Whether to save molecular embeddings")
    parser.add_argument("--save_visualizations", action="store_true",
                        help="Whether to save visualizations")
    parser.add_argument("--export_format", type=str, default="csv",
                        choices=["csv", "json", "npy"],
                        help="Format for exporting predictions")
    parser.add_argument("--random_seed", type=int, default=42,
                        help="Random seed for dataset splitting (must match training seed)")

    return parser


def load_model(checkpoint_path: str, strict_loading: bool, logger) -> pl.LightningModule:
    """
    Load a trained model from a checkpoint file.

    Args:
        checkpoint_path: Path to checkpoint file
        strict_loading: Whether to strictly enforce that keys match
        logger: Logger instance

    Returns:
        pl.LightningModule: Loaded model
    """
    logger.info(f"Loading model from {checkpoint_path}")

    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}")
        raise

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
        logger.warning("No hyperparameters found in checkpoint, using minimal defaults")
        hparams = {}

    # Ensure max_num_atoms is set
    if 'max_num_atoms_in_mol' not in hparams:
        logger.warning("max_num_atoms_in_mol not found in hyperparameters, setting to default 100")
        hparams['max_num_atoms_in_mol'] = 100

    # Create model
    try:
        model = Estimator(**hparams)
        logger.info("Model created successfully")
    except Exception as e:
        logger.error(f"Error creating model: {e}")
        # Try with minimal hyperparameters
        logger.info("Attempting to create model with minimal hyperparameters")
        minimal_hparams = {
            'readout': hparams.get('readout', 'mean'),
            'batch_size': hparams.get('batch_size', 32),
            'lr': hparams.get('lr', 0.001),
            'max_num_atoms_in_mol': hparams.get('max_num_atoms_in_mol', 100),
        }
        model = Estimator(**minimal_hparams)

    # Load weights
    try:
        model.load_state_dict(checkpoint['state_dict'], strict=strict_loading)
        logger.info("Model weights loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model weights: {e}")
        if not strict_loading:
            logger.warning("Continuing with partially loaded weights")
        else:
            raise

    # Set model to evaluation mode
    model.eval()

    # Check if scaler is available in the loaded model
    if hasattr(model, 'scaler') and model.scaler is not None:
        logger.info(f"Loaded model has scaler of type: {type(model.scaler).__name__}")
        if hasattr(model.scaler, 'mean_'):
            logger.info(f"Model scaler mean: {model.scaler.mean_}")
        if hasattr(model.scaler, 'scale_'):
            logger.info(f"Model scaler scale: {model.scaler.scale_}")
    else:
        logger.warning("Loaded model does not have a scaler attached")

    return model


def load_ensemble_models(ensemble_dir: str, strict_loading: bool, logger) -> List[pl.LightningModule]:
    """
    Load multiple models for ensemble inference.

    Args:
        ensemble_dir: Directory containing model checkpoints
        strict_loading: Whether to strictly enforce that keys match
        logger: Logger instance

    Returns:
        List[pl.LightningModule]: List of loaded models
    """
    logger.info(f"Loading ensemble models from {ensemble_dir}")

    # Check if directory exists
    if not os.path.exists(ensemble_dir):
        raise FileNotFoundError(f"Ensemble directory not found: {ensemble_dir}")

    # Find all checkpoint files
    checkpoint_files = []
    for root, dirs, files in os.walk(ensemble_dir):
        for file in files:
            if file.endswith('.ckpt'):
                checkpoint_files.append(os.path.join(root, file))

    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {ensemble_dir}")

    logger.info(f"Found {len(checkpoint_files)} checkpoint files")

    # Load models
    models = []
    for checkpoint_path in checkpoint_files:
        try:
            model = load_model(checkpoint_path, strict_loading, logger)
            models.append(model)
            logger.info(f"Loaded model from {checkpoint_path}")

            # Check if scaler is available in the loaded model
            if hasattr(model, 'scaler') and model.scaler is not None:
                logger.info(f"Ensemble model has scaler of type: {type(model.scaler).__name__}")
                if hasattr(model.scaler, 'mean_'):
                    logger.info(f"Ensemble model scaler mean: {model.scaler.mean_}")
            else:
                logger.warning(f"Ensemble model from {checkpoint_path} does not have a scaler attached")
        except Exception as e:
            logger.error(f"Error loading model from {checkpoint_path}: {e}")

    if not models:
        raise RuntimeError("Failed to load any models for ensemble")

    return models


def load_dataset(args, logger):
    """
    Load dataset for inference.

    Args:
        args: Command line arguments
        logger: Logger instance

    Returns:
        Tuple: Dataset splits and scaler
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
        train, val, test, scaler = load_reaction(
            args.random_seed,
            root=args.reaction_dataset_root,
            csv_file=args.reaction_dataset_csv,
            use_scaler=args.use_scaler,
            energy_field=args.reaction_energy_field,
            file_suffixes=args.reaction_file_suffixes
        )
    elif args.dataset in ['benzene', 'aspirin', 'malonaldehyde', 'ethanol', 'toluene']:
        train, val, test, scaler = load_MD17(ds=args.dataset, download_dir=args.dataset_download_dir)
    elif args.dataset is None and args.custom_input_file is not None:
        # Load custom data
        logger.info(f"Loading custom data from {args.custom_input_file}")
        data, scaler = load_custom_data(args.custom_input_file, logger)
        return {'custom': data}, scaler, num_workers
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

    # Return specified split or all
    if args.split == 'train':
        return {'train': train}, scaler, num_workers
    elif args.split == 'val':
        return {'val': val}, scaler, num_workers
    elif args.split == 'test':
        return {'test': test}, scaler, num_workers
    elif args.split == 'all':
        return {'train': train, 'val': val, 'test': test}, scaler, num_workers
    else:
        raise ValueError(f"Invalid split: {args.split}")


def load_custom_data(file_path, logger):
    """
    Load custom data from a file.

    Args:
        file_path: Path to input file
        logger: Logger instance

    Returns:
        Tuple: Custom dataset and scaler
    """
    logger.info(f"Loading custom data from {file_path}")

    # Check file type
    file_ext = os.path.splitext(file_path)[1].lower()

    if file_ext == '.csv':
        logger.info("Detected CSV file format")
        # Logic for CSV file loading
        # This is a placeholder and should be implemented based on your data format
        raise NotImplementedError("CSV loading not implemented yet")

    elif file_ext == '.sdf':
        logger.info("Detected SDF file format")
        # Logic for SDF file loading
        # This is a placeholder and should be implemented based on your data format
        raise NotImplementedError("SDF loading not implemented yet")

    elif file_ext == '.xyz':
        logger.info("Detected XYZ file format")
        # Logic for XYZ file loading
        # This is a placeholder and should be implemented based on your data format
        raise NotImplementedError("XYZ loading not implemented yet")

    elif file_ext == '.json':
        logger.info("Detected JSON file format")
        # Logic for JSON file loading
        # This is a placeholder and should be implemented based on your data format
        raise NotImplementedError("JSON loading not implemented yet")

    else:
        raise ValueError(f"Unsupported file format: {file_ext}")


def create_dataloaders(dataset_splits, num_workers, batch_size, is_reaction_dataset):
    """
    Create data loaders from dataset splits.

    Args:
        dataset_splits: Dictionary of dataset splits
        num_workers: Number of workers for data loading
        batch_size: Batch size for inference
        is_reaction_dataset: Whether the dataset is a reaction dataset

    Returns:
        Dict: Dictionary of data loaders
    """
    # Configure data loader parameters
    dataloader_kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'shuffle': False
    }

    if is_reaction_dataset:
        dataloader_kwargs['follow_batch'] = ['pos0', 'pos1', 'pos2']

    # Create dataloaders
    dataloaders = {}
    for split_name, split_data in dataset_splits.items():
        dataloaders[split_name] = GeometricDataLoader(
            split_data,
            **dataloader_kwargs
        )

    return dataloaders


def run_inference(model, dataloader, use_cuda, scaler=None, uncertainty=False, mc_samples=10, logger=None):
    """
    Run inference on a dataloader.

    Args:
        model: Model instance
        dataloader: Data loader
        use_cuda: Whether to use CUDA
        scaler: Scaler for target values
        uncertainty: Whether to estimate uncertainty
        mc_samples: Number of Monte Carlo samples for uncertainty estimation
        logger: Logger instance

    Returns:
        Dict: Dictionary containing predictions, true values, and metrics
    """
    # Move model to device
    device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Set model to evaluation mode
    model.eval()

    # Check if the model has a scaler attached
    model_scaler = getattr(model, 'scaler', None)

    # Determine which scaler to use
    effective_scaler = model_scaler if model_scaler is not None else scaler

    if effective_scaler is not None:
        if logger:
            logger.info(f"Will use scaler of type {type(effective_scaler).__name__} for inference")
            if hasattr(effective_scaler, 'mean_'):
                logger.info(f"Inference scaler mean: {effective_scaler.mean_}")
            if hasattr(effective_scaler, 'scale_'):
                logger.info(f"Inference scaler scale: {effective_scaler.scale_}")
    else:
        if logger:
            logger.info("No scaler will be used for inference")

    # Initialize lists to store outputs
    y_pred = []
    y_true = []
    graph_embeddings = []
    node_embeddings = []
    uncertainties = []

    # Run inference
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            batch = batch.to(device)

            if uncertainty:
                # Enable dropout for Monte Carlo sampling
                model.train()

                # Run multiple forward passes
                mc_preds = []
                for _ in range(mc_samples):
                    node_emb, graph_emb, pred = model(
                        pos0=batch.pos0,
                        pos1=batch.pos1,
                        pos2=batch.pos2,
                        atom_z=batch.z,
                        batch_mapping=batch.batch
                    )
                    mc_preds.append(pred.detach().cpu().numpy())

                # Calculate mean and standard deviation
                mc_preds = np.array(mc_preds)
                predictions = np.mean(mc_preds, axis=0)
                uncertainty = np.std(mc_preds, axis=0)

                # Store uncertainty
                uncertainties.extend(uncertainty)

                # Set model back to evaluation mode
                model.eval()
            else:
                # Single forward pass
                node_emb, graph_emb, predictions = model(
                    pos0=batch.pos0,
                    pos1=batch.pos1,
                    pos2=batch.pos2,
                    atom_z=batch.z,
                    batch_mapping=batch.batch
                )
                predictions = predictions.detach().cpu().numpy()

            # Store predictions and targets
            y_pred.extend(predictions)
            y_true.extend(batch.y.detach().cpu().numpy())

            # Store embeddings if needed
            graph_embeddings.extend(graph_emb.detach().cpu().numpy())
            node_embeddings.extend(node_emb.detach().cpu().numpy())

    # Convert to numpy arrays
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    graph_embeddings = np.array(graph_embeddings)

    # Apply inverse transform if effective_scaler is provided
    if effective_scaler is not None:
        try:
            if logger:
                logger.info("Applying inverse transform with scaler")
            y_pred = effective_scaler.inverse_transform(y_pred.reshape(-1, 1)).reshape(-1)
            y_true = effective_scaler.inverse_transform(y_true.reshape(-1, 1)).reshape(-1)
        except Exception as e:
            if logger:
                logger.warning(f"Failed to apply inverse transform: {e}")
            else:
                print(f"Warning: Failed to apply inverse transform: {e}")

    # Compute metrics
    metrics = compute_regression_metrics(y_true, y_pred)

    # Create result dictionary
    result = {
        'predictions': y_pred,
        'true_values': y_true,
        'metrics': metrics,
        'graph_embeddings': graph_embeddings,
        'node_embeddings': node_embeddings
    }

    if uncertainty:
        result['uncertainties'] = np.array(uncertainties)

    return result


def run_ensemble_inference(models, dataloader, use_cuda, scaler=None, logger=None):
    """
    Run ensemble inference with multiple models.

    Args:
        models: List of model instances
        dataloader: Data loader
        use_cuda: Whether to use CUDA
        scaler: Scaler for target values
        logger: Logger instance

    Returns:
        Dict: Dictionary containing predictions, true values, metrics, and standard deviations
    """
    # Initialize list to store predictions from each model
    all_predictions = []

    # Run inference with each model
    for i, model in enumerate(models):
        if logger:
            logger.info(f"Running inference with model {i + 1}/{len(models)}")
        else:
            print(f"Running inference with model {i + 1}/{len(models)}")

        # Run inference using model's own scaler if available
        result = run_inference(model, dataloader, use_cuda, scaler, logger=logger)

        # Store predictions
        all_predictions.append(result['predictions'])

    # Convert to numpy array
    all_predictions = np.array(all_predictions)

    # Calculate mean and standard deviation
    mean_predictions = np.mean(all_predictions, axis=0)
    std_predictions = np.std(all_predictions, axis=0)

    # Use true values from the last result
    y_true = result['true_values']

    # Compute metrics
    metrics = compute_regression_metrics(y_true, mean_predictions)

    # Create result dictionary
    ensemble_result = {
        'predictions': mean_predictions,
        'true_values': y_true,
        'metrics': metrics,
        'std_predictions': std_predictions,
        'individual_predictions': all_predictions,
        'graph_embeddings': result['graph_embeddings']  # Use embeddings from last model
    }

    return ensemble_result


def save_results(results, output_dir, export_format, save_embeddings, save_visualizations, split_name):
    """
    Save inference results to files.

    Args:
        results: Inference results
        output_dir: Output directory
        export_format: Format for exporting predictions
        save_embeddings: Whether to save embeddings
        save_visualizations: Whether to save visualizations
        split_name: Name of the dataset split
    """
    # Create directory for split
    split_dir = os.path.join(output_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)

    # Save predictions and true values
    if export_format == 'csv':
        # Save as CSV
        df = pd.DataFrame({
            'y_pred': results['predictions'],
            'y_true': results['true_values']
        })

        # Add uncertainties if available
        if 'uncertainties' in results:
            df['uncertainty'] = results['uncertainties']
        elif 'std_predictions' in results:
            df['std_predictions'] = results['std_predictions']

        # Save to CSV
        df.to_csv(os.path.join(split_dir, 'predictions.csv'), index=False)

    elif export_format == 'json':
        # Save as JSON
        json_data = {
            'predictions': results['predictions'].tolist(),
            'true_values': results['true_values'].tolist(),
            'metrics': results['metrics']
        }

        # Add uncertainties if available
        if 'uncertainties' in results:
            json_data['uncertainties'] = results['uncertainties'].tolist()
        elif 'std_predictions' in results:
            json_data['std_predictions'] = results['std_predictions'].tolist()

        # Save to JSON
        with open(os.path.join(split_dir, 'predictions.json'), 'w') as f:
            json.dump(json_data, f, indent=2)

    elif export_format == 'npy':
        # Save as NumPy arrays
        np.save(os.path.join(split_dir, 'y_pred.npy'), results['predictions'])
        np.save(os.path.join(split_dir, 'y_true.npy'), results['true_values'])

        # Save uncertainties if available
        if 'uncertainties' in results:
            np.save(os.path.join(split_dir, 'uncertainties.npy'), results['uncertainties'])
        elif 'std_predictions' in results:
            np.save(os.path.join(split_dir, 'std_predictions.npy'), results['std_predictions'])

    # Save metrics
    with open(os.path.join(split_dir, 'metrics.json'), 'w') as f:
        json.dump(results['metrics'], f, indent=2)

    # Save embeddings if requested
    if save_embeddings:
        np.save(os.path.join(split_dir, 'graph_embeddings.npy'), results['graph_embeddings'])

        if 'node_embeddings' in results:
            np.save(os.path.join(split_dir, 'node_embeddings.npy'), results['node_embeddings'])

    # Create visualizations if requested
    if save_visualizations:
        # Create directory for visualizations
        vis_dir = os.path.join(split_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)

        # Plot predictions vs. true values
        plt.figure(figsize=(10, 8))
        plt.scatter(results['true_values'], results['predictions'], alpha=0.5)
        plt.plot([min(results['true_values']), max(results['true_values'])],
                 [min(results['true_values']), max(results['true_values'])], 'r--')
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.title(f'Predictions vs. True Values ({split_name})')
        plt.savefig(os.path.join(vis_dir, 'predictions_vs_true.png'), dpi=300)
        plt.close()

        # Plot error distribution
        errors = results['predictions'] - results['true_values']
        plt.figure(figsize=(10, 6))
        plt.hist(errors, bins=30, alpha=0.7)
        plt.axvline(x=0, color='r', linestyle='--')
        plt.xlabel('Prediction Error')
        plt.ylabel('Frequency')
        plt.title(f'Error Distribution ({split_name})')
        plt.savefig(os.path.join(vis_dir, 'error_distribution.png'), dpi=300)
        plt.close()

        # Plot predictions with uncertainty if available
        if 'uncertainties' in results or 'std_predictions' in results:
            uncertainty = results.get('uncertainties', results.get('std_predictions'))

            # Sort by true values for better visualization
            sorted_indices = np.argsort(results['true_values'])
            sorted_true = results['true_values'][sorted_indices]
            sorted_pred = results['predictions'][sorted_indices]
            sorted_unc = uncertainty[sorted_indices]

            plt.figure(figsize=(12, 6))
            plt.errorbar(
                range(len(sorted_true)),
                sorted_pred,
                yerr=sorted_unc,
                fmt='o',
                alpha=0.5,
                elinewidth=1,
                capsize=2
            )
            plt.plot(range(len(sorted_true)), sorted_true, 'r-', alpha=0.7)
            plt.xlabel('Sample Index (sorted by true value)')
            plt.ylabel('Value')
            plt.title(f'Predictions with Uncertainty ({split_name})')
            plt.legend(['True Values', 'Predictions'])
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, 'predictions_with_uncertainty.png'), dpi=300)
            plt.close()

            # Plot uncertainty vs. error
            plt.figure(figsize=(10, 8))
            plt.scatter(uncertainty, np.abs(errors), alpha=0.5)
            plt.xlabel('Uncertainty')
            plt.ylabel('Absolute Error')
            plt.title(f'Uncertainty vs. Absolute Error ({split_name})')
            plt.savefig(os.path.join(vis_dir, 'uncertainty_vs_error.png'), dpi=300)
            plt.close()


def main():
    """
    Main inference function.
    """
    # Parse command line arguments
    parser = create_inference_argparser()
    args = parser.parse_args()

    # Create output directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.dataset:
        args.output_dir = os.path.join(args.output_dir, f"{args.dataset}_{timestamp}")
    else:
        args.output_dir = os.path.join(args.output_dir, f"inference_{timestamp}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Set up logging
    logger = setup_inference_logging(args)

    # Save configuration
    config_path = os.path.join(args.output_dir, 'inference_config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    logger.info(f"Configuration saved to {config_path}")

    # Print configuration summary
    logger.info("Inference configuration:")
    for key, value in vars(args).items():
        logger.info(f"  {key}: {value}")

    # Load model or ensemble
    if args.ensemble:
        if args.ensemble_dir:
            models = load_ensemble_models(args.ensemble_dir, args.strict_loading, logger)
        else:
            logger.error("Ensemble directory not specified")
            return
    else:
        model = load_model(args.model_path, args.strict_loading, logger)

    # Load dataset
    try:
        dataset_splits, scaler, num_workers = load_dataset(args, logger)
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return

    # Set the scaler on the model(s) ONLY if model doesn't already have one and forcing model scaler is not enabled
    if args.use_scaler and not args.force_model_scaler:
        if args.ensemble:
            for i, model in enumerate(models):
                if not hasattr(model, 'scaler') or model.scaler is None:
                    logger.info(f"Setting scaler for ensemble model {i} from dataset")
                    model.scaler = scaler
                else:
                    logger.info(f"Using existing scaler from ensemble model {i} checkpoint")
        else:
            if not hasattr(model, 'scaler') or model.scaler is None:
                logger.info("Setting scaler from dataset")
                model.scaler = scaler
            else:
                logger.info("Using existing scaler from model checkpoint")
    elif args.force_model_scaler:
        logger.info("Forcing use of model checkpoint scaler (will not overwrite with dataset scaler)")

    # Create dataloaders
    try:
        dataloaders = create_dataloaders(
            dataset_splits,
            num_workers,
            args.batch_size,
            args.dataset == 'XTB'
        )
    except Exception as e:
        logger.error(f"Error creating dataloaders: {e}")
        return

    # Run inference on each split
    for split_name, dataloader in dataloaders.items():
        logger.info(f"Running inference on {split_name} split ({len(dataloader.dataset)} samples)")

        try:
            if args.ensemble:
                # Run ensemble inference
                results = run_ensemble_inference(models, dataloader, args.cuda, scaler, logger)
                logger.info(f"Ensemble inference completed successfully")
            else:
                # Run single model inference
                results = run_inference(
                    model,
                    dataloader,
                    args.cuda,
                    scaler,
                    args.uncertainty,
                    args.monte_carlo_dropout,
                    logger
                )
                logger.info(f"Inference completed successfully")

            # Log metrics
            logger.info(f"Metrics for {split_name} split:")
            for key, value in results['metrics'].items():
                logger.info(f"  {key}: {value:.6f}")

            # Save results
            if args.save_predictions:
                save_results(
                    results,
                    args.output_dir,
                    args.export_format,
                    args.save_embeddings,
                    args.save_visualizations,
                    split_name
                )
                logger.info(f"Results saved to {os.path.join(args.output_dir, split_name)}")

        except Exception as e:
            logger.error(f"Error during inference on {split_name} split: {e}")
            import traceback
            logger.error(traceback.format_exc())

    logger.info("Inference completed successfully")


if __name__ == "__main__":
    main()