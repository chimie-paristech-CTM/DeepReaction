#!/usr/bin/env python
"""
Configuration module for the molecular property prediction package.

This module handles all configuration aspects of the training pipeline, including:
1. Command-line argument parsing
2. Configuration validation
3. Configuration serialization and deserialization
4. Default configurations
"""

import argparse
import json
import os
import sys
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
import torch
from datetime import datetime

# Constants
SUPPORTED_DATASETS = [
    'QM7', 'QM8', 'QM9', 'QMugs', 'benzene', 'aspirin',
    'malonaldehyde', 'ethanol', 'toluene', 'XTB'
]

SUPPORTED_READOUTS = [
    'set_transformer', 'mean', 'sum', 'max', 'attention',
    'multihead_attention', 'set2set', 'sort_pool'
]

SUPPORTED_OPTIMIZERS = [
    'adam', 'adamw', 'sgd', 'rmsprop', 'adagrad'
]

SUPPORTED_LR_SCHEDULERS = [
    'cosine', 'step', 'exponential', 'plateau', 'warmup_cosine',
    'cyclic', 'one_cycle', 'constant', 'warmup_constant'
]

SUPPORTED_LOSS_FUNCTIONS = [
    'mse', 'mae', 'huber', 'smooth_l1', 'cross_entropy',
    'binary_cross_entropy', 'evidence_lower_bound'
]

MAX_NUM_ATOMS_IN_MOL = {
    'QM7': 23,
    'QM8': 26,
    'QM9': 29,
    'QMugs': 228,
    'benzene': 12,
    'aspirin': 21,
    'malonaldehyde': 9,
    'ethanol': 9,
    'toluene': 15,
    'XTB': 100,
}

# Default dataset paths for reaction dataset
DEFAULT_REACTION_DATASET_ROOT = '/root/attention-based-pooling-for-quantum-properties-main/data_loading/DATASET_DA'
DEFAULT_REACTION_DATASET_CSV = '/root/attention-based-pooling-for-quantum-properties-main/data_loading/DATASET_DA/DA_dataset_cleaned.csv'


def get_parser() -> argparse.ArgumentParser:
    """
    Create and return the argument parser for the training script.

    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description='Train a DimeNet++ model for molecular property prediction',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Add argument groups
    parser = add_general_args(parser)
    parser = add_data_args(parser)
    parser = add_model_args(parser)
    parser = add_dimenet_args(parser)
    parser = add_readout_args(parser)
    parser = add_training_args(parser)
    parser = add_optimization_args(parser)
    parser = add_output_args(parser)
    parser = add_feature_args(parser)
    parser = add_ensemble_args(parser)
    parser = add_distributed_args(parser)
    parser = add_logging_args(parser)
    parser = add_dataset_specific_args(parser)

    return parser


def add_general_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Add general configuration arguments.

    Args:
        parser (argparse.ArgumentParser): Argument parser

    Returns:
        argparse.ArgumentParser: Updated argument parser
    """
    general_group = parser.add_argument_group('General Parameters')

    general_group.add_argument(
        '--config', type=str, default=None,
        help='Path to a YAML or JSON configuration file'
    )

    general_group.add_argument(
        '--random_seed', type=int, default=42,
        help='Random seed for reproducibility'
    )

    general_group.add_argument(
        '--cuda', default=True, action=argparse.BooleanOptionalAction,
        help='Use CUDA for training if available'
    )

    general_group.add_argument(
        '--precision', type=str, default='32', choices=['16', '32', 'bf16', 'mixed'],
        help='Floating point precision to use for training'
    )

    general_group.add_argument(
        '--debug', action='store_true', default=False,
        help='Enable debug mode with more verbose logging and fewer samples'
    )

    return parser


def add_data_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Add dataset-related arguments.

    Args:
        parser (argparse.ArgumentParser): Argument parser

    Returns:
        argparse.ArgumentParser: Updated argument parser
    """
    data_group = parser.add_argument_group('Data Parameters')

    data_group.add_argument(
        '--dataset', type=str, required=True, choices=SUPPORTED_DATASETS,
        help='Dataset to use for training'
    )

    data_group.add_argument(
        '--dataset_path', type=str, default=None,
        help='Path to custom dataset (override default locations)'
    )

    data_group.add_argument(
        '--dataset_download_dir', type=str, default='./data/raw',
        help='Directory to download dataset to'
    )

    data_group.add_argument(
        '--target_id', type=int, required=True,
        help='Index of target property to predict'
    )

    data_group.add_argument(
        '--split_type', type=str, default='random',
        choices=['random', 'scaffold', 'stratified', 'temporal'],
        help='Method to split the dataset'
    )

    data_group.add_argument(
        '--split_sizes', type=float, nargs=3, default=[0.8, 0.1, 0.1],
        help='Split ratios for train/validation/test'
    )

    data_group.add_argument(
        '--num_workers', type=int, default=4,
        help='Number of worker processes for data loading'
    )

    data_group.add_argument(
        '--cache_data', action='store_true', default=False,
        help='Cache processed data for faster loading in subsequent runs'
    )

    # data_group.add_argument(
    #     '--standard_scale_targets', action='store_true', default=True,
    #     help='Standardize regression targets to have zero mean and unit variance'
    # )
    data_group.add_argument(
        '--use_scaler', action='store_true', default=True,
        help='scaler targets'
    )
    data_group.add_argument(
        '--precompute_distances', action='store_true', default=False,
        help='Precompute distances between atoms for faster training'
    )

    data_group.add_argument(
        '--data_augmentation', action='store_true', default=False,
        help='Enable data augmentation for training'
    )

    return parser


def add_dataset_specific_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Add dataset-specific arguments.

    Args:
        parser (argparse.ArgumentParser): Argument parser

    Returns:
        argparse.ArgumentParser: Updated argument parser
    """
    dataset_specific_group = parser.add_argument_group('Dataset-Specific Parameters')

    # XTB Reaction dataset specific parameters
    dataset_specific_group.add_argument(
        '--reaction_dataset_root', type=str, default=DEFAULT_REACTION_DATASET_ROOT,
        help='Root directory for the reaction dataset (XTB)'
    )

    dataset_specific_group.add_argument(
        '--reaction_dataset_csv', type=str, default=DEFAULT_REACTION_DATASET_CSV,
        help='CSV file path for the reaction dataset (XTB)'
    )

    # New parameter for energy field name
    dataset_specific_group.add_argument(
        '--reaction_energy_field', type=str, default=None,
        help='Name of the energy field in the reaction dataset CSV (default: autodetect from ["dG(ts)", "G(TS)", "G(ts)", "dG(TS)"])'
    )

    # New parameter for XYZ file suffixes
    dataset_specific_group.add_argument(
        '--reaction_file_suffixes', type=str, nargs=3,
        default=['_reactant.xyz', '_ts.xyz', '_product.xyz'],
        help='Suffixes for the three XYZ files (reactant, transition state, product)'
    )

    # MD17 dataset specific parameters
    dataset_specific_group.add_argument(
        '--md17_molecule', type=str, default='benzene',
        choices=['benzene', 'aspirin', 'malonaldehyde', 'ethanol', 'toluene'],
        help='Molecule to use for MD17 dataset'
    )

    # QM9 dataset specific parameters
    dataset_specific_group.add_argument(
        '--qm9_with_hydrogen', action='store_true', default=True,
        help='Include hydrogen atoms in QM9 dataset'
    )

    dataset_specific_group.add_argument(
        '--qm9_target_properties', type=str, nargs='+',
        default=['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv'],
        help='Target properties to load from QM9 dataset'
    )

    # QM7 dataset specific parameters
    dataset_specific_group.add_argument(
        '--qm7_target_property', type=str, default='atomization_energy',
        choices=['atomization_energy'],
        help='Target property to predict for QM7 dataset'
    )

    # QM8 dataset specific parameters
    dataset_specific_group.add_argument(
        '--qm8_target_properties', type=str, nargs='+',
        default=['E1-CC2', 'E2-CC2', 'f1-CC2', 'f2-CC2', 'E1-PBE0', 'E2-PBE0', 'f1-PBE0', 'f2-PBE0',
                 'E1-CAM', 'E2-CAM', 'f1-CAM', 'f2-CAM'],
        help='Target properties to load from QM8 dataset'
    )

    return parser


def add_model_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Add model architecture arguments.

    Args:
        parser (argparse.ArgumentParser): Argument parser

    Returns:
        argparse.ArgumentParser: Updated argument parser
    """
    model_group = parser.add_argument_group('Model Architecture')

    model_group.add_argument(
        '--model_type', type=str, default='dimenet',
        choices=['dimenet', 'dimenet++', 'schnet', 'egnn'],
        help='Type of graph neural network model to use'
    )

    model_group.add_argument(
        '--readout', type=str, required=True, choices=SUPPORTED_READOUTS,
        help='Readout function to use'
    )

    model_group.add_argument(
        '--node_latent_dim', type=int, default=128,
        help='Dimension of node latent representations'
    )

    model_group.add_argument(
        '--edge_latent_dim', type=int, default=64,
        help='Dimension of edge latent representations'
    )

    model_group.add_argument(
        '--use_layer_norm', action='store_true', default=False,
        help='Apply layer normalization in the model'
    )

    model_group.add_argument(
        '--dropout', type=float, default=0.0,
        help='Dropout probability for regularization'
    )

    model_group.add_argument(
        '--activation', type=str, default='silu',
        choices=['relu', 'leaky_relu', 'elu', 'gelu', 'silu', 'swish'],
        help='Activation function to use in the model'
    )

    return parser


def add_dimenet_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Add DimeNet++ specific model arguments.

    Args:
        parser (argparse.ArgumentParser): Argument parser

    Returns:
        argparse.ArgumentParser: Updated argument parser
    """
    dimenet_group = parser.add_argument_group('DimeNet++ Parameters')

    dimenet_group.add_argument(
        '--dimenet_hidden_channels', type=int, default=128,
        help='Hidden channels in DimeNet++'
    )

    dimenet_group.add_argument(
        '--dimenet_num_blocks', type=int, default=4,
        help='Number of interaction blocks'
    )

    dimenet_group.add_argument(
        '--dimenet_int_emb_size', type=int, default=64,
        help='Interaction embedding size'
    )

    dimenet_group.add_argument(
        '--dimenet_basis_emb_size', type=int, default=8,
        help='Basis embedding size'
    )

    dimenet_group.add_argument(
        '--dimenet_out_emb_channels', type=int, default=256,
        help='Output embedding channels'
    )

    dimenet_group.add_argument(
        '--dimenet_num_spherical', type=int, default=7,
        help='Number of spherical harmonics'
    )

    dimenet_group.add_argument(
        '--dimenet_num_radial', type=int, default=6,
        help='Number of radial basis functions'
    )

    dimenet_group.add_argument(
        '--dimenet_cutoff', type=float, default=5.0,
        help='Cutoff distance for interatomic interactions'
    )

    dimenet_group.add_argument(
        '--dimenet_envelope_exponent', type=int, default=5,
        help='Envelope exponent for distance-based message scaling'
    )

    return parser


def add_readout_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Add readout-specific arguments.

    Args:
        parser (argparse.ArgumentParser): Argument parser

    Returns:
        argparse.ArgumentParser: Updated argument parser
    """
    readout_group = parser.add_argument_group('Readout Parameters')

    # Set Transformer parameters
    readout_group.add_argument(
        '--set_transformer_hidden_dim', type=int, default=512,
        help='Hidden dimension for set transformer readout'
    )

    readout_group.add_argument(
        '--set_transformer_num_heads', type=int, default=16,
        help='Number of attention heads in set transformer'
    )

    readout_group.add_argument(
        '--set_transformer_num_sabs', type=int, default=2,
        help='Number of Set Attention Blocks (SABs) in set transformer'
    )

    # Attention readout parameters
    readout_group.add_argument(
        '--attention_hidden_dim', type=int, default=256,
        help='Hidden dimension for attention readout'
    )

    readout_group.add_argument(
        '--attention_num_heads', type=int, default=8,
        help='Number of attention heads in attention readout'
    )

    return parser


def add_training_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Add training-related arguments.

    Args:
        parser (argparse.ArgumentParser): Argument parser

    Returns:
        argparse.ArgumentParser: Updated argument parser
    """
    train_group = parser.add_argument_group('Training Parameters')

    train_group.add_argument(
        '--batch_size', type=int, default=32,
        help='Batch size for training'
    )

    train_group.add_argument(
        '--eval_batch_size', type=int, default=None,
        help='Batch size for evaluation (defaults to training batch size if not specified)'
    )

    train_group.add_argument(
        '--max_epochs', type=int, default=100,
        help='Maximum number of training epochs'
    )

    train_group.add_argument(
        '--min_epochs', type=int, default=10,
        help='Minimum number of training epochs'
    )

    train_group.add_argument(
        '--early_stopping_patience', type=int, default=20,
        help='Number of epochs with no improvement after which training will be stopped'
    )

    train_group.add_argument(
        '--early_stopping_min_delta', type=float, default=0.0001,
        help='Minimum change in the monitored quantity to qualify as an improvement'
    )

    train_group.add_argument(
        '--loss_function', type=str, default='mse', choices=SUPPORTED_LOSS_FUNCTIONS,
        help='Loss function to use for training'
    )

    train_group.add_argument(
        '--uncertainty_method', type=str, default=None,
        choices=[None, 'ensemble', 'dropout', 'evidential'],
        help='Method to estimate prediction uncertainty'
    )

    train_group.add_argument(
        '--cross_validation_folds', type=int, default=0,
        help='Number of cross-validation folds (0 to disable cross-validation)'
    )

    train_group.add_argument(
        '--gradient_clip_val', type=float, default=0.0,
        help='Gradient clipping value (0 to disable)'
    )

    train_group.add_argument(
        '--gradient_accumulation_steps', type=int, default=1,
        help='Number of steps to accumulate gradients before updating weights'
    )

    return parser


def add_optimization_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Add optimization-related arguments.

    Args:
        parser (argparse.ArgumentParser): Argument parser

    Returns:
        argparse.ArgumentParser: Updated argument parser
    """
    optim_group = parser.add_argument_group('Optimization Parameters')

    optim_group.add_argument(
        '--optimizer', type=str, default='adam', choices=SUPPORTED_OPTIMIZERS,
        help='Optimizer to use for training'
    )

    optim_group.add_argument(
        '--lr', type=float, default=0.0001,
        help='Base learning rate'
    )

    optim_group.add_argument(
        '--weight_decay', type=float, default=0.0,
        help='Weight decay (L2 regularization) factor'
    )

    optim_group.add_argument(
        '--scheduler', type=str, default='cosine', choices=SUPPORTED_LR_SCHEDULERS,
        help='Learning rate scheduler to use'
    )

    optim_group.add_argument(
        '--warmup_epochs', type=int, default=10,
        help='Number of epochs for learning rate warmup'
    )

    optim_group.add_argument(
        '--scheduler_patience', type=int, default=5,
        help='Patience for schedulers like ReduceLROnPlateau'
    )

    optim_group.add_argument(
        '--scheduler_factor', type=float, default=0.5,
        help='Factor by which the learning rate will be reduced'
    )

    optim_group.add_argument(
        '--min_lr', type=float, default=1e-6,
        help='Minimum learning rate'
    )

    return parser


def add_output_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Add output-related arguments.

    Args:
        parser (argparse.ArgumentParser): Argument parser

    Returns:
        argparse.ArgumentParser: Updated argument parser
    """
    output_group = parser.add_argument_group('Output Parameters')

    output_group.add_argument(
        '--out_dir', type=str, default='./results',
        help='Base directory to save model checkpoints and results'
    )

    output_group.add_argument(
        '--experiment_name', type=str, default=None,
        help='Name of the experiment (default: autogenerated from parameters)'
    )

    output_group.add_argument(
        '--save_best_model', action='store_true', default=True,
        help='Save the best model based on validation metrics'
    )

    output_group.add_argument(
        '--save_last_model', action='store_true', default=False,
        help='Save the model from the last epoch'
    )

    output_group.add_argument(
        '--save_predictions', action='store_true', default=True,
        help='Save model predictions on test set'
    )

    output_group.add_argument(
        '--save_interval', type=int, default=0,
        help='Save checkpoints every N epochs (0 to disable interval saving)'
    )

    output_group.add_argument(
        '--save_visualizations', action='store_true', default=False,
        help='Save model visualizations (attention maps, feature importance, etc.)'
    )

    output_group.add_argument(
        '--ckpt_path', type=str, default=None,
        help='Path to checkpoint to resume training from'
    )

    return parser


def add_feature_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Add feature engineering and analysis arguments.

    Args:
        parser (argparse.ArgumentParser): Argument parser

    Returns:
        argparse.ArgumentParser: Updated argument parser
    """
    feature_group = parser.add_argument_group('Feature Parameters')

    feature_group.add_argument(
        '--feature_scaling', type=str, default='standard',
        choices=['standard', 'minmax', 'robust', 'none'],
        help='Method to scale input features'
    )

    feature_group.add_argument(
        '--feature_selection', action='store_true', default=False,
        help='Apply feature selection to input features'
    )

    feature_group.add_argument(
        '--feature_importance', action='store_true', default=False,
        help='Calculate feature importance after training'
    )

    # feature_group.add_argument(
    #     '--atom_features', type=str, nargs='+',
    #     default=['element', 'degree', 'hybridization', 'formal_charge'],
    #     help='Atom features to use in the model'
    # )

    # feature_group.add_argument(
    #     '--bond_features', type=str, nargs='+',
    #     default=['bond_type', 'distance', 'same_ring'],
    #     help='Bond features to use in the model'
    # )

    # feature_group.add_argument(
    #     '--use_3d_coordinates', action='store_true', default=True,
    #     help='Use 3D coordinates for spatial message passing'
    # )

    return parser


def add_ensemble_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Add ensemble model arguments.

    Args:
        parser (argparse.ArgumentParser): Argument parser

    Returns:
        argparse.ArgumentParser: Updated argument parser
    """
    ensemble_group = parser.add_argument_group('Ensemble Parameters')

    ensemble_group.add_argument(
        '--ensemble_size', type=int, default=1,
        help='Number of models to use in the ensemble'
    )

    ensemble_group.add_argument(
        '--ensemble_method', type=str, default='mean',
        choices=['mean', 'median', 'voting', 'weighted'],
        help='Method to combine ensemble predictions'
    )

    ensemble_group.add_argument(
        '--ensemble_seeds', type=int, nargs='+', default=None,
        help='Random seeds for ensemble models (default: auto-generated)'
    )

    return parser


def add_distributed_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Add distributed training arguments.

    Args:
        parser (argparse.ArgumentParser): Argument parser

    Returns:
        argparse.ArgumentParser: Updated argument parser
    """
    dist_group = parser.add_argument_group('Distributed Training Parameters')

    dist_group.add_argument(
        '--strategy', type=str, default='auto',
        choices=['auto', 'ddp', 'deepspeed', 'fsdp', 'none'],
        help='Distributed training strategy'
    )

    dist_group.add_argument(
        '--num_nodes', type=int, default=1,
        help='Number of nodes for distributed training'
    )

    dist_group.add_argument(
        '--devices', type=int, default=1,
        help='Number of devices (GPUs) per node'
    )

    dist_group.add_argument(
        '--sync_batchnorm', action='store_true', default=False,
        help='Synchronize batch normalization statistics across GPUs'
    )

    return parser


def add_logging_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Add logging and monitoring arguments.

    Args:
        parser (argparse.ArgumentParser): Argument parser

    Returns:
        argparse.ArgumentParser: Updated argument parser
    """
    logging_group = parser.add_argument_group('Logging Parameters')

    logging_group.add_argument(
        '--log_level', type=str, default='info',
        choices=['debug', 'info', 'warning', 'error', 'critical'],
        help='Logging level'
    )

    logging_group.add_argument(
        '--log_to_file', action='store_true', default=False,
        help='Save logs to a file'
    )

    logging_group.add_argument(
        '--log_every_n_steps', type=int, default=50,
        help='Log metrics every N training steps'
    )

    logging_group.add_argument(
        '--logger_type', type=str, default='tensorboard',
        choices=['tensorboard', 'wandb', 'csv', 'all'],
        help='Logger to use for training metrics'
    )

    logging_group.add_argument(
        '--wandb_project', type=str, default='molecular-property-prediction',
        help='Project name for Weights & Biases logging'
    )

    logging_group.add_argument(
        '--progress_bar', action='store_true', default=True,
        help='Show progress bar during training'
    )

    return parser


def process_args(parser: argparse.ArgumentParser = None) -> argparse.Namespace:
    """
    Process and validate command line arguments.

    Args:
        parser (argparse.ArgumentParser, optional): Custom argument parser.
                                                   If None, a new parser will be created.

    Returns:
        argparse.Namespace: Processed and validated arguments
    """
    if parser is None:
        parser = get_parser()

    args = parser.parse_args()

    # Load configuration from file if specified
    if args.config is not None:
        config_dict = load_config(args.config)

        # Update args with values from config file (command line args take precedence)
        for key, value in config_dict.items():
            if not hasattr(args, key) or getattr(args, key) == parser.get_default(key):
                setattr(args, key, value)

    # Validate arguments
    validate_args(args)

    # Process and derive additional arguments
    process_derived_args(args)

    return args


def validate_args(args: argparse.Namespace) -> None:
    """
    Validate command line arguments for consistency and correctness.

    Args:
        args (argparse.Namespace): Command line arguments

    Raises:
        ValueError: If arguments are invalid or inconsistent
    """
    # Validate dataset and target
    if args.dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset: {args.dataset}. Choose from {SUPPORTED_DATASETS}")

    # Validate split sizes
    if len(args.split_sizes) != 3 or sum(args.split_sizes) != 1.0:
        raise ValueError(f"Split sizes must be 3 values that sum to 1.0, got {args.split_sizes}")

    # Validate ensemble settings
    if args.ensemble_size > 1 and args.ensemble_seeds is not None:
        if len(args.ensemble_seeds) != args.ensemble_size:
            raise ValueError(
                f"Number of ensemble seeds ({len(args.ensemble_seeds)}) must match ensemble size ({args.ensemble_size})")

    # Validate batch sizes
    if args.batch_size <= 0:
        raise ValueError(f"Batch size must be positive, got {args.batch_size}")

    # Validate eval batch size
    if args.eval_batch_size is not None and args.eval_batch_size <= 0:
        raise ValueError(f"Evaluation batch size must be positive, got {args.eval_batch_size}")

    # Validate cross-validation settings
    if args.cross_validation_folds < 0:
        raise ValueError(f"Number of cross-validation folds must be non-negative, got {args.cross_validation_folds}")

    # Validate learning rate and epochs
    if args.lr <= 0:
        raise ValueError(f"Learning rate must be positive, got {args.lr}")

    if args.max_epochs < args.min_epochs:
        raise ValueError(
            f"Maximum epochs ({args.max_epochs}) must be greater than or equal to minimum epochs ({args.min_epochs})")

    # Validate precision
    if args.precision not in ['16', '32', 'bf16', 'mixed']:
        raise ValueError(f"Precision must be one of ['16', '32', 'bf16', 'mixed'], got {args.precision}")

    # Validate dataset-specific paths
    if args.dataset == 'XTB':
        if not os.path.exists(args.reaction_dataset_root):
            raise ValueError(f"Reaction dataset root directory does not exist: {args.reaction_dataset_root}")
        if not os.path.exists(args.reaction_dataset_csv):
            raise ValueError(f"Reaction dataset CSV file does not exist: {args.reaction_dataset_csv}")

        # Validate reaction file suffixes
        if args.reaction_file_suffixes is not None and len(args.reaction_file_suffixes) != 3:
            raise ValueError(
                f"reaction_file_suffixes must specify exactly 3 suffixes, got {len(args.reaction_file_suffixes)}")


def process_derived_args(args: argparse.Namespace) -> None:
    """
    Process and derive additional arguments based on the provided ones.

    Args:
        args (argparse.Namespace): Command line arguments
    """
    # Set evaluation batch size if not specified
    if args.eval_batch_size is None:
        args.eval_batch_size = args.batch_size

    # Set output directory structure
    if args.experiment_name is None:
        # Generate a descriptive name based on arguments
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"{args.dataset}_target{args.target_id}_{args.model_type}_{args.readout}_seed{args.random_seed}_{timestamp}"

    # Create complete output directory path
    args.output_dir = os.path.join(
        args.out_dir,
        args.model_type,
        args.dataset,
        str(args.random_seed),
        str(args.target_id),
        args.readout,
        args.experiment_name
    )

    # Set ensemble seeds if not specified
    if args.ensemble_size > 1 and args.ensemble_seeds is None:
        base_seed = args.random_seed
        args.ensemble_seeds = [base_seed + i for i in range(args.ensemble_size)]

    # Set model specific parameters
    args.max_num_atoms = MAX_NUM_ATOMS_IN_MOL.get(args.dataset, 100)

    # Adjust number of workers based on dataset
    if args.dataset in ['benzene', 'aspirin', 'malonaldehyde', 'ethanol', 'toluene']:
        args.num_workers = 0


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML or JSON file.

    Args:
        config_path (str): Path to configuration file

    Returns:
        Dict[str, Any]: Configuration dictionary

    Raises:
        ValueError: If file format is not supported or file does not exist
    """
    if not os.path.exists(config_path):
        raise ValueError(f"Configuration file does not exist: {config_path}")

    file_ext = os.path.splitext(config_path)[1].lower()

    if file_ext == '.yaml' or file_ext == '.yml':
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    elif file_ext == '.json':
        with open(config_path, 'r') as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported configuration file format: {file_ext}. Use .yaml, .yml, or .json")


def save_config(args: argparse.Namespace, output_dir: str) -> str:
    """
    Save configuration to YAML and JSON files.

    Args:
        args (argparse.Namespace): Command line arguments
        output_dir (str): Output directory

    Returns:
        str: Path to saved configuration file
    """
    os.makedirs(output_dir, exist_ok=True)

    # Convert args to dictionary
    config_dict = vars(args)

    # Save as YAML
    yaml_path = os.path.join(output_dir, 'config.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)

    # Save as JSON
    json_path = os.path.join(output_dir, 'config.json')
    with open(json_path, 'w') as f:
        json.dump(config_dict, f, indent=2)

    return yaml_path


def get_model_name(args: argparse.Namespace) -> str:
    """
    Generate a descriptive model name from command line arguments.

    Args:
        args (argparse.Namespace): Command line arguments

    Returns:
        str: Descriptive model name
    """
    name = f'DATASET={args.dataset}+target_id={args.target_id}+random_seed={args.random_seed}+readout={args.readout}+lr={args.lr}'

    if args.model_type in ['dimenet', 'dimenet++']:
        name += f'+dn_hd_cnl={args.dimenet_hidden_channels}+dn_num_blk={args.dimenet_num_blocks}+dn_int_emb_sz={args.dimenet_int_emb_size}'
        name += f'+dn_basis_emb_sz={args.dimenet_basis_emb_size}+dn_out_emb_cnl={args.dimenet_out_emb_channels}'

    if args.readout == 'set_transformer':
        name += f'+set_transformer_num_SABs={args.set_transformer_num_sabs}+set_transformer_hidden_dim={args.set_transformer_hidden_dim}'
        name += f'+set_transformer_num_heads={args.set_transformer_num_heads}'

    return name


def setup_logging(args: argparse.Namespace) -> logging.Logger:
    """
    Set up logging configuration.

    Args:
        args (argparse.Namespace): Command line arguments

    Returns:
        logging.Logger: Configured logger
    """
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Set up logging level
    log_level = getattr(logging, args.log_level.upper())

    # Create logger
    logger = logging.getLogger('deep')
    logger.setLevel(log_level)

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Create file handler if requested
    if args.log_to_file:
        log_file = os.path.join(args.output_dir, 'training.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_experiment_config(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Get experiment configuration dictionary for logging with experiment trackers.

    Args:
        args (argparse.Namespace): Command line arguments

    Returns:
        Dict[str, Any]: Experiment configuration
    """
    # Get basic system information
    config = {
        'dataset': args.dataset,
        'target_id': args.target_id,
        'model_type': args.model_type,
        'readout': args.readout,
        'batch_size': args.batch_size,
        'max_epochs': args.max_epochs,
        'optimizer': args.optimizer,
        'learning_rate': args.lr,
        'random_seed': args.random_seed,
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'num_gpus': torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }

    if args.model_type in ['dimenet', 'dimenet++']:
        config.update({
            'dimenet_hidden_channels': args.dimenet_hidden_channels,
            'dimenet_num_blocks': args.dimenet_num_blocks,
            'dimenet_int_emb_size': args.dimenet_int_emb_size,
            'dimenet_basis_emb_size': args.dimenet_basis_emb_size,
            'dimenet_out_emb_channels': args.dimenet_out_emb_channels,
        })

    if args.readout == 'set_transformer':
        config.update({
            'set_transformer_hidden_dim': args.set_transformer_hidden_dim,
            'set_transformer_num_heads': args.set_transformer_num_heads,
            'set_transformer_num_sabs': args.set_transformer_num_sabs,
        })

    # Dataset-specific configurations
    if args.dataset == 'XTB':
        config.update({
            'reaction_dataset_root': args.reaction_dataset_root,
            'reaction_dataset_csv': args.reaction_dataset_csv,
            'reaction_energy_field': args.reaction_energy_field,
            'reaction_file_suffixes': args.reaction_file_suffixes
        })

    return config


def print_args_summary(args: argparse.Namespace) -> None:
    """
    Print a summary of important arguments.

    Args:
        args (argparse.Namespace): Command line arguments
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT CONFIGURATION")
    print("=" * 80)
    print(f"Dataset:              {args.dataset}")
    print(f"Target ID:            {args.target_id}")
    print(f"Model Type:           {args.model_type}")
    print(f"Readout:              {args.readout}")
    print(f"Random Seed:          {args.random_seed}")
    print(f"Batch Size:           {args.batch_size}")
    print(f"Learning Rate:        {args.lr}")
    print(f"Optimizer:            {args.optimizer}")
    print(f"Scheduler:            {args.scheduler}")
    print(f"Max Epochs:           {args.max_epochs}")
    print(f"Early Stop Patience:  {args.early_stopping_patience}")
    print(f"Ensemble Size:        {args.ensemble_size}")
    print(f"Output Directory:     {args.output_dir}")
    print(f"CUDA Available:       {torch.cuda.is_available()}")
    print(f"CUDA Devices:         {torch.cuda.device_count() if torch.cuda.is_available() else 0}")

    # Print dataset-specific parameters
    if args.dataset == 'XTB':
        print("\nREACTION DATASET SETTINGS:")
        print(f"Dataset Root:         {args.reaction_dataset_root}")
        print(f"Dataset CSV:          {args.reaction_dataset_csv}")
        print(f"Energy Field:         {args.reaction_energy_field or 'Auto-detect'}")
        print(f"File Suffixes:        {args.reaction_file_suffixes}")

    print("=" * 80 + "\n")


if __name__ == "__main__":
    # Example of how to use this module
    args = process_args()
    print_args_summary(args)

    # Save configuration
    save_config(args, args.output_dir)