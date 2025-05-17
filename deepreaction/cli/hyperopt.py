#!/usr/bin/env python

import os
import sys
import time
import json
import numpy as np
import logging
import random
import shutil
import fcntl
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
from datetime import datetime
import copy
import importlib.util
import hashlib

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


def get_parser():
    from argparse import ArgumentParser

    parser = ArgumentParser(description='Train a molecular graph neural network for property prediction')

    general_group = parser.add_argument_group('General Parameters')
    general_group.add_argument('--config', type=str, default=None, help='Path to a YAML or JSON configuration file')
    general_group.add_argument('--random_seed', type=int, default=42, help='Random seed for reproducibility')
    general_group.add_argument('--cuda', default=True, action='store_true',
                               help='Use CUDA for training if available')
    general_group.add_argument('--no-cuda', dest='cuda', action='store_false',
                               help='Do not use CUDA even if available')
    general_group.add_argument('--precision', type=str, default='32', choices=['16', '32', 'bf16', 'mixed'],
                               help='Floating point precision')
    general_group.add_argument('--debug', action='store_true', default=False, help='Enable debug mode')

    data_group = parser.add_argument_group('Data Parameters')
    data_group.add_argument('--dataset', type=str, required=True, choices=['XTB'], help='Dataset to use')
    data_group.add_argument('--split_type', type=str, default='random',
                            choices=['random', 'scaffold', 'stratified', 'temporal'])
    data_group.add_argument('--train_ratio', type=float, default=0.8,
                            help='Ratio of training data when using automatic splitting')
    data_group.add_argument('--val_ratio', type=float, default=0.1,
                            help='Ratio of validation data when using automatic splitting')
    data_group.add_argument('--test_ratio', type=float, default=0.1,
                            help='Ratio of test data when using automatic splitting')
    data_group.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    data_group.add_argument('--use_scaler', action='store_true', default=True, help='Scale targets')
    data_group.add_argument('--dataset_csv', type=str, default=None,
                            help='Path to dataset CSV file (used as training set if val_csv and test_csv are provided)')
    data_group.add_argument('--val_csv', type=str, default=None, help='Path to validation set CSV (optional)')
    data_group.add_argument('--test_csv', type=str, default=None, help='Path to test set CSV (optional)')

    data_group.add_argument('--cv_folds', type=int, default=0,
                            help='Number of folds for cross-validation (0 to disable)')
    data_group.add_argument('--cv_test_fold', type=int, default=-1,
                            help='Fold to use for testing (-1 means use a fraction of each fold)')
    data_group.add_argument('--cv_stratify', action='store_true', default=False,
                            help='Stratify folds based on target values')
    data_group.add_argument('--cv_grouped', action='store_true', default=True,
                            help='Keep molecules with the same reaction_id in the same fold')
    data_group.add_argument('--no-cv_grouped', dest='cv_grouped', action='store_false',
                            help='Do not group molecules with the same reaction_id in the same fold')

    model_group = parser.add_argument_group('Model Architecture')
    model_group.add_argument('--model_type', type=str, default='dimenet++', choices=['dimenet++', 'schnet', 'egnn'],
                             help='Type of molecular model to use')
    model_group.add_argument('--readout', type=str, required=True,
                             choices=['set_transformer', 'mean', 'sum', 'max', 'attention',
                                      'multihead_attention', 'set2set', 'sort_pool'],
                             help='Readout function')
    model_group.add_argument('--node_latent_dim', type=int, default=128, help='Node latent dimension')
    model_group.add_argument('--edge_latent_dim', type=int, default=64, help='Edge latent dimension')
    model_group.add_argument('--use_layer_norm', action='store_true', default=False, help='Use layer normalization')
    model_group.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
    model_group.add_argument('--activation', type=str, default='silu',
                             choices=['relu', 'leaky_relu', 'elu', 'gelu', 'silu', 'swish'])
    model_group.add_argument('--use_xtb_features', action='store_true', default=True, help='Use XTB features')
    model_group.add_argument('--prediction_hidden_layers', type=int, default=3,
                             help='Number of hidden layers in prediction MLP')
    model_group.add_argument('--prediction_hidden_dim', type=int, default=128,
                             help='Hidden dimension for all layers in prediction MLP')

    model_params_group = parser.add_argument_group('Model Parameters')
    model_params_group.add_argument('--hidden_channels', type=int, default=128, help='Hidden channels in the model')
    model_params_group.add_argument('--num_blocks', type=int, default=4, help='Number of interaction blocks')
    model_params_group.add_argument('--int_emb_size', type=int, default=64, help='Interaction embedding size')
    model_params_group.add_argument('--basis_emb_size', type=int, default=8, help='Basis embedding size')
    model_params_group.add_argument('--out_emb_channels', type=int, default=256, help='Output embedding channels')
    model_params_group.add_argument('--num_spherical', type=int, default=7, help='Number of spherical harmonics')
    model_params_group.add_argument('--num_radial', type=int, default=6, help='Number of radial basis functions')
    model_params_group.add_argument('--cutoff', type=float, default=5.0,
                                    help='Cutoff distance for neighbor calculation')
    model_params_group.add_argument('--envelope_exponent', type=int, default=5, help='Envelope exponent')
    model_params_group.add_argument('--num_before_skip', type=int, default=1,
                                    help='Number of layers before skip connection')
    model_params_group.add_argument('--num_after_skip', type=int, default=2,
                                    help='Number of layers after skip connection')
    model_params_group.add_argument('--num_output_layers', type=int, default=3, help='Number of output layers')
    model_params_group.add_argument('--max_num_neighbors', type=int, default=32, help='Maximum number of neighbors')

    readout_group = parser.add_argument_group('Readout Parameters')
    readout_group.add_argument('--set_transformer_hidden_dim', type=int, default=512,
                               help='Hidden dimension for set transformer')
    readout_group.add_argument('--set_transformer_num_heads', type=int, default=16, help='Number of attention heads')
    readout_group.add_argument('--set_transformer_num_sabs', type=int, default=2, help='Number of Set Attention Blocks')
    readout_group.add_argument('--attention_hidden_dim', type=int, default=256, help='Hidden dimension for attention')
    readout_group.add_argument('--attention_num_heads', type=int, default=8, help='Number of attention heads')

    train_group = parser.add_argument_group('Training Parameters')
    train_group.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    train_group.add_argument('--eval_batch_size', type=int, default=None, help='Batch size for evaluation')
    train_group.add_argument('--max_epochs', type=int, default=100, help='Maximum number of epochs')
    train_group.add_argument('--min_epochs', type=int, default=10, help='Minimum number of epochs')
    train_group.add_argument('--early_stopping_patience', type=int, default=20, help='Early stopping patience')
    train_group.add_argument('--early_stopping_min_delta', type=float, default=0.0001,
                             help='Minimum improvement for early stopping')
    train_group.add_argument('--loss_function', type=str, default='mse',
                             choices=['mse', 'mae', 'huber', 'smooth_l1', 'cross_entropy', 'binary_cross_entropy',
                                      'evidence_lower_bound'],
                             help='Loss function')
    train_group.add_argument('--target_weights', type=float, nargs='+', default=None,
                             help='Weights for each target in loss calculation')
    train_group.add_argument('--uncertainty_method', type=str, default=None,
                             choices=[None, 'ensemble', 'dropout', 'evidential'])
    train_group.add_argument('--gradient_clip_val', type=float, default=0.0, help='Gradient clipping value')
    train_group.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Gradient accumulation steps')

    optim_group = parser.add_argument_group('Optimization Parameters')
    optim_group.add_argument('--optimizer', type=str, default='adam',
                             choices=['adam', 'adamw', 'sgd', 'rmsprop', 'adagrad'],
                             help='Optimizer')
    optim_group.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    optim_group.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')
    optim_group.add_argument('--scheduler', type=str, default='cosine',
                             choices=['cosine', 'step', 'exponential', 'plateau', 'warmup_cosine',
                                      'cyclic', 'one_cycle', 'constant', 'warmup_constant'],
                             help='LR scheduler')
    optim_group.add_argument('--warmup_epochs', type=int, default=10, help='Warmup epochs')
    optim_group.add_argument('--scheduler_patience', type=int, default=5, help='Scheduler patience')
    optim_group.add_argument('--scheduler_factor', type=float, default=0.5, help='Scheduler factor')
    optim_group.add_argument('--min_lr', type=float, default=1e-6, help='Minimum learning rate')

    output_group = parser.add_argument_group('Output Parameters')
    output_group.add_argument('--out_dir', type=str, default='./results', help='Output directory')
    output_group.add_argument('--experiment_name', type=str, default=None, help='Experiment name')
    output_group.add_argument('--save_best_model', action='store_true', default=True, help='Save best model')
    output_group.add_argument('--save_last_model', action='store_true', default=False, help='Save last model')
    output_group.add_argument('--save_predictions', action='store_true', default=True, help='Save predictions')
    output_group.add_argument('--save_interval', type=int, default=0, help='Save checkpoints every N epochs')
    output_group.add_argument('--ckpt_path', type=str, default=None, help='Path to checkpoint')

    dist_group = parser.add_argument_group('Distributed Training Parameters')
    dist_group.add_argument('--strategy', type=str, default='auto',
                            choices=['auto', 'ddp', 'deepspeed', 'fsdp', 'none'])
    dist_group.add_argument('--num_nodes', type=int, default=1, help='Number of nodes')
    dist_group.add_argument('--devices', type=int, default=1, help='Number of devices per node')

    logging_group = parser.add_argument_group('Logging Parameters')
    logging_group.add_argument('--log_level', type=str, default='info',
                               choices=['debug', 'info', 'warning', 'error', 'critical'])
    logging_group.add_argument('--log_to_file', action='store_true', default=False, help='Log to file')
    logging_group.add_argument('--log_every_n_steps', type=int, default=50, help='Log every N steps')
    logging_group.add_argument('--logger_type', type=str, default='tensorboard',
                               choices=['tensorboard', 'wandb', 'csv', 'all'])
    logging_group.add_argument('--progress_bar', action='store_true', default=True, help='Show progress bar')

    dataset_specific_group = parser.add_argument_group('Dataset-Specific Parameters')
    dataset_specific_group.add_argument('--reaction_dataset_root', type=str, default=None)
    dataset_specific_group.add_argument('--reaction_target_fields', type=str, nargs='+', default=None,
                                        help='Target field(s) to predict')
    dataset_specific_group.add_argument('--reaction_file_suffixes', type=str, nargs=3,
                                        default=['_reactant.xyz', '_ts.xyz', '_product.xyz'])
    dataset_specific_group.add_argument('--input_features', type=str, nargs='+',
                                        default=['G(TS)_xtb', 'DrG_xtb'],
                                        help='Input feature columns to read from CSV')

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
    hyperopt_group.add_argument('--metric_for_best', type=str, default='val_mae',
                                help='Metric to use for selecting best hyperparameters')
    hyperopt_group.add_argument('--metric_mode', type=str, default='min', choices=['min', 'max'],
                                help='Whether to minimize or maximize the metric')
    hyperopt_group.add_argument('--clean_data', action='store_true', default=True,
                                help='Clean processed data directory before loading')

    return parser


def parse_command_line_args():
    parser = get_parser()
    args = parser.parse_args()

    args = process_derived_args(args)

    return args


def process_derived_args(args):
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

    return args


def create_experiment_name(params):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"hyperopt_cut{params['cutoff']}_blk{params['num_blocks']}_" \
           f"phl{params['prediction_hidden_layers']}_phd{params['prediction_hidden_dim']}_{timestamp}"
    return name


def setup_logging(args):
    os.makedirs(args.output_dir, exist_ok=True)

    log_level = getattr(logging, args.log_level.upper())

    logger = logging.getLogger('deepreaction')
    logger.setLevel(log_level)

    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if args.log_to_file:
        log_file = os.path.join(args.output_dir, 'training.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def setup_loggers(args, output_dir=None):
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


def create_dataloader(dataset, batch_size, eval_mode=False, num_workers=4, **kwargs):
    loader_kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'worker_init_fn': lambda worker_id: np.random.seed(torch.initial_seed() % 2 ** 32) if not eval_mode else None,
        'shuffle': not eval_mode,
        **kwargs
    }

    return GeometricDataLoader(dataset, **loader_kwargs)


def save_config(args, output_dir):
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


def run_single_fold(fold_idx, fold_data, args, logger):
    from train import create_model, train_model, evaluate_model

    fold_dir = os.path.join(args.output_dir, f"fold_{fold_idx}")
    os.makedirs(fold_dir, exist_ok=True)

    logger.info(f"=== Processing fold {fold_idx + 1}/{args.cv_folds} ===")

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

    model = create_model(args, scalers)

    fold_loggers = setup_loggers(args, fold_dir)
    fold_callbacks = setup_callbacks(args, fold_dir)

    trainer, train_metrics = train_model(
        args, model, train_loader, val_loader, fold_loggers, fold_callbacks, logger
    )

    test_metrics = evaluate_model(
        trainer, model, test_loader, fold_dir, args.save_predictions,
        args.reaction_target_fields, logger
    )

    val_metrics = {}
    test_metrics_processed = {}

    if hasattr(model, 'val_metrics') and 'val_mae' in model.val_metrics:
        val_metrics['val_mae'] = model.val_metrics['val_mae']
        val_metrics['val_rmse'] = model.val_metrics.get('val_rmse', float('inf'))
    else:
        for k, v in trainer.callback_metrics.items():
            if k.startswith('Validation'):
                metric_name = k.lower().replace(' ', '_')
                if 'mae' in metric_name:
                    val_metrics['val_mae'] = v.item() if hasattr(v, 'item') else float(v)
                elif 'rmse' in metric_name:
                    val_metrics['val_rmse'] = v.item() if hasattr(v, 'item') else float(v)

    if 'val_mae' not in val_metrics:
        for k, v in test_metrics.items():
            if 'Test Avg MAE' in k:
                val_metrics['val_mae'] = float(v)
            elif 'Test Avg RMSE' in k:
                val_metrics['val_rmse'] = float(v)

    if 'val_mae' not in val_metrics:
        val_metrics['val_mae'] = test_metrics.get('Test Avg MAE',
                                                  test_metrics.get('test_total_loss', 999.0))
        val_metrics['val_rmse'] = test_metrics.get('Test Avg RMSE',
                                                   test_metrics.get('test_total_loss', 999.0))

    if hasattr(model, 'test_metrics') and 'test_mae' in model.test_metrics:
        test_metrics_processed['test_mae'] = model.test_metrics['test_mae']
        test_metrics_processed['test_rmse'] = model.test_metrics.get('test_rmse', float('inf'))
    else:
        for k, v in test_metrics.items():
            if 'Test Avg MAE' in k:
                test_metrics_processed['test_mae'] = float(v)
            elif 'Test Avg RMSE' in k:
                test_metrics_processed['test_rmse'] = float(v)

    if 'test_mae' not in test_metrics_processed:
        test_metrics_processed['test_mae'] = test_metrics.get('Test Avg MAE',
                                                              test_metrics.get('test_total_loss', 999.0))
        test_metrics_processed['test_rmse'] = test_metrics.get('Test Avg RMSE',
                                                               test_metrics.get('test_total_loss', 999.0))

    fold_metrics = {
        'train': train_metrics,
        'val': val_metrics,
        'test': test_metrics_processed,
        'fold': fold_idx
    }

    with open(os.path.join(fold_dir, 'metrics.json'), 'w') as f:
        json.dump(fold_metrics, f, indent=2, cls=HyperoptEncoder)

    return fold_metrics


def train_function(config, checkpoint_dir=None):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(os.path.dirname(script_dir))
    deep_dir = os.path.dirname(script_dir)

    sys.path.insert(0, parent_dir)
    sys.path.insert(0, deep_dir)
    sys.path.insert(0, script_dir)

    if torch.cuda.is_available() and ray.get_gpu_ids():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(ray.get_gpu_ids()[0])
        torch.cuda.set_device(0)

    base_args = config.pop("args")
    args = copy.deepcopy(base_args)

    for param, value in config.items():
        if param != "args":
            setattr(args, param, value)

    experiment_name = create_experiment_name({
        'cutoff': args.cutoff,
        'num_blocks': args.num_blocks,
        'prediction_hidden_layers': args.prediction_hidden_layers,
        'prediction_hidden_dim': args.prediction_hidden_dim
    })

    args.experiment_name = experiment_name
    args.output_dir = os.path.join(args.out_dir, experiment_name)
    os.makedirs(args.output_dir, exist_ok=True)

    log_level = getattr(logging, args.log_level.upper())
    logger = logging.getLogger('deepreaction')
    logger.setLevel(log_level)

    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    save_config(args, args.output_dir)

    pl.seed_everything(args.random_seed)

    from data.load_Reaction import load_reaction

    logger.info(f"Loading dataset for {args.cv_folds}-fold CV with hyperparameters: cutoff={args.cutoff}, " +
                f"num_blocks={args.num_blocks}, prediction_hidden_layers={args.prediction_hidden_layers}, " +
                f"prediction_hidden_dim={args.prediction_hidden_dim}")

    # Clean up processed directory to avoid corrupted files
    if hasattr(args, 'clean_data') and args.clean_data:
        clean_processed_dir(args.reaction_dataset_root)

    try:
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
        logger.error(f"Error loading dataset: {e}")
        logger.info("Cleaning processed directory and trying again...")
        clean_processed_dir(args.reaction_dataset_root)

        # Try again with force_reload=True
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

    all_fold_metrics = []
    val_mae_values = []
    val_rmse_values = []
    test_mae_values = []
    test_rmse_values = []

    for fold_idx, fold_data in enumerate(fold_datasets):
        try:
            fold_metrics = run_single_fold(fold_idx, fold_data, args, logger)
            all_fold_metrics.append(fold_metrics)

            if 'val' in fold_metrics:
                if 'val_mae' in fold_metrics['val']:
                    val_mae_values.append(fold_metrics['val']['val_mae'])
                if 'val_rmse' in fold_metrics['val']:
                    val_rmse_values.append(fold_metrics['val']['val_rmse'])

            if 'test' in fold_metrics:
                if 'test_mae' in fold_metrics['test']:
                    test_mae_values.append(fold_metrics['test']['test_mae'])
                if 'test_rmse' in fold_metrics['test']:
                    test_rmse_values.append(fold_metrics['test']['test_rmse'])

        except Exception as e:
            logger.error(f"Error in fold {fold_idx}: {e}")
            import traceback
            logger.error(traceback.format_exc())

    avg_val_mae = np.mean(val_mae_values) if val_mae_values else 999.0
    avg_val_rmse = np.mean(val_rmse_values) if val_rmse_values else 999.0
    avg_test_mae = np.mean(test_mae_values) if test_mae_values else 999.0
    avg_test_rmse = np.mean(test_rmse_values) if test_rmse_values else 999.0

    avg_metrics = {
        'val_mae': avg_val_mae,
        'val_rmse': avg_val_rmse,
        'test_mae': avg_test_mae,
        'test_rmse': avg_test_rmse
    }

    result_summary = {
        'hyperparameters': {
            'cutoff': args.cutoff,
            'num_blocks': args.num_blocks,
            'prediction_hidden_layers': args.prediction_hidden_layers,
            'prediction_hidden_dim': args.prediction_hidden_dim
        },
        'avg_metrics': avg_metrics,
        'fold_metrics': all_fold_metrics
    }

    with open(os.path.join(args.output_dir, 'trial_summary.json'), 'w') as f:
        json.dump(result_summary, f, indent=2, cls=HyperoptEncoder)

    logger.info(f"Trial completed. Validation metrics collected: {len(val_mae_values)}/{args.cv_folds} folds")
    logger.info(f"Average validation MAE: {avg_val_mae:.6f}, Average validation RMSE: {avg_val_rmse:.6f}")
    logger.info(f"Average test MAE: {avg_test_mae:.6f}, Average test RMSE: {avg_test_rmse:.6f}")

    tune.report(
        {"val_mae": avg_val_mae, "val_rmse": avg_val_rmse, "test_mae": avg_test_mae, "test_rmse": avg_test_rmse})


def main():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)

    args = parse_command_line_args()
    os.makedirs(args.out_dir, exist_ok=True)

    logger = setup_logging(args)
    logger.info("Starting hyperparameter optimization with Ray Tune")

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    logger.info(f"Found {num_gpus} GPUs for parallel training")

    # Clean processed data directory before starting
    if args.clean_data:
        logger.info("Cleaning processed data directory before starting hyperparameter search")
        clean_processed_dir(args.reaction_dataset_root)

    param_space = {
        "cutoff": tune.grid_search(args.cutoff_values),
        "num_blocks": tune.grid_search(args.num_blocks_values),
        "prediction_hidden_layers": tune.grid_search(args.prediction_hidden_layers_values),
        "prediction_hidden_dim": tune.grid_search(args.prediction_hidden_dim_values),
        "args": args
    }

    ray.init(num_cpus=args.parallel_jobs, num_gpus=num_gpus, include_dashboard=False)

    storage_path = os.path.join(args.out_dir, "ray_results")
    os.makedirs(storage_path, exist_ok=True)

    metric = args.metric_for_best
    mode = args.metric_mode

    search_alg = BasicVariantGenerator(points_to_evaluate=None)

    start_time = time.time()

    analysis = tune.run(
        train_function,
        config=param_space,
        resources_per_trial={"cpu": 1, "gpu": 1 if num_gpus > 0 else 0},
        metric=metric,
        mode=mode,
        storage_path=storage_path,
        name=f"hyperopt_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        verbose=1,
        max_failures=3,
        fail_fast=False,
        num_samples=1,
        search_alg=search_alg
    )

    total_time = time.time() - start_time

    try:
        best_trial = analysis.get_best_trial(metric=metric, mode=mode)
        best_config = best_trial.config
        best_result = best_trial.last_result

        best_hyperparams = {
            'cutoff': best_config['cutoff'],
            'num_blocks': best_config['num_blocks'],
            'prediction_hidden_layers': best_config['prediction_hidden_layers'],
            'prediction_hidden_dim': best_config['prediction_hidden_dim']
        }

        overall_results = {
            'total_time': total_time,
            'num_combinations': len(analysis.trials),
            'best_hyperparams': best_hyperparams,
            'best_metrics': {k: v for k, v in best_result.items() if
                             k in ['val_mae', 'val_rmse', 'test_mae', 'test_rmse']},
            'all_trials': {
                t.trial_id: {
                    'config': {
                        'cutoff': t.config['cutoff'],
                        'num_blocks': t.config['num_blocks'],
                        'prediction_hidden_layers': t.config['prediction_hidden_layers'],
                        'prediction_hidden_dim': t.config['prediction_hidden_dim']
                    },
                    'result': {k: v for k, v in (t.last_result or {}).items() if
                               k in ['val_mae', 'val_rmse', 'test_mae', 'test_rmse']}
                } for t in analysis.trials
            }
        }
    except Exception as e:
        logger.error(f"Error determining best trial: {e}")
        overall_results = {
            'total_time': total_time,
            'error': str(e),
            'all_trials': {
                t.trial_id: {
                    'config': {
                        'cutoff': t.config.get('cutoff'),
                        'num_blocks': t.config.get('num_blocks'),
                        'prediction_hidden_layers': t.config.get('prediction_hidden_layers'),
                        'prediction_hidden_dim': t.config.get('prediction_hidden_dim')
                    },
                    'result': {k: v for k, v in (t.last_result or {}).items() if
                               k in ['val_mae', 'val_rmse', 'test_mae', 'test_rmse']}
                } for t in analysis.trials
            }
        }

    results_path = os.path.join(args.out_dir, 'hyperopt_results.json')
    with open(results_path, 'w') as f:
        json.dump(overall_results, f, indent=2, cls=HyperoptEncoder)

    logger.info(f"=== Ray Tune Hyperparameter Optimization Completed ===")
    logger.info(f"Total time: {total_time:.2f} seconds")
    logger.info(f"Results saved to: {results_path}")

    if 'best_hyperparams' in overall_results:
        logger.info(f"Best hyperparameters: {overall_results['best_hyperparams']}")
        if 'best_metrics' in overall_results and overall_results['best_metrics']:
            for metric, value in overall_results['best_metrics'].items():
                logger.info(f"{metric}: {value}")

    ray.shutdown()

    return 0


if __name__ == "__main__":
    sys.exit(main())