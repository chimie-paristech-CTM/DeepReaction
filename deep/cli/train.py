#!/usr/bin/env python

import os
import sys
import time
import json
import numpy as np
import logging
import random
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd

import torch
import torch.nn as nn
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
    CSVLogger
)
from pytorch_lightning.strategies import (
    DDPStrategy
)

parent_path = str(Path(Path(__file__).parent.absolute()).parent.absolute())
sys.path.insert(0, parent_path)

from cli.config import process_args, save_config, setup_logging
from data.load_Reaction import load_reaction
from module.pl_wrap import Estimator
from utils.metrics import compute_regression_metrics
from utils.visualization import plot_predictions, plot_multi_target_predictions


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.float32):
            return float(obj)
        return json.JSONEncoder.default(self, obj)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def setup_distributed_training(args) -> Dict[str, Any]:
    strategy_config = {}

    if args.strategy == 'auto':
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

    strategy_config['num_nodes'] = args.num_nodes
    strategy_config['devices'] = args.devices if args.devices > 0 else "auto"

    return strategy_config


def setup_loggers(args, output_dir=None) -> List:
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


def setup_callbacks(args, output_dir=None) -> List:
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
        LearningRateMonitor(logging_interval='epoch'),
        Timer(duration=None)
    ])

    if args.progress_bar:
        callbacks.append(TQDMProgressBar(refresh_rate=10))

    return callbacks


def create_dataloader(dataset, batch_size, eval_mode=False, num_workers=4, **kwargs):
    loader_kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'worker_init_fn': seed_worker if not eval_mode else None,
        'shuffle': not eval_mode,
        **kwargs
    }

    return GeometricDataLoader(dataset, **loader_kwargs)


def load_datasets(args, logger):
    logger.info(f"Loading dataset: {args.dataset}")

    if args.dataset == 'XTB':
        result = load_reaction(
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
            val_csv=args.val_csv,
            test_csv=args.test_csv,
            cv_folds=args.cv_folds,
            cv_test_fold=args.cv_test_fold,
            cv_stratify=args.cv_stratify,
            cv_grouped=args.cv_grouped
        )
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    if args.cv_folds > 0:
        return result

    train, val, test, scalers = result

    if args.use_scaler and scalers is not None:
        logger.info(f"Dataset loaded with {len(scalers)} scalers")
        for i, scaler in enumerate(scalers):
            if hasattr(scaler, 'mean_'):
                logger.info(f"Scaler {i} mean: {scaler.mean_}")
            if hasattr(scaler, 'scale_'):
                logger.info(f"Scaler {i} scale: {scaler.scale_}")
    elif args.use_scaler:
        logger.warning("use_scaler is True but no scalers were returned")
    else:
        logger.info("No scalers being used (use_scaler=False)")

    follow_batch = ['z0', 'z1', 'z2', 'pos0', 'pos1', 'pos2'] if args.dataset == 'XTB' else None
    dataloader_kwargs = {'follow_batch': follow_batch} if follow_batch else {}

    train_loader = create_dataloader(
        train, args.batch_size, eval_mode=False,
        num_workers=args.num_workers, **dataloader_kwargs
    )

    val_loader = create_dataloader(
        val, args.eval_batch_size, eval_mode=True,
        num_workers=args.num_workers, **dataloader_kwargs
    )

    test_loader = create_dataloader(
        test, args.eval_batch_size, eval_mode=True,
        num_workers=args.num_workers, **dataloader_kwargs
    )

    logger.info(f"Dataset loaded: {len(train)} train, {len(val)} validation, {len(test)} test samples")

    return train_loader, val_loader, test_loader, scalers


def create_model(args, scalers) -> pl.LightningModule:
    logger = logging.getLogger('deep')

    scalers_to_use = scalers if args.use_scaler else None
    num_targets = len(args.reaction_target_fields) if args.reaction_target_fields else 1
    num_features = len(args.input_features) if args.input_features else 2

    if scalers_to_use is not None:
        logger.info(f"Creating model with {len(scalers_to_use)} scalers for {num_targets} targets")
    else:
        logger.info(f"Creating model for {num_targets} targets without scalers")

    model_kwargs = {
        'hidden_channels': args.hidden_channels,
        'num_blocks': args.num_blocks,
        'int_emb_size': args.int_emb_size,
        'basis_emb_size': args.basis_emb_size,
        'out_emb_channels': args.out_emb_channels,
        'num_spherical': args.num_spherical,
        'num_radial': args.num_radial,
        'cutoff': args.cutoff,
        'max_num_neighbors': args.max_num_neighbors,
        'envelope_exponent': args.envelope_exponent,
        'num_before_skip': args.num_before_skip,
        'num_after_skip': args.num_after_skip,
        'num_output_layers': args.num_output_layers,
    }

    readout_kwargs = {
        'hidden_dim': args.attention_hidden_dim if args.readout == 'attention' else args.set_transformer_hidden_dim,
        'num_heads': args.attention_num_heads if args.readout == 'attention' else args.set_transformer_num_heads,
        'num_sabs': args.set_transformer_num_sabs,
        'layer_norm': args.use_layer_norm
    }

    model_config = {
        'model_type': args.model_type,
        'readout': args.readout,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'max_num_atoms_in_mol': args.max_num_atoms,
        'scaler': scalers_to_use,
        'use_layer_norm': args.use_layer_norm,
        'node_latent_dim': args.node_latent_dim,
        'edge_latent_dim': args.edge_latent_dim,
        'dropout': args.dropout,
        'model_kwargs': model_kwargs,
        'readout_kwargs': readout_kwargs,
        'optimizer': args.optimizer,
        'weight_decay': args.weight_decay,
        'scheduler': args.scheduler,
        'scheduler_patience': args.scheduler_patience,
        'scheduler_factor': args.scheduler_factor,
        'warmup_epochs': args.warmup_epochs,
        'min_lr': args.min_lr,
        'loss_function': args.loss_function,
        'target_weights': args.target_weights,
        'uncertainty_method': args.uncertainty_method,
        'gradient_clip_val': args.gradient_clip_val,
        'prediction_hidden_layers': args.prediction_hidden_layers,
        'prediction_hidden_dim': args.prediction_hidden_dim,
        'use_xtb_features': args.use_xtb_features,
        'num_xtb_features': num_features,
        'target_field_names': args.reaction_target_fields
    }

    return Estimator(**model_config)


def train_model(
        args,
        model: pl.LightningModule,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loggers: List,
        callbacks: List,
        logger: logging.Logger
) -> Tuple[pl.Trainer, Dict[str, Any]]:
    logger.info("Initializing training")

    distributed_config = setup_distributed_training(args)
    log_steps = min(10, len(train_loader))

    trainer_config = {
        'logger': loggers,
        'callbacks': callbacks,
        'max_epochs': args.max_epochs,
        'min_epochs': args.min_epochs,
        'log_every_n_steps': log_steps,
        'deterministic': True,
        'accelerator': 'gpu' if args.cuda and torch.cuda.is_available() else 'cpu',
        'num_sanity_val_steps': 2,
        'gradient_clip_val': args.gradient_clip_val if args.gradient_clip_val > 0 else None,
        'accumulate_grad_batches': args.gradient_accumulation_steps,
        **distributed_config
    }

    if args.precision in ['16', '32', 'bf16', 'mixed']:
        trainer_config['precision'] = args.precision

    if args.ckpt_path:
        trainer_config['resume_from_checkpoint'] = args.ckpt_path

    trainer = pl.Trainer(**trainer_config)

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


def save_predictions(y_pred, y_true, output_dir, target_fields, model):
    for epoch, predictions in y_pred.items():
        for i in range(predictions.shape[1]):
            target_name = target_fields[i] if target_fields and i < len(target_fields) else f"target_{i}"

            pred_values = predictions[:, i].reshape(-1, 1)
            if hasattr(model, 'scaler') and model.scaler is not None and i < len(model.scaler):
                pred_values = model.scaler[i].inverse_transform(pred_values)

            pd.DataFrame(pred_values).to_csv(
                os.path.join(output_dir, f'test_y_pred_{target_name}.csv'),
                index=False, header=[target_name]
            )

            truths = y_true.get(epoch, None)
            if truths is not None:
                truth_values = truths[:, i].reshape(-1, 1)
                if hasattr(model, 'scaler') and model.scaler is not None and i < len(model.scaler):
                    truth_values = model.scaler[i].inverse_transform(truth_values)

                pd.DataFrame(truth_values).to_csv(
                    os.path.join(output_dir, f'test_y_true_{target_name}.csv'),
                    index=False, header=[target_name]
                )

    for epoch, predictions in y_pred.items():
        np.save(os.path.join(output_dir, f'test_y_pred_epoch_{epoch}.npy'), predictions)

    for epoch, truths in y_true.items():
        np.save(os.path.join(output_dir, f'test_y_true_epoch_{epoch}.npy'), truths)

    if hasattr(model, 'scaler') and model.scaler is not None:
        for epoch, predictions in y_pred.items():
            unscaled_predictions = np.zeros_like(predictions)
            for i in range(predictions.shape[1]):
                if i < len(model.scaler):
                    unscaled_predictions[:, i:i + 1] = model.scaler[i].inverse_transform(predictions[:, i:i + 1])

            np.save(os.path.join(output_dir, f'test_y_pred_unscaled_epoch_{epoch}.npy'), unscaled_predictions)

        for epoch, truths in y_true.items():
            unscaled_truths = np.zeros_like(truths)
            for i in range(truths.shape[1]):
                if i < len(model.scaler):
                    unscaled_truths[:, i:i + 1] = model.scaler[i].inverse_transform(truths[:, i:i + 1])

            np.save(os.path.join(output_dir, f'test_y_true_unscaled_epoch_{epoch}.npy'), unscaled_truths)


def evaluate_model(
        trainer: pl.Trainer,
        model: pl.LightningModule,
        test_loader: DataLoader,
        output_dir: str,
        save_predictions_flag: bool,
        target_fields: List[str],
        logger: logging.Logger
) -> Dict[str, Any]:
    logger.info("Evaluating model on test set")
    test_results = trainer.test(model, dataloaders=test_loader)

    y_pred = model.test_output
    y_true = model.test_true

    if save_predictions_flag:
        save_predictions(y_pred, y_true, output_dir, target_fields, model)

    num_targets = len(model.test_output[1][0]) if len(model.test_output) > 0 else 0
    try:
        if num_targets > 1:
            plot_multi_target_predictions(y_true, y_pred, target_fields, output_dir)
        else:
            plot_predictions(y_true, y_pred, output_dir)
    except Exception as e:
        logger.warning(f"Failed to create prediction visualization: {e}")

    with open(os.path.join(output_dir, 'test_metrics.json'), 'w') as f:
        json.dump(model.test_metrics, f, indent=2, cls=NumpyEncoder)

    logger.info(f"Test metrics: {test_results[0]}")
    return test_results[0]


def save_fold_data_to_csv(fold_data, fold_idx, output_dir, logger):
    fold_dir = os.path.join(output_dir, f"fold_{fold_idx}")
    os.makedirs(fold_dir, exist_ok=True)

    scalers = fold_data.get('scalers', None)

    for split_name in ['train', 'val', 'test']:
        dataset = fold_data[split_name]

        data_records = []
        for i, data_item in enumerate(dataset):
            record = {'index': i}

            for attr in ['reaction_id', 'id', 'reaction']:
                if hasattr(data_item, attr):
                    record[attr] = getattr(data_item, attr)

            if hasattr(data_item, 'y'):
                if len(data_item.y.shape) == 2:
                    for j in range(data_item.y.shape[1]):
                        scaled_value = data_item.y[0, j].item()
                        record[f'y_{j}_scaled'] = scaled_value

                        if scalers and j < len(scalers):
                            unscaled_value = scalers[j].inverse_transform([[scaled_value]])[0, 0]
                            record[f'y_{j}'] = unscaled_value
                        else:
                            record[f'y_{j}'] = scaled_value
                else:
                    record['y_scaled'] = data_item.y.item()

                    if scalers and len(scalers) > 0:
                        unscaled_value = scalers[0].inverse_transform([[data_item.y.item()]])[0, 0]
                        record['y'] = unscaled_value
                    else:
                        record['y'] = data_item.y.item()

            if hasattr(data_item, 'xtb_features'):
                for j in range(data_item.xtb_features.shape[1]):
                    feature_name = f'feature_{j}'
                    if hasattr(data_item, 'feature_names') and j < len(data_item.feature_names):
                        feature_name = data_item.feature_names[j]

                    record[feature_name] = data_item.xtb_features[0, j].item()

            data_records.append(record)

        if data_records:
            df = pd.DataFrame(data_records)
            csv_path = os.path.join(fold_dir, f"{split_name}_metadata.csv")
            df.to_csv(csv_path, index=False)
            logger.info(f"Saved {split_name} metadata for fold {fold_idx} to {csv_path}")


def run_fold(fold_idx, fold_data, args, logger):
    fold_dir = os.path.join(args.output_dir, f"fold_{fold_idx}")
    os.makedirs(fold_dir, exist_ok=True)

    logger.info(f"=== Processing fold {fold_idx + 1}/{args.cv_folds} ===")

    save_fold_data_to_csv(fold_data, fold_idx, args.output_dir, logger)

    fold_loggers = setup_loggers(args, fold_dir)
    fold_callbacks = setup_callbacks(args, fold_dir)

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

    trainer, train_metrics = train_model(
        args, model, train_loader, val_loader, fold_loggers, fold_callbacks, logger
    )

    test_metrics = evaluate_model(
        trainer, model, test_loader, fold_dir, args.save_predictions,
        args.reaction_target_fields, logger
    )

    fold_metrics = {
        'train': train_metrics,
        'test': test_metrics
    }

    with open(os.path.join(fold_dir, 'metrics.json'), 'w') as f:
        json.dump(fold_metrics, f, indent=2, cls=NumpyEncoder)

    return test_metrics


def run_cross_validation(args, fold_datasets, logger):
    logger.info(f"Starting {args.cv_folds}-fold cross-validation")

    all_metrics = []
    fold_metrics = {}

    for fold_idx, fold_data in enumerate(fold_datasets):
        test_metrics = run_fold(fold_idx, fold_data, args, logger)

        fold_metrics[fold_idx] = test_metrics
        all_metrics.append(test_metrics)

    avg_metrics = {}
    std_metrics = {}

    if all_metrics and all_metrics[0]:
        metric_names = list(all_metrics[0].keys())

        for metric in metric_names:
            if 'MAE' in metric or 'RMSE' in metric or 'R2' in metric or 'loss' in metric:
                values = [metrics.get(metric, 0) for metrics in all_metrics]
                avg_metrics[metric] = float(np.mean(values))
                std_metrics[metric] = float(np.std(values))

    cv_summary = {
        'fold_metrics': fold_metrics,
        'avg_metrics': avg_metrics,
        'std_metrics': std_metrics
    }

    cv_summary_path = os.path.join(args.output_dir, 'cv_summary.json')
    with open(cv_summary_path, 'w') as f:
        json.dump(cv_summary, f, indent=2, cls=NumpyEncoder)

    logger.info("=== Cross-Validation Summary ===")
    for metric, value in avg_metrics.items():
        logger.info(f"{metric}: {value:.6f} ± {std_metrics[metric]:.6f}")

    logger.info(f"Cross-validation results saved to {cv_summary_path}")

    return cv_summary


def parse_command_line_args():
    from cli.config import process_args
    return process_args()


def main() -> None:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)

    args = parse_command_line_args()
    os.makedirs(args.output_dir, exist_ok=True)
    config_path = save_config(args, args.output_dir)

    logger = setup_logging(args)
    logger.info(f"Configuration saved to {config_path}")

    pl.seed_everything(args.random_seed)

    if args.cv_folds > 0:
        fold_datasets = load_datasets(args, logger)

        slurm_task_id = os.environ.get('SLURM_ARRAY_TASK_ID')
        if slurm_task_id is not None:
            fold_idx = int(slurm_task_id)
            if fold_idx < 0 or fold_idx >= args.cv_folds:
                logger.error(f"Invalid SLURM_ARRAY_TASK_ID: {fold_idx}, must be between 0 and {args.cv_folds - 1}")
                return

            logger.info(f"Running in SLURM array job, processing only fold {fold_idx}")

            fold_dir = os.path.join(args.output_dir, f"fold_{fold_idx}")
            os.makedirs(fold_dir, exist_ok=True)

            if fold_idx < len(fold_datasets):
                fold_data = fold_datasets[fold_idx]
                run_fold(fold_idx, fold_data, args, logger)
                logger.info(f"Fold {fold_idx} completed successfully")
            else:
                logger.error(f"Fold {fold_idx} not found in dataset")
        else:
            cv_summary = run_cross_validation(args, fold_datasets, logger)
            logger.info("Cross-validation completed successfully")
        return

    train_loader, val_loader, test_loader, scalers = load_datasets(args, logger)
    model = create_model(args, scalers)
    logger.info(f"Created model: {model.__class__.__name__}")

    loggers = setup_loggers(args)
    callbacks = setup_callbacks(args)

    trainer, train_metrics = train_model(
        args, model, train_loader, val_loader, loggers, callbacks, logger
    )

    test_metrics = evaluate_model(
        trainer, model, test_loader, args.output_dir, args.save_predictions,
        args.reaction_target_fields, logger
    )

    metrics = {
        'train': train_metrics,
        'test': test_metrics
    }

    with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2, cls=NumpyEncoder)

    logger.info("Training and evaluation completed successfully")
    logger.info(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()