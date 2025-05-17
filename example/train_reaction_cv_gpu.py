#!/usr/bin/env python
import os
import sys
import argparse
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import json
import ray
import time
import gc
from typing import Dict, Any, List

from deepreaction import ReactionDataset, ReactionTrainer
from deepreaction.config import ReactionConfig, ModelConfig, TrainingConfig, Config, save_config

def get_parser():
    parser = argparse.ArgumentParser(description='Train a molecular reaction prediction model with N-fold cross-validation (multi-GPU)')
    
    parser.add_argument('--dataset', type=str, default='XTB', help='Dataset name')
    parser.add_argument('--readout', type=str, default='mean', help='Readout function')
    parser.add_argument('--dataset_root', type=str, default='./dataset/DATASET_DA_F', help='Dataset root directory')
    parser.add_argument('--dataset_csv', type=str, default='./dataset/DATASET_DA_F/dataset_xtb_final.csv', help='Dataset CSV file (relative to dataset_root)')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Validation set ratio')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='Test set ratio')
    parser.add_argument('--target_fields', type=str, nargs='+', default=['G(TS)', 'DrG'], help='Target fields')
    parser.add_argument('--target_weights', type=float, nargs='+', default=[1.0, 1.0], help='Target weights')
    parser.add_argument('--input_features', type=str, nargs='+', default=['G(TS)_xtb', 'DrG_xtb'], help='Input features')
    parser.add_argument('--file_patterns', type=str, nargs='+', default=['*_reactant.xyz', '*_ts.xyz', '*_product.xyz'], help='File patterns')
    parser.add_argument('--file_dir_pattern', type=str, default='reaction_*', help='Directory pattern')
    parser.add_argument('--id_field', type=str, default='ID', help='ID field name')
    parser.add_argument('--dir_field', type=str, default='R_dir', help='Directory field name')
    parser.add_argument('--reaction_field', type=str, default='reaction', help='Reaction field name')
    parser.add_argument('--cv_folds', type=int, default=10, help='Number of cross-validation folds')
    
    parser.add_argument('--model_type', type=str, default='dimenet++', help='Model type')
    parser.add_argument('--node_dim', type=int, default=128, help='Node dimension')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--prediction_hidden_layers', type=int, default=3, help='Number of prediction hidden layers')
    parser.add_argument('--prediction_hidden_dim', type=int, default=512, help='Prediction hidden dimension')
    parser.add_argument('--use_layer_norm', action='store_true', help='Use layer normalization')
    
    parser.add_argument('--hidden_channels', type=int, default=128, help='Hidden channels')
    parser.add_argument('--num_blocks', type=int, default=5, help='Number of blocks')
    parser.add_argument('--int_emb_size', type=int, default=64, help='Interaction embedding size')
    parser.add_argument('--basis_emb_size', type=int, default=8, help='Basis embedding size')
    parser.add_argument('--out_emb_channels', type=int, default=256, help='Output embedding channels')
    parser.add_argument('--num_spherical', type=int, default=7, help='Number of spherical harmonics')
    parser.add_argument('--num_radial', type=int, default=6, help='Number of radial basis functions')
    parser.add_argument('--cutoff', type=float, default=5.0, help='Cutoff distance')
    parser.add_argument('--envelope_exponent', type=int, default=5, help='Envelope exponent')
    parser.add_argument('--num_before_skip', type=int, default=1, help='Number of layers before skip')
    parser.add_argument('--num_after_skip', type=int, default=2, help='Number of layers after skip')
    parser.add_argument('--num_output_layers', type=int, default=3, help='Number of output layers')
    parser.add_argument('--max_num_neighbors', type=int, default=32, help='Maximum number of neighbors')
    
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--eval_batch_size', type=int, default=None, help='Evaluation batch size')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--finetune_lr', type=float, default=None, help='Fine-tuning learning rate (defaults to 10% of normal lr)')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--min_epochs', type=int, default=0, help='Minimum number of epochs')
    parser.add_argument('--early_stopping', type=int, default=40, help='Early stopping patience')
    parser.add_argument('--optimizer', type=str, default='adamw', help='Optimizer')
    parser.add_argument('--scheduler', type=str, default='warmup_cosine', help='Learning rate scheduler')
    parser.add_argument('--warmup_epochs', type=int, default=10, help='Warmup epochs')
    parser.add_argument('--min_lr', type=float, default=1e-7, help='Minimum learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay')
    parser.add_argument('--random_seed', type=int, default=42234, help='Random seed')
    
    parser.add_argument('--out_dir', type=str, default='./results/reaction_model_cv', help='Output directory')
    parser.add_argument('--save_best_model', action='store_true', default=True, help='Save best model')
    parser.add_argument('--save_last_model', action='store_true', default=False, help='Save last model')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Checkpoint path for resuming training')
    parser.add_argument('--mode', type=str, default='continue', choices=['continue', 'finetune'], 
                        help='Mode for checkpoint loading: continue training or finetune')
    parser.add_argument('--freeze_base_model', action='store_true', help='Freeze base model when fine-tuning')
    
    parser.add_argument('--cuda', action='store_true', default=True, help='Use CUDA')
    parser.add_argument('--no-cuda', action='store_false', dest='cuda', help='Do not use CUDA')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--num_gpus', type=int, default=None, help='Number of GPUs to use (default: use all available)')
    parser.add_argument('--gpus_per_fold', type=int, default=1, help='Number of GPUs to use per fold')
    
    return parser

@ray.remote(num_gpus=1)
def train_fold(fold_idx, args_dict, dataset_dict, out_dir, scalers):
    """
    Function to train a single fold, intended to be executed as a Ray remote function.
    
    Args:
        fold_idx: The fold index to train
        args_dict: Dictionary of command line arguments
        dataset_dict: Dictionary with train, val, test datasets
        out_dir: Output directory for this fold
        scalers: Data scalers
        
    Returns:
        Dictionary with fold training results
    """
    # Create fold-specific output directory
    fold_out_dir = os.path.join(out_dir, f"fold_{fold_idx}")
    os.makedirs(fold_out_dir, exist_ok=True)
    
    # Get dataset for this fold
    train_data = dataset_dict['train']
    val_data = dataset_dict['val']
    test_data = dataset_dict['test']
    
    # Setup GPU for this process
    gpu_id = ray.get_gpu_ids()[0]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device(f"cuda:{0}")  # Just use cuda:0 since we've set CUDA_VISIBLE_DEVICES
    print(f"Fold {fold_idx+1}: Using GPU {gpu_id}: {torch.cuda.get_device_name(device)}")
    
    # Prepare additional kwargs
    additional_kwargs = {}
    if args_dict.get('finetune_lr') is not None:
        additional_kwargs['finetune_lr'] = args_dict.get('finetune_lr')
    if args_dict.get('freeze_base_model', False):
        additional_kwargs['freeze_base_model'] = True
    
    # Create trainer for this fold
    trainer = ReactionTrainer(
        model_type=args_dict.get('model_type'),
        readout=args_dict.get('readout'),
        batch_size=args_dict.get('batch_size'),
        max_epochs=args_dict.get('epochs'),
        learning_rate=args_dict.get('lr'),
        output_dir=fold_out_dir,
        early_stopping_patience=args_dict.get('early_stopping'),
        save_best_model=args_dict.get('save_best_model'),
        save_last_model=args_dict.get('save_last_model'),
        random_seed=args_dict.get('random_seed') + fold_idx,  # Different seed for each fold
        num_targets=len(args_dict.get('target_fields', [])),
        use_scaler=True,
        scalers=scalers,
        optimizer=args_dict.get('optimizer'),
        weight_decay=args_dict.get('weight_decay'),
        scheduler=args_dict.get('scheduler'),
        warmup_epochs=args_dict.get('warmup_epochs'),
        min_lr=args_dict.get('min_lr'),
        gpu=True,  # Always use GPU in this function
        node_dim=args_dict.get('node_dim'),
        dropout=args_dict.get('dropout'),
        use_layer_norm=args_dict.get('use_layer_norm'),
        target_field_names=args_dict.get('target_fields'),
        use_xtb_features=len(args_dict.get('input_features', [])) > 0,
        num_xtb_features=len(args_dict.get('input_features', [])),
        prediction_hidden_layers=args_dict.get('prediction_hidden_layers'),
        prediction_hidden_dim=args_dict.get('prediction_hidden_dim'),
        min_epochs=args_dict.get('min_epochs'),
        hidden_channels=args_dict.get('hidden_channels'),
        num_blocks=args_dict.get('num_blocks'),
        cutoff=args_dict.get('cutoff'),
        int_emb_size=args_dict.get('int_emb_size'),
        basis_emb_size=args_dict.get('basis_emb_size'),
        out_emb_channels=args_dict.get('out_emb_channels'),
        num_spherical=args_dict.get('num_spherical'),
        num_radial=args_dict.get('num_radial'),
        envelope_exponent=args_dict.get('envelope_exponent'),
        num_before_skip=args_dict.get('num_before_skip'),
        num_after_skip=args_dict.get('num_after_skip'),
        num_output_layers=args_dict.get('num_output_layers'),
        max_num_neighbors=args_dict.get('max_num_neighbors'),
        num_workers=args_dict.get('num_workers'),
        **additional_kwargs
    )
    
    # Train model for this fold
    print(f"Training model for fold {fold_idx+1}")
    train_metrics = trainer.fit(
        train_dataset=train_data,
        val_dataset=val_data,
        test_dataset=test_data,
        run_dir=fold_out_dir
    )
    
    # Evaluate model on test set
    test_metrics = None
    if trainer.trainer is not None and test_data is not None and len(test_data) > 0:
        from torch_geometric.loader import DataLoader
        follow_batch = ['z0', 'z1', 'z2', 'pos0', 'pos1', 'pos2']
        test_loader = DataLoader(
            test_data,
            batch_size=args_dict.get('batch_size'),
            shuffle=False,
            num_workers=args_dict.get('num_workers'),
            follow_batch=follow_batch
        )
        
        print(f"Evaluating fold {fold_idx+1} on test set")
        test_results = trainer.trainer.test(trainer.lightning_model, test_loader)
        print(f"Fold {fold_idx+1} test metrics: {test_results}")
        
        if test_results and isinstance(test_results, list) and len(test_results) > 0:
            test_metrics = test_results[0]
    
    # Save fold results
    fold_results = {
        'fold': fold_idx,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics
    }
    
    # Save fold results to file
    fold_results_file = os.path.join(fold_out_dir, 'fold_results.json')
    with open(fold_results_file, 'w') as f:
        json.dump(fold_results, f, indent=2)
    
    # Clean up GPU memory
    del trainer
    torch.cuda.empty_cache()
    
    return fold_results

def main():
    parser = get_parser()
    args = parser.parse_args()
    
    # Initialize Ray
    if args.num_gpus is None:
        # Auto-detect number of GPUs
        ray.init()
    else:
        # Use specified number of GPUs
        ray.init(num_gpus=args.num_gpus)
    
    print(f"Ray initialized with {ray.available_resources()['GPU'] if 'GPU' in ray.available_resources() else 0} GPUs")
    
    # Get available GPU IDs
    available_gpus = ray.get_gpu_ids() if hasattr(ray, 'get_gpu_ids') else []
    if not available_gpus and args.cuda:
        # Fallback: check CUDA availability
        num_gpus = torch.cuda.device_count()
        if num_gpus > 0:
            available_gpus = list(range(num_gpus))
            print(f"Detected {num_gpus} GPUs via PyTorch")
        else:
            print("No GPUs detected, using CPU")
            args.cuda = False
    else:
        print(f"Using GPUs: {available_gpus}")
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Save the configuration
    reaction_config = ReactionConfig(
        dataset_root=args.dataset_root,
        dataset_csv=args.dataset_csv,
        target_fields=args.target_fields,
        file_patterns=args.file_patterns,
        input_features=args.input_features,
        use_scaler=True,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        cv_folds=args.cv_folds,
        cv_test_fold=-1,
        cv_stratify=False,
        cv_grouped=True,
        id_field=args.id_field,
        dir_field=args.dir_field,
        reaction_field=args.reaction_field,
        random_seed=args.random_seed
    )
    
    model_config = ModelConfig(
        model_type=args.model_type,
        readout=args.readout,
        hidden_channels=args.hidden_channels,
        num_blocks=args.num_blocks,
        cutoff=args.cutoff,
        int_emb_size=args.int_emb_size,
        basis_emb_size=args.basis_emb_size,
        out_emb_channels=args.out_emb_channels,
        num_spherical=args.num_spherical,
        num_radial=args.num_radial,
        envelope_exponent=args.envelope_exponent,
        num_before_skip=args.num_before_skip,
        num_after_skip=args.num_after_skip,
        num_output_layers=args.num_output_layers,
        max_num_neighbors=args.max_num_neighbors,
        node_dim=args.node_dim,
        dropout=args.dropout,
        use_layer_norm=args.use_layer_norm,
        use_xtb_features=len(args.input_features) > 0,
        num_xtb_features=len(args.input_features),
        prediction_hidden_layers=args.prediction_hidden_layers,
        prediction_hidden_dim=args.prediction_hidden_dim
    )
    
    training_config = TrainingConfig(
        output_dir=args.out_dir,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_epochs=args.epochs,
        min_epochs=args.min_epochs,
        early_stopping_patience=args.early_stopping,
        save_best_model=args.save_best_model,
        save_last_model=args.save_last_model,
        optimizer=args.optimizer,
        weight_decay=args.weight_decay,
        scheduler=args.scheduler,
        warmup_epochs=args.warmup_epochs,
        min_lr=args.min_lr,
        target_weights=args.target_weights,
        gpu=args.cuda,
        num_workers=args.num_workers,
        resume_from_checkpoint=args.checkpoint_path
    )
    
    config = Config(
        reaction=reaction_config,
        model=model_config,
        training=training_config
    )
    
    config_path = os.path.join(args.out_dir, 'config')
    save_config(config, config_path)
    print(f"Configuration saved to {config_path}.yaml and {config_path}.json")
    
    # Load dataset
    print(f"Loading dataset from {args.dataset_root}")
    dataset = ReactionDataset(
        root=args.dataset_root,
        csv_file=args.dataset_csv,
        target_fields=args.target_fields,
        file_patterns=args.file_patterns,
        input_features=args.input_features,
        use_scaler=True,
        random_seed=args.random_seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        cv_folds=args.cv_folds,
        id_field=args.id_field,
        dir_field=args.dir_field,
        reaction_field=args.reaction_field
    )
    
    print("Dataset loaded successfully")
    data_stats = dataset.get_data_stats()
    print(f"Dataset stats: Train: {data_stats['train_size']}, Validation: {data_stats['val_size']}, Test: {data_stats['test_size']}")
    
    # Convert args to dictionary for serialization
    args_dict = vars(args)
    
    # Start parallel training for each fold
    num_folds = dataset.get_num_folds()
    print(f"Starting {num_folds}-fold cross-validation with Ray parallelization")
    
    # Submit Ray tasks for each fold
    fold_futures = []
    for fold_idx in range(num_folds):
        print(f"Preparing fold {fold_idx+1}/{num_folds}")
        
        # Set current fold in dataset
        dataset.set_fold(fold_idx)
        
        # Get data for this fold
        fold_data = {
            'train': dataset.train_data,
            'val': dataset.val_data,
            'test': dataset.test_data
        }
        
        # Get scalers
        scalers = dataset.get_scalers()
        
        # Submit task
        future = train_fold.remote(
            fold_idx=fold_idx,
            args_dict=args_dict,
            dataset_dict=fold_data,
            out_dir=args.out_dir,
            scalers=scalers
        )
        fold_futures.append(future)
    
    # Wait for all folds to complete and collect results
    start_time = time.time()
    cv_metrics = []
    completed_folds = 0
    
    # Process results as they complete
    while fold_futures:
        # Wait for the next future to complete
        done_futures, fold_futures = ray.wait(fold_futures, num_returns=1)
        
        # Get the result
        fold_result = ray.get(done_futures[0])
        cv_metrics.append(fold_result)
        
        # Track progress
        completed_folds += 1
        elapsed_time = time.time() - start_time
        avg_time_per_fold = elapsed_time / completed_folds
        remaining_folds = num_folds - completed_folds
        est_remaining_time = avg_time_per_fold * remaining_folds
        
        print(f"Completed fold {fold_result['fold']+1}/{num_folds} in {elapsed_time:.2f} sec. "
              f"Est. remaining time: {est_remaining_time:.2f} sec")
    
    # Compute combined metrics
    cv_test_losses = []
    
    for fold_result in cv_metrics:
        if fold_result['test_metrics'] and 'test_total_loss' in fold_result['test_metrics']:
            cv_test_losses.append(fold_result['test_metrics']['test_total_loss'])
    
    # Sort results by fold index
    cv_metrics = sorted(cv_metrics, key=lambda x: x['fold'])
    
    # Compute average metrics across all folds
    print("\n============ Cross-Validation Summary ============")
    if cv_test_losses:
        avg_test_loss = np.mean(cv_test_losses)
        std_test_loss = np.std(cv_test_losses)
        print(f"Average test loss: {avg_test_loss:.4f} ± {std_test_loss:.4f}")
    
    # Get all metrics from test results
    all_metrics = {}
    for fold_result in cv_metrics:
        if fold_result['test_metrics']:
            for key, value in fold_result['test_metrics'].items():
                if key not in all_metrics:
                    all_metrics[key] = []
                all_metrics[key].append(value)
    
    # Compute average and standard deviation for each metric
    for metric_name, values in all_metrics.items():
        if values:
            mean_value = np.mean(values)
            std_value = np.std(values)
            print(f"{metric_name}: {mean_value:.4f} ± {std_value:.4f}")
    
    # Save CV summary
    cv_summary = {
        'config': {
            'num_folds': args.cv_folds,
            'model_type': args.model_type,
            'readout': args.readout,
            'target_fields': args.target_fields,
            'input_features': args.input_features
        },
        'fold_metrics': cv_metrics,
        'average_metrics': {metric: {'mean': float(np.mean(values)), 'std': float(np.std(values))} 
                           for metric, values in all_metrics.items() if values}
    }
    
    cv_summary_file = os.path.join(args.out_dir, 'cv_summary.json')
    with open(cv_summary_file, 'w') as f:
        json.dump(cv_summary, f, indent=2)
    
    print(f"Cross-validation summary saved to: {cv_summary_file}")
    
    # Shut down Ray
    ray.shutdown()
    print("Ray shutdown complete")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())