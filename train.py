#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DeepReaction Training Script

This script demonstrates how to train a molecular reaction prediction model 
using the DeepReaction framework with a simplified unified configuration interface.
Converted from Jupyter notebook to standalone Python script.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

from deepreaction import ReactionTrainer, ReactionDataset, Config


def main():
    """Main function to run the training process"""
    
    # Define Training Parameters
    # All parameters are defined in a single dictionary for simplicity
    params = {
        # Dataset parameters
        'dataset': 'XTB',
        'readout': 'mean',
        'dataset_root': './dataset/DATASET_DA_F',  # Adjust path if needed
        'dataset_csv': './dataset/DATASET_DA_F/dataset_xtb_final.csv', # Adjust path if needed
        'train_ratio': 0.8,
        'val_ratio': 0.1,
        'test_ratio': 0.1,
        'target_fields': ['G(TS)', 'DrG'],
        'target_weights': [1.0, 1.0],
        'input_features': ['G(TS)_xtb', 'DrG_xtb'],
        'file_patterns': ['*_reactant.xyz', '*_ts.xyz', '*_product.xyz'],
        'file_dir_pattern': 'reaction_*',
        'id_field': 'ID',
        'dir_field': 'R_dir',
        'reaction_field': 'reaction',
        'cv_folds': 0, # Set > 0 for cross-validation
        'use_scaler': True,  # Controls whether to scale target values and pass scalers to trainer
        
        # Model parameters (DimeNet++ specific)
        'model_type': 'dimenet++',
        'node_dim': 128,
        'dropout': 0.1,
        'prediction_hidden_layers': 3,
        'prediction_hidden_dim': 512,
        'use_layer_norm': False,
        
        'hidden_channels': 128,
        'num_blocks': 5,
        'int_emb_size': 64,
        'basis_emb_size': 8,
        'out_emb_channels': 256,
        'num_spherical': 7,
        'num_radial': 6,
        'cutoff': 5.0,
        'envelope_exponent': 5,
        'num_before_skip': 1,
        'num_after_skip': 2,
        'num_output_layers': 3,
        'max_num_neighbors': 32,
        
        # Training parameters
        'batch_size': 16,
        'eval_batch_size': None, # Uses batch_size if None
        'lr': 0.0005,
        'finetune_lr': None,
        'epochs': 10,
        'min_epochs': 0,
        'early_stopping': 40,
        'optimizer': 'adamw',
        'scheduler': 'warmup_cosine',
        'warmup_epochs': 10,
        'min_lr': 1e-7,
        'weight_decay': 0.0001,
        'random_seed': 42234,
        
        'out_dir': './results/reaction_model',  # Output directory for saving results
        'save_best_model': True,
        'save_last_model': False,
        'checkpoint_path': None, # Path to a .ckpt file to resume/continue
        'mode': 'continue', # 'train' or 'continue'
        'freeze_base_model': False,
        
        'cuda': True, # Set to False to force CPU
        'gpu_id': 0,
        'num_workers': 4 # Number of workers for data loading
    }

    # Setup GPU or CPU
    if params['cuda'] and torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(params['gpu_id'])
        device = torch.device(f"cuda:{params['gpu_id']}")
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        device = torch.device("cpu")
        print("Using CPU")
        params['cuda'] = False

    # Create output directory
    os.makedirs(params['out_dir'], exist_ok=True)
    print(f"Output directory created/exists: {params['out_dir']}")

    # Create configuration directly from parameters dictionary
    config = Config.from_params(params)
    print("Configuration created successfully")

    # Load dataset using the unified configuration
    print("Loading dataset from unified configuration")
    
    # Pass the entire config object to the dataset
    dataset = ReactionDataset(config=config)
    
    print("Dataset loaded successfully")
    data_stats = dataset.get_data_stats()
    print(f"Dataset stats: Train: {data_stats['train_size']}, "
          f"Validation: {data_stats['val_size']}, "
          f"Test: {data_stats['test_size']}")
    
    if config.reaction.cv_folds > 0:
        print(f"Cross-validation enabled with {dataset.get_num_folds()} folds.")

    # Create trainer - use scaler based on the config parameter
    scalers = dataset.get_scalers() if config.reaction.use_scaler else None
    
    print(f"Using scaler: {config.reaction.use_scaler}")
    
    trainer = ReactionTrainer(
        config=config,
        scalers=scalers  # Pass scalers only if use_scaler is True
    )
    
    print("Trainer initialized successfully")
    print(f"Starting training with {config.training.max_epochs} epochs")

    # Train the model
    train_metrics = trainer.fit(
        train_dataset=dataset.train_data,
        val_dataset=dataset.val_data,
        test_dataset=dataset.test_data,
        checkpoint_path=config.training.resume_from_checkpoint,
        mode=config.training.mode
    )
    
    print("Training completed successfully")
    print("Metrics:", train_metrics)
    if 'best_model_path' in train_metrics and train_metrics['best_model_path']:
        print(f"Best model saved to: {train_metrics['best_model_path']}")
    elif config.training.save_last_model and 'last_model_path' in train_metrics and train_metrics['last_model_path']:
        print(f"Last model saved to: {train_metrics['last_model_path']}")


if __name__ == "__main__":
    main()