import torch
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from deepreaction import Config, ReactionDataset, ReactionTrainer


def main():
    # Simplified parameters - removed redundant ones
    params = {
        # Dataset parameters
        'dataset_root': '../dataset/DATASET_DA_F',
        'dataset_csv': '../dataset/DATASET_DA_F/dataset_xtb_final.csv',
        'train_ratio': 0.8,
        'val_ratio': 0.1,
        'test_ratio': 0.1,
        'target_fields': ['DG_act', 'DrG'],
        'target_weights': [1.0, 1.0],
        'input_features': ['DG_act_xtb', 'DrG_xtb'],
        'file_keywords': ['reactant', 'ts', 'product'],  # Simplified from file_suffixes
        'use_scaler': True,
        'id_field': 'ID',           # Column name for reaction ID
        'dir_field': 'R_dir',       # Column name for directory containing reaction files
        'reaction_field': 'smiles', # Column name for SMILES string representation

        # Readout parameter
        'readout': 'mean',
        'model_type': 'dimenet++',
        'node_dim': 128,
        'dropout': 0.1,
        'prediction_hidden_layers': 4,
        'prediction_hidden_dim': 512,
        'use_layer_norm': False,
        'use_xtb_features': True,
        'max_num_atoms': 100,

        # DimeNet++ specific parameters
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

        # Readout parameters
        'readout_hidden_dim': 128,
        'readout_num_heads': 4,
        'readout_num_sabs': 2,

        # Training parameters
        'batch_size': 16,
        'eval_batch_size': 32,
        'lr': 0.0005,
        'max_epochs': 100,
        'early_stopping_patience': 40,
        'early_stopping_min_delta': 0.0001,
        'optimizer': 'adamw',
        'scheduler': 'warmup_cosine',
        'warmup_epochs': 10,
        'min_lr': 1e-7,
        'weight_decay': 0.0001,
        'random_seed': 42234,
        'loss_function': 'mse',
        'gradient_clip_val': 0.0,
        'gradient_accumulation_steps': 1,
        'precision': '32',
        'out_dir': './results/reaction_model',
        'save_best_model': True,

        # System parameters
        'cuda': True,
        'gpu_id': 0,
        'num_workers': 4,
        'strategy': 'auto',
        'num_nodes': 1,
        'devices': 1,
        'log_level': 'info',
        'matmul_precision': 'medium',
    }

    # GPU setup
    if params['cuda'] and torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(params['gpu_id'])
        device = torch.device(f"cuda:{params['gpu_id']}")
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")

        device_name = torch.cuda.get_device_name(device)
        if any(gpu in device_name for gpu in
               ['V100', 'A100', 'A10', 'A30', 'A40', 'RTX 30', 'RTX 40', '3080', '3090', '4080', '4090']):
            torch.set_float32_matmul_precision(params.get('matmul_precision', 'high'))
            print(
                f"Set float32 matmul precision to '{params.get('matmul_precision', 'high')}' for better Tensor Core utilization")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        device = torch.device("cpu")
        print("Using CPU")
        params['cuda'] = False

    print("Creating configuration...")
    config = Config.from_params(params)

    # Print key configuration
    print(f"\nKey Configuration:")
    print(f"  Model: {config.model.model_type}")
    print(f"  Target fields: {config.dataset.target_fields}")
    print(f"  Input features: {config.dataset.input_features}")
    print(f"  File keywords: {config.dataset.file_keywords}")
    print(f"  CSV field mapping: id='{config.dataset.id_field}', dir='{config.dataset.dir_field}', reaction='{config.dataset.reaction_field}'")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"  Learning rate: {config.training.lr}")
    print(f"  Max epochs: {config.training.max_epochs}")
    print(f"  Output directory: {config.training.out_dir}")

    # Load dataset
    print("\nLoading dataset...")
    dataset = ReactionDataset(config=config)
    train_data, val_data, test_data, scalers = dataset.get_data_splits()
    print(f"Dataset loaded: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")

    # Initialize trainer
    print("\nInitializing trainer...")
    trainer = ReactionTrainer(config=config)

    # Start training
    print(f"\nStarting training with {config.training.max_epochs} epochs...")
    try:
        train_metrics = trainer.fit(
            train_dataset=train_data,
            val_dataset=val_data,
            test_dataset=test_data,
            scalers=scalers,
            checkpoint_path=config.training.checkpoint_path,
            mode=config.training.mode
        )

        print("\n" + "=" * 50)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("=" * 50)
        print(f"Training time: {train_metrics.get('training_time', 0):.2f} seconds")
        print(f"Epochs completed: {train_metrics.get('epochs_completed', 0)}")

        if 'best_model_path' in train_metrics and train_metrics['best_model_path']:
            print(f"Best model saved to: {train_metrics['best_model_path']}")

        if 'test_results' in train_metrics and train_metrics['test_results']:
            print(f"Test results: {train_metrics['test_results']}")

        print(f"All outputs saved in: {config.training.out_dir}")
        print("=" * 50)

    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)