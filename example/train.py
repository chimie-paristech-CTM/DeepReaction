# deepreaction/cli/train.py
#!/usr/bin/env python
import argparse
import os
import sys
import torch
from pathlib import Path
from deepreaction import ReactionTrainer, ReactionDataset, Config, load_config

def main():
    parser = argparse.ArgumentParser(description='Train a molecular reaction prediction model')
    parser.add_argument('-c', '--config', type=str, help='Path to configuration file')
    parser.add_argument('--dataset_root', type=str, default='./dataset/DATASET_DA_F', help='Dataset root directory')
    parser.add_argument('--dataset_csv', type=str, default='./dataset/DATASET_DA_F/dataset_xtb_final.csv',
                        help='Dataset CSV file (relative to dataset_root)')
    parser.add_argument('--target_fields', type=str, nargs='+', default=['G(TS)', 'DrG'], help='Target fields')
    parser.add_argument('--input_features', type=str, nargs='+', default=['G(TS)_xtb', 'DrG_xtb'], help='Input features')
    parser.add_argument('--model_type', type=str, default='dimenet++', help='Model type')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--out_dir', type=str, default='./results', help='Output directory')
    parser.add_argument('--cuda', action='store_true', default=True, help='Use CUDA if available')
    # parser.add_argument('--no-cuda', action='store_false', dest='cuda', help='Do not use CUDA')
    
    args = parser.parse_args()
    
    # Set up GPU
    if args.cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
        args.cuda = False
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Either load from config file or create from command line args
    if args.config:
        config = load_config(args.config)
    else:
        # Use command line arguments to create params dict
        params = vars(args)
        config = Config.from_params(params)
    
    # Load dataset with config
    dataset = ReactionDataset(config=config)
    print(f"Dataset loaded successfully")
    data_stats = dataset.get_data_stats()
    print(f"Dataset stats: Train: {data_stats['train_size']}, Validation: {data_stats['val_size']}, Test: {data_stats['test_size']}")
    
    # Create and train model
    trainer = ReactionTrainer(config=config, scalers=dataset.get_scalers())
    print(f"Starting training with {config.training.max_epochs} epochs")
    
    train_metrics = trainer.fit(
        train_dataset=dataset.train_data,
        val_dataset=dataset.val_data,
        test_dataset=dataset.test_data
    )
    
    print(f"Training completed.")
    print(f"Metrics: {train_metrics}")
    if 'best_model_path' in train_metrics and train_metrics['best_model_path']:
        print(f"Best model saved to: {train_metrics['best_model_path']}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())