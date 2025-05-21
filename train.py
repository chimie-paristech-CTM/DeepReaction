#!/usr/bin/env python
import argparse
import os
import sys
import torch
from pathlib import Path
from deepreaction import ReactionTrainer, ReactionDataset, load_config

def main():
    parser = argparse.ArgumentParser(description='Train a molecular reaction prediction model')
    parser.add_argument('-c', '--config', type=str, help='Path to configuration file')
    parser.add_argument('--dataset', type=str, default='XTB', help='Dataset name')
    parser.add_argument('--readout', type=str, default='mean', help='Readout function')
    parser.add_argument('--dataset_root', type=str, default='./dataset/DATASET_DA_F', help='Dataset root directory')
    parser.add_argument('--dataset_csv', type=str, default='./dataset/DATASET_DA_F/dataset_xtb_final.csv',
                        help='Dataset CSV file (relative to dataset_root)')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Validation set ratio')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='Test set ratio')
    parser.add_argument('--target_fields', type=str, nargs='+', default=['G(TS)', 'DrG'], help='Target fields')
    parser.add_argument('--target_weights', type=float, nargs='+', default=[1.0, 1.0], help='Target weights')
    parser.add_argument('--input_features', type=str, nargs='+', default=['G(TS)_xtb', 'DrG_xtb'],
                        help='Input features')
    parser.add_argument('--file_patterns', type=str, nargs='+', default=['*_reactant.xyz', '*_ts.xyz', '*_product.xyz'],
                        help='File patterns')
    parser.add_argument('--id_field', type=str, default='ID', help='ID field name')
    parser.add_argument('--dir_field', type=str, default='R_dir', help='Directory field name')
    parser.add_argument('--reaction_field', type=str, default='reaction', help='Reaction field name')
    parser.add_argument('--cv_folds', type=int, default=0, help='Number of cross-validation folds')
    parser.add_argument('--model_type', type=str, default='dimenet++', help='Model type')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--out_dir', type=str, default='./results/reaction_model', help='Output directory')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Checkpoint path for resuming training')
    parser.add_argument('--mode', type=str, default='continue', choices=['continue', 'finetune'],
                        help='Mode for checkpoint loading: continue training or finetune')
    parser.add_argument('--cuda', action='store_true', default=True, help='Use CUDA')
    parser.add_argument('--no-cuda', action='store_false', dest='cuda', help='Do not use CUDA')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
    
    args = parser.parse_args()
    
    # Set up GPU
    if args.cuda and torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        device = torch.device(f"cuda:{args.gpu_id}")
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        device = torch.device("cpu")
        print("Using CPU")
        args.cuda = False
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Load configuration if provided
    if args.config:
        config = load_config(args.config)
    else:
        # Use CLI arguments
        from deepreaction.config.config import Config, ReactionConfig, ModelConfig, TrainingConfig, save_config
        
        reaction_config = ReactionConfig(
            dataset_root=args.dataset_root,
            dataset_csv=args.dataset_csv,
            target_fields=args.target_fields,
            file_patterns=args.file_patterns,
            input_features=args.input_features,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            cv_folds=args.cv_folds,
            id_field=args.id_field,
            dir_field=args.dir_field,
            reaction_field=args.reaction_field
        )

        model_config = ModelConfig(
            model_type=args.model_type,
            readout=args.readout
        )

        training_config = TrainingConfig(
            output_dir=args.out_dir,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            max_epochs=args.epochs,
            gpu=args.cuda,
            resume_from_checkpoint=args.checkpoint_path
        )

        config = Config(
            reaction=reaction_config,
            model=model_config,
            training=training_config
        )
        
        # Save configuration
        config_path = os.path.join(args.out_dir, 'config')
        save_config(config, config_path)
        print(f"Configuration saved to {config_path}.yaml and {config_path}.json")
    
    # Load dataset
    print(f"Loading dataset from {config.reaction.dataset_root}")
    dataset = ReactionDataset(
        root=config.reaction.dataset_root,
        csv_file=config.reaction.dataset_csv,
        target_fields=config.reaction.target_fields,
        file_patterns=config.reaction.file_patterns,
        input_features=config.reaction.input_features,
        use_scaler=True,
        random_seed=config.reaction.random_seed,
        train_ratio=config.reaction.train_ratio,
        val_ratio=config.reaction.val_ratio,
        test_ratio=config.reaction.test_ratio,
        cv_folds=config.reaction.cv_folds,
        id_field=config.reaction.id_field,
        dir_field=config.reaction.dir_field,
        reaction_field=config.reaction.reaction_field
    )
    
    print("Dataset loaded successfully")
    data_stats = dataset.get_data_stats()
    print(f"Dataset stats: Train: {data_stats['train_size']}, Validation: {data_stats['val_size']}, Test: {data_stats['test_size']}")
    
    # Create trainer
    trainer = ReactionTrainer(
        model_type=config.model.model_type,
        readout=config.model.readout,
        batch_size=config.training.batch_size,
        max_epochs=config.training.max_epochs,
        learning_rate=config.training.learning_rate,
        output_dir=config.training.output_dir,
        early_stopping_patience=config.training.early_stopping_patience,
        save_best_model=config.training.save_best_model,
        save_last_model=config.training.save_last_model,
        random_seed=config.reaction.random_seed,
        num_targets=len(config.reaction.target_fields),
        use_scaler=True,
        scalers=dataset.get_scalers(),
        optimizer=config.training.optimizer,
        weight_decay=config.training.weight_decay,
        scheduler=config.training.scheduler,
        warmup_epochs=config.training.warmup_epochs,
        min_lr=config.training.min_lr,
        gpu=config.training.gpu,
        node_dim=config.model.node_dim,
        dropout=config.model.dropout,
        use_layer_norm=config.model.use_layer_norm,
        target_field_names=config.reaction.target_fields,
        use_xtb_features=config.model.use_xtb_features,
        num_xtb_features=config.model.num_xtb_features,
        prediction_hidden_layers=config.model.prediction_hidden_layers,
        prediction_hidden_dim=config.model.prediction_hidden_dim,
        min_epochs=config.training.min_epochs,
        num_workers=config.training.num_workers,
        hidden_channels=config.model.hidden_channels,
        num_blocks=config.model.num_blocks,
        cutoff=config.model.cutoff,
        int_emb_size=config.model.int_emb_size,
        basis_emb_size=config.model.basis_emb_size,
        out_emb_channels=config.model.out_emb_channels,
        num_spherical=config.model.num_spherical,
        num_radial=config.model.num_radial,
        envelope_exponent=config.model.envelope_exponent,
        num_before_skip=config.model.num_before_skip,
        num_after_skip=config.model.num_after_skip,
        num_output_layers=config.model.num_output_layers,
        max_num_neighbors=config.model.max_num_neighbors,
    )
    
    # Train model
    print(f"Starting {args.mode} training with {config.training.max_epochs} epochs")
    train_metrics = trainer.fit(
        train_dataset=dataset.train_data,
        val_dataset=dataset.val_data,
        test_dataset=dataset.test_data,
        checkpoint_path=config.training.resume_from_checkpoint,
        mode=args.mode
    )
    
    print(f"Training completed.")
    print(f"Metrics: {train_metrics}")
    if 'best_model_path' in train_metrics and train_metrics['best_model_path']:
        print(f"Best model saved to: {train_metrics['best_model_path']}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())