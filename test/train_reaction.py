#!/usr/bin/env python
import os
import sys
import argparse
from pathlib import Path
import torch

from deepreaction import ReactionDataset, ReactionTrainer
from deepreaction.config import ReactionConfig, ModelConfig, TrainingConfig, Config, save_config


def get_parser():
    parser = argparse.ArgumentParser(description='Train a molecular reaction prediction model')

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
    parser.add_argument('--file_dir_pattern', type=str, default='reaction_*', help='Directory pattern')
    parser.add_argument('--id_field', type=str, default='ID', help='ID field name')
    parser.add_argument('--dir_field', type=str, default='R_dir', help='Directory field name')
    parser.add_argument('--reaction_field', type=str, default='reaction', help='Reaction field name')
    parser.add_argument('--cv_folds', type=int, default=0, help='Number of cross-validation folds')

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

    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--eval_batch_size', type=int, default=None, help='Evaluation batch size')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--finetune_lr', type=float, default=None,
                        help='Fine-tuning learning rate (defaults to 10% of normal lr)')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--min_epochs', type=int, default=0, help='Minimum number of epochs')
    parser.add_argument('--early_stopping', type=int, default=40, help='Early stopping patience')
    parser.add_argument('--optimizer', type=str, default='adamw', help='Optimizer')
    parser.add_argument('--scheduler', type=str, default='warmup_cosine', help='Learning rate scheduler')
    parser.add_argument('--warmup_epochs', type=int, default=10, help='Warmup epochs')
    parser.add_argument('--min_lr', type=float, default=1e-7, help='Minimum learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay')
    parser.add_argument('--random_seed', type=int, default=42234, help='Random seed')

    parser.add_argument('--out_dir', type=str, default='./results/reaction_model', help='Output directory')
    parser.add_argument('--save_best_model', action='store_true', default=True, help='Save best model')
    parser.add_argument('--save_last_model', action='store_true', default=False, help='Save last model')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Checkpoint path for resuming training')
    parser.add_argument('--mode', type=str, default='continue', choices=['continue', 'finetune'],
                        help='Mode for checkpoint loading: continue training or finetune')
    parser.add_argument('--freeze_base_model', action='store_true', help='Freeze base model when fine-tuning')

    parser.add_argument('--cuda', action='store_true', default=True, help='Use CUDA')
    parser.add_argument('--no-cuda', action='store_false', dest='cuda', help='Do not use CUDA')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    if args.cuda and torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        device = torch.device(f"cuda:{args.gpu_id}")
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        device = torch.device("cpu")
        print("Using CPU")

    os.makedirs(args.out_dir, exist_ok=True)

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
    print(
        f"Dataset stats: Train: {data_stats['train_size']}, Validation: {data_stats['val_size']}, Test: {data_stats['test_size']}")

    model_params = {
        "model_type": args.model_type,
        "readout": args.readout,
        "hidden_channels": args.hidden_channels,
        "num_blocks": args.num_blocks,
        "cutoff": args.cutoff,
        "int_emb_size": args.int_emb_size,
        "basis_emb_size": args.basis_emb_size,
        "out_emb_channels": args.out_emb_channels,
        "num_spherical": args.num_spherical,
        "num_radial": args.num_radial,
        "envelope_exponent": args.envelope_exponent,
        "num_before_skip": args.num_before_skip,
        "num_after_skip": args.num_after_skip,
        "num_output_layers": args.num_output_layers,
        "max_num_neighbors": args.max_num_neighbors,
        "node_dim": args.node_dim,
        "dropout": args.dropout,
        "use_layer_norm": args.use_layer_norm,
    }

    additional_kwargs = {}
    if args.finetune_lr is not None:
        additional_kwargs['finetune_lr'] = args.finetune_lr
    if args.freeze_base_model:
        additional_kwargs['freeze_base_model'] = True

    trainer = ReactionTrainer(
        model_type=args.model_type,
        readout=args.readout,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        learning_rate=args.lr,
        output_dir=args.out_dir,
        early_stopping_patience=args.early_stopping,
        save_best_model=args.save_best_model,
        save_last_model=args.save_last_model,
        random_seed=args.random_seed,
        num_targets=len(args.target_fields),
        use_scaler=True,
        scalers=dataset.get_scalers(),
        optimizer=args.optimizer,
        weight_decay=args.weight_decay,
        scheduler=args.scheduler,
        warmup_epochs=args.warmup_epochs,
        min_lr=args.min_lr,
        gpu=args.cuda,
        node_dim=args.node_dim,
        dropout=args.dropout,
        use_layer_norm=args.use_layer_norm,
        target_field_names=args.target_fields,
        target_weights=args.target_weights,
        use_xtb_features=len(args.input_features) > 0,
        num_xtb_features=len(args.input_features),
        prediction_hidden_layers=args.prediction_hidden_layers,
        prediction_hidden_dim=args.prediction_hidden_dim,
        min_epochs=args.min_epochs,
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
        num_workers=args.num_workers,
        **additional_kwargs
    )

    print(f"Starting {args.mode} training with {args.epochs} epochs")
    train_metrics = trainer.fit(
        train_dataset=dataset.train_data,
        val_dataset=dataset.val_data,
        test_dataset=dataset.test_data,
        checkpoint_path=args.checkpoint_path,
        mode=args.mode
    )

    print(f"Training completed: {train_metrics}")
    if 'best_model_path' in train_metrics:
        print(f"Best model saved to: {train_metrics['best_model_path']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())