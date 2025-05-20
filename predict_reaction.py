#!/usr/bin/env python
import os
import sys
import argparse
import torch
import pandas as pd
import numpy as np
from pathlib import Path

from deepreaction.data.PygReaction import ReactionXYZDataset
from deepreaction.module.pl_wrap import Estimator
from torch_geometric.loader import DataLoader


def get_parser():
    parser = argparse.ArgumentParser(description='Make predictions with a trained molecular reaction model')

    parser.add_argument('--dataset_root', type=str, default='./dataset/DATASET_DA_F', help='Dataset root directory')
    parser.add_argument('--dataset_csv', type=str, default='./dataset/DATASET_DA_F/dataset_xtb_final.csv',
                        help='Dataset CSV file')
    parser.add_argument('--input_features', type=str, nargs='+', default=['G(TS)_xtb', 'DrG_xtb'],
                        help='Input features')
    parser.add_argument('--file_patterns', type=str, nargs='+', default=['*_reactant.xyz', '*_ts.xyz', '*_product.xyz'],
                        help='File patterns')
    parser.add_argument('--id_field', type=str, default='ID', help='ID field name')
    parser.add_argument('--dir_field', type=str, default='R_dir', help='Directory field name')
    parser.add_argument('--reaction_field', type=str, default='reaction', help='Reaction field name')

    # Model parameters
    parser.add_argument('--checkpoint_path', type=str,
                        default='./results/reaction_model/checkpoints/best-epoch=0000-val_total_loss=33.4806.ckpt',
                        help='Path to the trained model checkpoint')

    # Output parameters
    parser.add_argument('--output_csv', type=str, default='./predictions.csv', help='Path to save predictions CSV')
    parser.add_argument('--output_dir', type=str, default='./predictions', help='Directory to save prediction output')

    # Inference parameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for inference')
    parser.add_argument('--cuda', action='store_true', default=True, help='Use CUDA')
    parser.add_argument('--no-cuda', action='store_false', dest='cuda', help='Do not use CUDA')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--debug', action='store_true', default=False, help='Enable debug output')

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

    os.makedirs(args.output_dir, exist_ok=True)

    if args.input_features is None:
        args.input_features = []

    # First, load the model to get target fields
    print(f"Loading model from checkpoint: {args.checkpoint_path}")
    model = Estimator.load_from_checkpoint(args.checkpoint_path)
    model = model.to(device)
    model.eval()

    # Get target fields from the model
    target_fields = model.target_field_names if hasattr(model, 'target_field_names') else None

    # If the model doesn't have target field names defined, try to infer from num_targets
    if not target_fields and hasattr(model, 'num_targets'):
        print(f"Model has {model.num_targets} targets but no defined field names")
        target_fields = ['G(TS)', 'DrG'] if model.num_targets == 2 else [f'target_{i}' for i in
                                                                         range(model.num_targets)]

    print(f"Using target fields from model: {target_fields}")

    if args.debug:
        print("Model details:")
        print(f"- Number of targets: {model.num_targets if hasattr(model, 'num_targets') else 'unknown'}")
        print(
            f"- Target field names: {model.target_field_names if hasattr(model, 'target_field_names') else 'unknown'}")
        print(f"- Use XTB features: {model.use_xtb_features if hasattr(model, 'use_xtb_features') else 'unknown'}")
        print(
            f"- Number of XTB features: {model.num_xtb_features if hasattr(model, 'num_xtb_features') else 'unknown'}")

    # Now load the dataset directly using ReactionXYZDataset
    print(f"Loading dataset from {args.dataset_root}")
    dataset = ReactionXYZDataset(
        root=args.dataset_root,
        csv_file=args.dataset_csv,
        target_fields=target_fields,
        file_patterns=args.file_patterns,
        input_features=args.input_features,
        id_field=args.id_field,
        dir_field=args.dir_field,
        reaction_field=args.reaction_field,
        inference_mode=True
    )

    print(f"Dataset loaded with {len(dataset)} samples")

    # Create data loader
    follow_batch = ['z0', 'z1', 'z2', 'pos0', 'pos1', 'pos2']
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        follow_batch=follow_batch
    )

    # Make predictions
    all_predictions = []
    all_reaction_ids = []
    all_reaction_data = []

    print("Running inference...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            batch = batch.to(device)
            pos0, pos1, pos2 = batch.pos0, batch.pos1, batch.pos2
            z0, z1, z2, batch_mapping = batch.z0, batch.z1, batch.z2, batch.batch
            xtb_features = getattr(batch, 'xtb_features', None)

            if args.debug and batch_idx == 0:
                print(f"Batch data shapes:")
                print(f"- z0: {z0.shape}, z1: {z1.shape}, z2: {z2.shape}")
                print(f"- pos0: {pos0.shape}, pos1: {pos1.shape}, pos2: {pos2.shape}")
                if xtb_features is not None:
                    print(f"- xtb_features: {xtb_features.shape}")

            # 直接调用模型的forward方法
            _, _, predictions = model.model(pos0, pos1, pos2, z0, z1, z2, batch_mapping, xtb_features)

            if args.debug and batch_idx == 0:
                print(f"Predictions shape: {predictions.shape}")
                print(f"Expected num_targets: {model.num_targets}")
                if predictions.shape[1] != model.num_targets:
                    print(
                        f"WARNING: Predictions shape {predictions.shape[1]} doesn't match num_targets {model.num_targets}")

            all_predictions.append(predictions.cpu().numpy())

            for i in range(len(predictions)):
                reaction_id = batch.reaction_id[i] if hasattr(batch, 'reaction_id') else f"sample_{i}"
                all_reaction_ids.append(reaction_id)

                reaction_data = {}
                for attr in ['id', 'reaction']:
                    if hasattr(batch, attr):
                        value = getattr(batch, attr)
                        if isinstance(value, list) and i < len(value):
                            reaction_data[attr] = value[i]
                        else:
                            reaction_data[attr] = value
                all_reaction_data.append(reaction_data)

    # Stack predictions
    predictions = np.vstack(all_predictions) if all_predictions else np.array([])

    if args.debug:
        print(f"Raw predictions shape: {predictions.shape}")

    # Apply inverse scaling if available
    results = {}
    for i, target_name in enumerate(target_fields):
        if i < predictions.shape[1]:
            target_preds = predictions[:, i].reshape(-1, 1)
            if hasattr(model, 'scaler') and model.scaler is not None and isinstance(model.scaler, list) and i < len(
                    model.scaler):
                target_preds = model.scaler[i].inverse_transform(target_preds)
            results[target_name] = target_preds.flatten()

    # Create output DataFrame
    results_df = pd.DataFrame()
    results_df['ID'] = all_reaction_ids

    for i, data in enumerate(all_reaction_data):
        for key, value in data.items():
            if key not in results_df.columns:
                results_df[key] = None
            if value is not None and i < len(results_df):
                results_df.at[i, key] = value

    for target_name, preds in results.items():
        results_df[f'{target_name}_predicted'] = preds

    # Save predictions
    results_df.to_csv(args.output_csv, index=False)
    np.save(os.path.join(args.output_dir, 'predictions.npy'), predictions)

    print(f"Predictions saved to: {args.output_csv}")
    print(f"Number of predictions: {len(results_df)}")

    if len(results_df) > 0:
        print("\nSample predictions:")
        for col in results_df.columns:
            if '_predicted' in col:
                print(f"Column: {col}")
        print(results_df.head())

    return 0


if __name__ == "__main__":
    sys.exit(main())