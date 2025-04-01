#!/usr/bin/env pythonimport os
import sys
import json
import numpy as np
import logging
import random
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import argparse

import torch
from torch_geometric.loader import DataLoader as GeometricDataLoader
import os
import pytorch_lightning as pl

parent_path = str(Path(Path(__file__).parent.absolute()).parent.absolute())
sys.path.insert(0, parent_path)

try:
    from module.pl_wrap import Estimator
    from utils.metrics import compute_regression_metrics
    from data.load_Reaction import create_data_split, scale_reaction_dataset
except ImportError as e:
    print(f"Failed to import required modules: {e}")
    print("Make sure the path to deep learning modules is correct")
    sys.exit(1)


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


def load_model_from_checkpoint(ckpt_path):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

    logger = logging.getLogger('deep')
    logger.info(f"Loading model from checkpoint: {ckpt_path}")

    model = Estimator.load_from_checkpoint(ckpt_path)
    model.eval()

    return model


def load_dataset(root, csv_file, target_fields=None, file_suffixes=None, input_features=None, force_reload=False):
    try:
        from data.load_Reaction import load_dataset as original_load_dataset
        return original_load_dataset(
            root=root,
            csv_file=csv_file,
            target_fields=target_fields,
            file_suffixes=file_suffixes,
            input_features=input_features,
            force_reload=force_reload
        )
    except ImportError:
        # For standalone use, include a simplified version
        from data.PygReaction import ReactionXYZDataset
        force_reload = force_reload or (target_fields and len(target_fields) > 1)

        try:
            dataset = ReactionXYZDataset(
                root=root,
                csv_file=csv_file,
                target_fields=target_fields,
                file_suffixes=file_suffixes,
                input_features=input_features,
                force_reload=force_reload
            )
        except Exception as e:
            logger = logging.getLogger('deep')
            logger.error(f"Error loading dataset: {e}")
            dataset = ReactionXYZDataset(
                root=root,
                csv_file=csv_file,
                target_fields=target_fields,
                file_suffixes=file_suffixes,
                input_features=input_features,
                force_reload=True
            )

        return dataset


def create_inference_dataloader(dataset, batch_size, num_workers=4):
    loader_kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'shuffle': False,
    }

    follow_batch = ['z0', 'z1', 'z2', 'pos0', 'pos1', 'pos2']
    dataloader_kwargs = {'follow_batch': follow_batch} if follow_batch else {}

    return GeometricDataLoader(dataset, **loader_kwargs, **dataloader_kwargs)


def run_inference(model, data_loader, output_dir, target_fields=None):
    logger = logging.getLogger('deep')
    logger.info("Running inference")

    cuda_available = torch.cuda.is_available()
    device = torch.device('cuda' if cuda_available else 'cpu')
    model = model.to(device)

    all_predictions = []
    all_reaction_ids = []
    all_reaction_data = []

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)

            reaction_ids = []
            reaction_data = []

            batch_size = len(batch.ptr) - 1
            for i in range(batch_size):
                item_data = {}

                if hasattr(batch, 'reaction_id'):
                    try:
                        item_data['reaction_id'] = batch.reaction_id[i]
                        reaction_ids.append(batch.reaction_id[i])
                    except:
                        item_data['reaction_id'] = f"unknown_{i}"
                        reaction_ids.append(f"unknown_{i}")

                if hasattr(batch, 'id'):
                    try:
                        item_data['id'] = batch.id[i]
                    except:
                        pass

                if hasattr(batch, 'reaction'):
                    try:
                        item_data['reaction'] = batch.reaction[i]
                    except:
                        pass

                if hasattr(batch, 'y'):
                    try:
                        y_values = batch.y[i].cpu().numpy()
                        if len(target_fields) == len(y_values):
                            for j, field in enumerate(target_fields):
                                item_data[f"true_{field}"] = y_values[j]
                    except:
                        pass

                reaction_data.append(item_data)

            _, _, predictions = model(
                batch.pos0, batch.pos1, batch.pos2,
                batch.z0, batch.z1, batch.z2,
                batch.batch,
                getattr(batch, 'xtb_features', None)
            )

            all_predictions.append(predictions.cpu())
            all_reaction_ids.extend(reaction_ids)
            all_reaction_data.extend(reaction_data)

    all_predictions = torch.cat(all_predictions, dim=0).numpy()

    unscaled_predictions = np.zeros_like(all_predictions)
    if hasattr(model, 'scaler') and model.scaler is not None:
        for i in range(all_predictions.shape[1]):
            if i < len(model.scaler):
                unscaled_predictions[:, i:i + 1] = model.scaler[i].inverse_transform(all_predictions[:, i:i + 1])
            else:
                unscaled_predictions[:, i] = all_predictions[:, i]
    else:
        unscaled_predictions = all_predictions

    results_dict = {}

    # Add reaction data
    for field in all_reaction_data[0].keys():
        results_dict[field] = [item.get(field, None) for item in all_reaction_data]

    # Add unscaled predictions only
    for i in range(unscaled_predictions.shape[1]):
        target_name = target_fields[i] if target_fields and i < len(target_fields) else f"target_{i}"

        # Add only unscaled predictions
        results_dict[f"pred_{target_name}"] = unscaled_predictions[:, i]

    # Create dataframe
    results_df = pd.DataFrame(results_dict)

    output_file = os.path.join(output_dir, 'predictions.csv')
    results_df.to_csv(output_file, index=False)
    logger.info(f"Saved predictions to {output_file}")

    np.save(os.path.join(output_dir, 'raw_predictions.npy'), all_predictions)
    np.save(os.path.join(output_dir, 'unscaled_predictions.npy'), unscaled_predictions)

    metrics_dict = {}

    for i, target_field in enumerate(target_fields):
        true_col = f"true_{target_field}"
        pred_col = f"pred_{target_field}"

        if true_col in results_df.columns and pred_col in results_df.columns:
            valid_indices = results_df[true_col].notnull() & results_df[pred_col].notnull()

            if valid_indices.sum() > 0:
                true_values = results_df.loc[valid_indices, true_col].values.reshape(-1, 1)
                pred_values = results_df.loc[valid_indices, pred_col].values.reshape(-1, 1)

                field_metrics = compute_regression_metrics(true_values, pred_values)
                metrics_dict[target_field] = field_metrics

                logger.info(
                    f"Metrics for {target_field}: MAE={field_metrics['mae']:.4f}, RMSE={field_metrics['rmse']:.4f}, R2={field_metrics['r2']:.4f}")

    if metrics_dict:
        with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics_dict, f, indent=2, cls=NumpyEncoder)

    return results_df


def get_inference_parser():
    parser = argparse.ArgumentParser(description='Inference for molecular graph neural network')

    parser.add_argument('--dataset', type=str, required=True, choices=['XTB'], help='Dataset type')
    parser.add_argument('--readout', type=str, required=True, help='Readout function')
    parser.add_argument('--model_type', type=str, default='dimenet++', help='Type of model')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--node_latent_dim', type=int, default=128, help='Node latent dimension')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed')
    parser.add_argument('--out_dir', type=str, default='./inference_results', help='Output directory')
    parser.add_argument('--reaction_dataset_root', type=str, required=True, help='Path to dataset root')
    parser.add_argument('--infer_csv', type=str, required=True, help='CSV file with inference data')
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--reaction_target_fields', type=str, nargs='+', required=True, help='Target fields to predict')
    parser.add_argument('--reaction_file_suffixes', type=str, nargs=3,
                        default=['_reactant.xyz', '_ts.xyz', '_product.xyz'], help='Suffixes for reaction files')
    parser.add_argument('--input_features', type=str, nargs='+', default=['G(TS)_xtb', 'DrG_xtb'],
                        help='Input feature columns')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--cuda', action='store_true', help='Use CUDA if available')
    parser.add_argument('--no-cuda', action='store_false', dest='cuda', help='Disable CUDA')
    parser.add_argument('--log_level', type=str, default='info',
                        choices=['debug', 'info', 'warning', 'error', 'critical'], help='Logging level')

    return parser


def main():
    parser = get_inference_parser()
    args = parser.parse_args()

    pl.seed_everything(args.random_seed)

    os.makedirs(args.out_dir, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(args.out_dir, 'inference.log'))
        ]
    )

    logger = logging.getLogger('deep')
    logger.info(f"Starting inference with checkpoint: {args.ckpt_path}")

    model = load_model_from_checkpoint(args.ckpt_path)
    logger.info(f"Loaded model: {model.model_type} with readout {model.readout}")

    # Check if infer_csv exists
    if not os.path.exists(args.infer_csv):
        logger.warning(f"Inference CSV file not found at {args.infer_csv}")
        # Try to use dataset_csv from the dataset directory
        default_csv = os.path.join(args.reaction_dataset_root, "dataset_xtb_final.csv")
        if os.path.exists(default_csv):
            logger.info(f"Using default dataset CSV instead: {default_csv}")
            args.infer_csv = default_csv
        else:
            # Look for any CSV file in the dataset directory
            csv_files = [f for f in os.listdir(args.reaction_dataset_root) if f.endswith('.csv')]
            if csv_files:
                default_csv = os.path.join(args.reaction_dataset_root, csv_files[0])
                logger.info(f"Using found CSV file instead: {default_csv}")
                args.infer_csv = default_csv
            else:
                raise FileNotFoundError(f"Could not find any CSV files in {args.reaction_dataset_root}")

    logger.info(f"Loading inference dataset from {args.infer_csv}")
    try:
        inference_dataset = load_dataset(
            root=args.reaction_dataset_root,
            csv_file=args.infer_csv,
            target_fields=args.reaction_target_fields,
            file_suffixes=args.reaction_file_suffixes,
            input_features=args.input_features,
            force_reload=True
        )
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise

    inference_data = create_data_split(inference_dataset, range(len(inference_dataset)))
    logger.info(f"Loaded {len(inference_data)} reactions for inference")

    if hasattr(model, 'scaler') and model.scaler is not None:
        logger.info("Scaling inference data using model's scalers")
        inference_data_scaled = scale_reaction_dataset(inference_data, model.scaler)
    else:
        logger.info("No scalers found in model, using unscaled data")
        inference_data_scaled = inference_data

    inference_loader = create_inference_dataloader(
        inference_data_scaled, args.batch_size, args.num_workers
    )

    results = run_inference(
        model, inference_loader, args.out_dir, args.reaction_target_fields
    )

    logger.info(f"Inference completed successfully with {len(results)} predictions")

    if args.reaction_target_fields:
        logger.info("Prediction statistics:")
        for field in args.reaction_target_fields:
            pred_col = f"pred_{field}"
            if pred_col in results.columns:
                values = results[pred_col].values
                logger.info(
                    f"  {field}: mean={np.mean(values):.4f}, std={np.std(values):.4f}, min={np.min(values):.4f}, max={np.max(values):.4f}")


if __name__ == "__main__":
    main()