#!/usr/bin/env python

import os
import sys
import argparse
import numpy as np
import pandas as pd
import logging
import torch
import yaml
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import csv

from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as GeometricDataLoader

import pytorch_lightning as pl

parent_path = str(Path(Path(__file__).parent.absolute()).parent.absolute())
sys.path.insert(0, parent_path)

from cli.config import setup_logging
from data.PygReaction import ReactionXYZDataset, read_xyz, symbols_to_atomic_numbers
from module.pl_wrap import Estimator
from torch_geometric.data import Data


class InferenceDataset(ReactionXYZDataset):
    def __init__(self, root, csv_file='inference.csv', file_suffixes=None, input_features=None):
        self.csv_file = csv_file
        self.file_suffixes = file_suffixes or ['_reactant.xyz', '_ts.xyz', '_product.xyz']
        self.input_features = input_features or ['G(TS)_xtb', 'DrG_xtb']

        if not isinstance(self.input_features, list):
            self.input_features = [self.input_features]

        super(ReactionXYZDataset, self).__init__(root, None, None, None)

        self.data_list = self.process_inference_data()
        self.data, self.slices = self.collate(self.data_list)

    def download(self):
        pass

    def process(self):
        pass

    def process_inference_data(self):
        data_list = []

        if not os.path.exists(self.csv_file):
            raise FileNotFoundError(f"CSV file {self.csv_file} does not exist")

        # Read CSV file
        with open(self.csv_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            raise ValueError("Empty CSV file")

        reactant_suffix, ts_suffix, product_suffix = self.file_suffixes

        for row in rows:
            reaction_id = row.get('ID', '').strip()
            R_dir = row.get('R_dir', '').strip()
            reaction_str = row.get('reaction', '').strip()

            # Validation checks
            if not reaction_id or not R_dir:
                print(f"Warning: Missing required fields, skipping record: {row}")
                continue

            folder_path = os.path.join(self.root, R_dir)
            if not os.path.isdir(folder_path):
                print(f"Warning: Folder {folder_path} does not exist, skipping reaction_id {reaction_id}")
                continue

            # Get input feature values
            feature_values = []
            skip_record = False

            for feature_name in self.input_features:
                feature_str = row.get(feature_name, '').strip()
                if not feature_str:
                    print(f"Warning: Missing feature {feature_name}, skipping record: {reaction_id}")
                    skip_record = True
                    break

                try:
                    feature_values.append(float(feature_str))
                except ValueError:
                    print(f"Error parsing feature {feature_name} in reaction_id {reaction_id}")
                    skip_record = True
                    break

            if skip_record:
                continue

            # Prepare file paths
            prefix = R_dir
            if prefix.startswith("reaction_"):
                prefix = prefix[len("reaction_"):]

            reactant_file = os.path.join(folder_path, f"{prefix}{reactant_suffix}")
            ts_file = os.path.join(folder_path, f"{prefix}{ts_suffix}")
            product_file = os.path.join(folder_path, f"{prefix}{product_suffix}")

            if not (os.path.exists(reactant_file) and os.path.exists(ts_file) and os.path.exists(product_file)):
                print(
                    f"Warning: One or more xyz files are missing in {folder_path}, skipping reaction_id {reaction_id}")
                continue

            # Read atomic symbols and positions for all three files
            symbols0, pos0 = read_xyz(reactant_file)
            symbols1, pos1 = read_xyz(ts_file)
            symbols2, pos2 = read_xyz(product_file)

            if None in (symbols0, pos0, symbols1, pos1, symbols2, pos2):
                print(f"Warning: Failed to read XYZ files for {reaction_id}, skipping")
                continue

            # Convert atomic symbols to atomic numbers for each file
            z0 = symbols_to_atomic_numbers(symbols0)
            z1 = symbols_to_atomic_numbers(symbols1)
            z2 = symbols_to_atomic_numbers(symbols2)

            if None in (z0, z1, z2):
                print(f"Warning: Failed to convert atomic symbols for {reaction_id}, skipping")
                continue

            # Consistency check for atom counts
            if len({pos0.size(0), pos1.size(0), pos2.size(0), z0.size(0), z1.size(0), z2.size(0)}) > 1:
                print(f"Warning: Inconsistent atom count in {reaction_id}, skipping")
                continue

            # Create data object with dummy y value (will be ignored during inference)
            data = Data(
                z0=z0, z1=z1, z2=z2,
                pos0=pos0, pos1=pos1, pos2=pos2,
                y=torch.zeros((1, 2), dtype=torch.float),  # Dummy y value
                xtb_features=torch.tensor([feature_values], dtype=torch.float),
                feature_names=self.input_features,
                reaction_id=reaction_id,
                id=R_dir,
                reaction=reaction_str,
                num_nodes=z0.size(0)
            )

            data_list.append(data)

        if not data_list:
            raise RuntimeError("No reaction data processed, please check the CSV and xyz file formats.")

        return data_list


def load_model(checkpoint_path, cuda=True):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    device = torch.device('cuda' if cuda and torch.cuda.is_available() else 'cpu')

    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract hyperparameters
    if 'hyper_parameters' in checkpoint:
        hparams = checkpoint['hyper_parameters']
    else:
        raise ValueError(f"No hyperparameters found in checkpoint: {checkpoint_path}")

    # Create model from hyperparameters
    model = Estimator(**hparams)

    # Load the state dict
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()

    return model


def create_inference_dataloader(dataset, batch_size=32, num_workers=4):
    dataloader_kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'shuffle': False
    }

    # Add follow_batch for XTB dataset
    follow_batch = ['z0', 'z1', 'z2', 'pos0', 'pos1', 'pos2']
    dataloader_kwargs['follow_batch'] = follow_batch

    return GeometricDataLoader(dataset, **dataloader_kwargs)


def run_inference(model, dataloader, output_file=None, return_results=True):
    device = next(model.parameters()).device
    predictions = []
    reaction_ids = []
    r_dirs = []

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to the same device as the model
            batch = batch.to(device)

            # Extract data from batch
            pos0, pos1, pos2 = batch.pos0, batch.pos1, batch.pos2
            z0, z1, z2 = batch.z0, batch.z1, batch.z2
            batch_mapping = batch.batch
            xtb_features = getattr(batch, 'xtb_features', None)

            # Get model output
            _, _, predictions_batch = model(pos0, pos1, pos2, z0, z1, z2, batch_mapping, xtb_features)

            # Store predictions and identifiers
            predictions.append(predictions_batch.cpu().numpy())
            reaction_ids.extend([getattr(data, 'reaction_id') for data in batch.to_data_list()])
            r_dirs.extend([getattr(data, 'id') for data in batch.to_data_list()])

    # Concatenate all predictions
    all_predictions = np.vstack(predictions)

    # Inverse scale if model has scalers
    unscaled_predictions = np.zeros_like(all_predictions)
    if hasattr(model, 'scaler') and model.scaler is not None:
        for i in range(all_predictions.shape[1]):
            if i < len(model.scaler):
                unscaled_predictions[:, i:i + 1] = model.scaler[i].inverse_transform(all_predictions[:, i:i + 1])
    else:
        unscaled_predictions = all_predictions

    # Create a DataFrame with the results
    target_names = model.target_field_names if hasattr(model, 'target_field_names') and model.target_field_names else [
        f"target_{i}" for i in range(all_predictions.shape[1])]

    results = {
        'reaction_id': reaction_ids,
        'R_dir': r_dirs,
    }

    for i, target_name in enumerate(target_names):
        results[f"{target_name}_pred"] = unscaled_predictions[:, i]

    results_df = pd.DataFrame(results)

    # Save to file if requested
    if output_file:
        results_df.to_csv(output_file, index=False)
        print(f"Predictions saved to {output_file}")

    if return_results:
        return results_df
    else:
        return None


def parse_args():
    parser = argparse.ArgumentParser(description='Run inference with a trained model on new data')

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to the model checkpoint file (.ckpt)')
    parser.add_argument('--csv_file', type=str, required=True,
                        help='Path to the CSV file with new data for inference')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory containing the xyz files')
    parser.add_argument('--output', type=str, default='predictions.csv',
                        help='Path to save the output predictions')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for inference')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='Use CUDA for inference if available')
    parser.add_argument('--no_cuda', action='store_false', dest='cuda',
                        help='Do not use CUDA for inference')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    parser.add_argument('--reactant_suffix', type=str, default='_reactant.xyz',
                        help='File suffix for reactant xyz files')
    parser.add_argument('--ts_suffix', type=str, default='_ts.xyz',
                        help='File suffix for transition state xyz files')
    parser.add_argument('--product_suffix', type=str, default='_product.xyz',
                        help='File suffix for product xyz files')
    parser.add_argument('--input_features', type=str, nargs='+', default=['G(TS)_xtb', 'DrG_xtb'],
                        help='Input feature columns to read from CSV')

    return parser.parse_args()


def main():
    args = parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger('inference')

    logger.info(f"Loading model from checkpoint: {args.checkpoint}")
    model = load_model(args.checkpoint, args.cuda)

    logger.info(f"Loading data from CSV: {args.csv_file}")
    dataset = InferenceDataset(
        root=args.data_root,
        csv_file=args.csv_file,
        file_suffixes=[args.reactant_suffix, args.ts_suffix, args.product_suffix],
        input_features=args.input_features
    )

    logger.info(f"Loaded {len(dataset)} samples for inference")
    dataloader = create_inference_dataloader(dataset, args.batch_size, args.num_workers)

    logger.info("Running inference...")
    predictions = run_inference(model, dataloader, args.output)

    logger.info(f"Inference completed. Results saved to {args.output}")


if __name__ == "__main__":
    main()