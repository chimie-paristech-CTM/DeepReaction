import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Optional, Union, Any, Dict
import os.path as osp
import logging
from collections import defaultdict
import pandas as pd
import random

from .PygReaction import ReactionXYZDataset


def train_scaler(ds_list):
    num_targets = ds_list[0].y.shape[1] if len(ds_list) > 0 else 0

    scalers = []
    for i in range(num_targets):
        ys = np.array([data.y[0, i].item() for data in ds_list]).reshape(-1, 1)
        scaler = StandardScaler().fit(ys)
        scalers.append(scaler)

    return scalers


def scale_reaction_dataset(ds_list, scalers):
    if not ds_list:
        return []

    num_targets = ds_list[0].y.shape[1]

    new_data_list = []
    for data in ds_list:
        y_scaled = torch.zeros_like(data.y)
        for i in range(num_targets):
            y_val = data.y[0, i].item()
            if scalers and i < len(scalers):
                y_scaled[0, i] = torch.tensor(scalers[i].transform([[y_val]])[0, 0], dtype=torch.float)
            else:
                y_scaled[0, i] = torch.tensor(y_val, dtype=torch.float)

        d = Data(
            z0=data.z0,
            z1=data.z1,
            z2=data.z2,
            pos0=data.pos0,
            pos1=data.pos1,
            pos2=data.pos2,
            y=y_scaled,
            num_nodes=data.num_nodes
        )

        if hasattr(data, 'xtb_features'):
            xtb_features_value = data.xtb_features.numpy()
            xtb_features_scaled = np.zeros_like(xtb_features_value)

            num_features = xtb_features_value.shape[1]
            for j in range(num_features):
                feature_column = xtb_features_value[:, j].reshape(-1, 1)
                if data.y.shape[1] == 1:
                    scaled_column = scalers[0].transform(feature_column)
                elif scalers and j < len(scalers):
                    scaled_column = scalers[j].transform(feature_column)
                else:
                    scaled_column = feature_column

                xtb_features_scaled[:, j] = scaled_column.flatten()

            d.xtb_features = torch.tensor(xtb_features_scaled, dtype=torch.float)

        for attr in ['feature_names', 'reaction_id', 'id', 'reaction']:
            if hasattr(data, attr):
                setattr(d, attr, getattr(data, attr))

        new_data_list.append(d)

    return new_data_list


def check_reaction_id_overlap(train_data, val_data, test_data):
    logger = logging.getLogger('deep')

    train_reaction_ids = {data.reaction_id for data in train_data if hasattr(data, 'reaction_id')}
    val_reaction_ids = {data.reaction_id for data in val_data if hasattr(data, 'reaction_id')}
    test_reaction_ids = {data.reaction_id for data in test_data if hasattr(data, 'reaction_id')}

    train_val_overlap = train_reaction_ids.intersection(val_reaction_ids)
    train_test_overlap = train_reaction_ids.intersection(test_reaction_ids)
    val_test_overlap = val_reaction_ids.intersection(test_reaction_ids)

    has_overlap = False

    if train_val_overlap:
        logger.warning(
            f"WARNING: {len(train_val_overlap)} reaction_ids overlap between train and validation sets: {sorted(list(train_val_overlap))[:5]}{' (and more...)' if len(train_val_overlap) > 5 else ''}")
        has_overlap = True

    if train_test_overlap:
        logger.warning(
            f"WARNING: {len(train_test_overlap)} reaction_ids overlap between train and test sets: {sorted(list(train_test_overlap))[:5]}{' (and more...)' if len(train_test_overlap) > 5 else ''}")
        has_overlap = True

    if val_test_overlap:
        logger.warning(
            f"WARNING: {len(val_test_overlap)} reaction_ids overlap between validation and test sets: {sorted(list(val_test_overlap))[:5]}{' (and more...)' if len(val_test_overlap) > 5 else ''}")
        has_overlap = True

    return not has_overlap


def load_dataset(root, csv_file, target_fields=None, file_suffixes=None, input_features=None, force_reload=False):
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
        print(f"Error loading dataset: {e}")
        dataset = ReactionXYZDataset(
            root=root,
            csv_file=csv_file,
            target_fields=target_fields,
            file_suffixes=file_suffixes,
            input_features=input_features,
            force_reload=True
        )

    return dataset


def create_data_split(dataset, indices, ensure_2d=True):
    data_list = []
    for i in indices:
        try:
            data = dataset[i]
            if ensure_2d and len(data.y.shape) == 1:
                data.y = data.y.unsqueeze(0)
            data_list.append(data)
        except Exception as e:
            print(f"Error accessing item at index {i}: {e}")

    return data_list


def prepare_data_splits(dataset, train_indices, val_indices, test_indices, use_scaler=False):
    train_data = create_data_split(dataset, train_indices)
    val_data = create_data_split(dataset, val_indices)
    test_data = create_data_split(dataset, test_indices)

    if len(train_data) == 0:
        raise ValueError("No training data could be loaded")

    scalers = train_scaler(train_data) if use_scaler else None

    train_scaled = scale_reaction_dataset(train_data, scalers)
    val_scaled = scale_reaction_dataset(val_data, scalers)
    test_scaled = scale_reaction_dataset(test_data, scalers)

    return train_scaled, val_scaled, test_scaled, scalers


def create_cv_splits(dataset, cv_folds, random_seed, val_ratio=0.1, train_ratio=0.8, grouped=True):
    logger = logging.getLogger('deep')

    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    random.seed(random_seed)

    reaction_groups = {}
    for i in range(len(dataset)):
        data = dataset[i]
        if not hasattr(data, 'reaction_id'):
            raise ValueError("No reaction_id attribute in dataset")

        reaction_id = data.reaction_id
        if reaction_id not in reaction_groups:
            reaction_groups[reaction_id] = []
        reaction_groups[reaction_id].append(i)

    reaction_ids = list(reaction_groups.keys())
    reaction_sizes = {r_id: len(indices) for r_id, indices in reaction_groups.items()}

    random.shuffle(reaction_ids)
    reaction_ids = sorted(reaction_ids, key=lambda x: reaction_sizes[x], reverse=True)

    fold_sizes = [0] * cv_folds
    fold_assignments = {}

    for r_id in reaction_ids:
        min_fold = np.argmin(fold_sizes)
        fold_assignments[r_id] = min_fold
        fold_sizes[min_fold] += reaction_sizes[r_id]

    logger.info(f"Balanced fold sizes: {fold_sizes}")

    cv_splits = []

    for fold_idx in range(cv_folds):
        test_reactions = [r_id for r_id, assigned_fold in fold_assignments.items() if assigned_fold == fold_idx]
        remaining_reactions = [r_id for r_id, assigned_fold in fold_assignments.items() if assigned_fold != fold_idx]

        test_indices = []
        for r_id in test_reactions:
            test_indices.extend(reaction_groups[r_id])

        remaining_sample_count = sum(reaction_sizes[r_id] for r_id in remaining_reactions)
        val_sample_target = int(remaining_sample_count * val_ratio / (val_ratio + train_ratio))

        remaining_reactions = sorted(remaining_reactions, key=lambda x: reaction_sizes[x], reverse=True)

        val_reactions = []
        val_samples_so_far = 0

        for r_id in remaining_reactions:
            if val_samples_so_far < val_sample_target:
                val_reactions.append(r_id)
                val_samples_so_far += reaction_sizes[r_id]
            else:
                break

        train_reactions = [r_id for r_id in remaining_reactions if r_id not in val_reactions]

        val_indices = []
        for r_id in val_reactions:
            val_indices.extend(reaction_groups[r_id])

        train_indices = []
        for r_id in train_reactions:
            train_indices.extend(reaction_groups[r_id])

        split = {
            'fold': fold_idx,
            'train': torch.tensor(train_indices, dtype=torch.long),
            'val': torch.tensor(val_indices, dtype=torch.long),
            'test': torch.tensor(test_indices, dtype=torch.long)
        }

        cv_splits.append(split)

        logger.info(
            f"Fold {fold_idx}: {len(train_indices)} train, {len(val_indices)} val, {len(test_indices)} test samples")

    return cv_splits


def load_reaction(
        random_seed,
        root,
        dataset_csv='DA_dataset_cleaned.csv',
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        use_scaler=False,
        target_fields=None,
        file_suffixes=None,
        input_features=None,
        cv_folds=0,
        val_csv=None,
        test_csv=None,
        cv_test_fold=-1,
        cv_stratify=False,
        cv_grouped=True,
):
    logger = logging.getLogger('deep')
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    # Cross-validation mode
    if cv_folds > 0:
        logger.info(f"Setting up {cv_folds}-fold cross-validation")
        dataset = load_dataset(
            root=root,
            csv_file=dataset_csv,
            target_fields=target_fields,
            file_suffixes=file_suffixes,
            input_features=input_features
        )

        cv_splits = create_cv_splits(
            dataset=dataset,
            cv_folds=cv_folds,
            random_seed=random_seed,
            val_ratio=val_ratio,
            train_ratio=train_ratio,
            grouped=cv_grouped
        )

        fold_datasets = []
        for fold_idx, fold_split in enumerate(cv_splits):
            logger.info(f"Preparing data for fold {fold_idx + 1}/{cv_folds}")

            train_data, val_data, test_data, scalers = prepare_data_splits(
                dataset=dataset,
                train_indices=fold_split['train'],
                val_indices=fold_split['val'],
                test_indices=fold_split['test'],
                use_scaler=use_scaler
            )

            fold_datasets.append({
                'fold': fold_idx,
                'train': train_data,
                'val': val_data,
                'test': test_data,
                'scalers': scalers
            })

        return fold_datasets

    # Separate validation and test files
    if val_csv and test_csv:
        logger.info(f"Loading separate train/val/test datasets from CSVs")

        train_dataset = load_dataset(
            root=root,
            csv_file=dataset_csv,
            target_fields=target_fields,
            file_suffixes=file_suffixes,
            input_features=input_features
        )

        val_dataset = load_dataset(
            root=root,
            csv_file=val_csv,
            target_fields=target_fields,
            file_suffixes=file_suffixes,
            input_features=input_features
        )

        test_dataset = load_dataset(
            root=root,
            csv_file=test_csv,
            target_fields=target_fields,
            file_suffixes=file_suffixes,
            input_features=input_features
        )

        train_data = create_data_split(train_dataset, range(len(train_dataset)))
        val_data = create_data_split(val_dataset, range(len(val_dataset)))
        test_data = create_data_split(test_dataset, range(len(test_dataset)))

        logger.info(
            f"Loaded {len(train_data)} training, {len(val_data)} validation, and {len(test_data)} test samples from separate CSVs")

        check_reaction_id_overlap(train_data, val_data, test_data)

        scalers = train_scaler(train_data) if use_scaler else None

        train_scaled = scale_reaction_dataset(train_data, scalers)
        val_scaled = scale_reaction_dataset(val_data, scalers)
        test_scaled = scale_reaction_dataset(test_data, scalers)

        return train_scaled, val_scaled, test_scaled, scalers

    # Single dataset with automatic split
    logger.info(f"Loading single dataset with automatic train/val/test split")
    dataset = load_dataset(
        root=root,
        csv_file=dataset_csv,
        target_fields=target_fields,
        file_suffixes=file_suffixes,
        input_features=input_features
    )

    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)

    splits = dataset.get_idx_split(train_size, val_size, seed=random_seed)

    return prepare_data_splits(
        dataset=dataset,
        train_indices=splits['train'],
        val_indices=splits['valid'],
        test_indices=splits['test'],
        use_scaler=use_scaler
    )