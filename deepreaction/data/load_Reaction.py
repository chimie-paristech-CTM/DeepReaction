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
    if not ds_list:
        return None
    
    first_item = ds_list[0]
    if not hasattr(first_item, 'y') or first_item.y is None:
        return None
        
    num_targets = first_item.y.shape[1] if len(first_item.y.shape) > 1 else 1
    
    scalers = []
    for i in range(num_targets):
        try:
            ys = []
            for data in ds_list:
                if hasattr(data, 'y') and data.y is not None:
                    if len(data.y.shape) > 1:
                        ys.append(data.y[0, i].item())
                    else:
                        ys.append(data.y.item())
            
            if ys:
                ys = np.array(ys).reshape(-1, 1)
                scaler = StandardScaler().fit(ys)
                scalers.append(scaler)
                
                logger = logging.getLogger('deepreaction')
                logger.info(f"Scaler {i}: mean={scaler.mean_[0]:.4f}, std={np.sqrt(scaler.var_[0]):.4f}")
            else:
                scalers.append(None)
        except Exception as e:
            logger = logging.getLogger('deepreaction')
            logger.error(f"Error training scaler {i}: {e}")
            scalers.append(None)
    
    return scalers

def scale_reaction_dataset(ds_list, scalers):
    if not ds_list:
        return []
    
    if not scalers:
        return ds_list
    
    if not ds_list[0].y.shape or len(ds_list[0].y.shape) < 2:
        num_targets = 1
    else:
        num_targets = ds_list[0].y.shape[1]
    
    new_data_list = []
    for data in ds_list:
        try:
            if hasattr(data, 'y') and data.y is not None:
                if len(data.y.shape) == 1:
                    data.y = data.y.unsqueeze(0)
                
                y_scaled = torch.zeros_like(data.y)
                for i in range(min(num_targets, data.y.shape[1])):
                    y_val = data.y[0, i].item()
                    if i < len(scalers) and scalers[i] is not None:
                        try:
                            y_scaled[0, i] = torch.tensor(scalers[i].transform([[y_val]])[0, 0], dtype=torch.float)
                        except Exception as e:
                            logger = logging.getLogger('deepreaction')
                            logger.warning(f"Failed to scale target {i}: {e}, using original values")
                            y_scaled[0, i] = torch.tensor(y_val, dtype=torch.float)
                    else:
                        y_scaled[0, i] = torch.tensor(y_val, dtype=torch.float)
            else:
                y_scaled = torch.zeros((1, num_targets), dtype=torch.float)
            
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
                    
                    if j < len(scalers) and scalers[j] is not None:
                        try:
                            scaled_column = scalers[j].transform(feature_column)
                        except Exception as e:
                            logger = logging.getLogger('deepreaction')
                            logger.warning(f"Failed to scale feature {j}: {e}, using original values")
                            scaled_column = feature_column
                    else:
                        scaled_column = feature_column
                    
                    xtb_features_scaled[:, j] = scaled_column.flatten()
                
                d.xtb_features = torch.tensor(xtb_features_scaled, dtype=torch.float)
            
            for attr in ['feature_names', 'reaction_id', 'id', 'smiles']:
                if hasattr(data, attr):
                    setattr(d, attr, getattr(data, attr))
                    
            new_data_list.append(d)
            
        except Exception as e:
            logger = logging.getLogger('deepreaction')
            logger.error(f"Error scaling data item: {e}")
            continue

    return new_data_list

def check_reaction_id_overlap(train_data, val_data, test_data):
    logger = logging.getLogger('deepreaction')
    
    train_reaction_ids = {data.reaction_id for data in train_data if hasattr(data, 'reaction_id')}
    val_reaction_ids = {data.reaction_id for data in val_data if hasattr(data, 'reaction_id')}
    test_reaction_ids = {data.reaction_id for data in test_data if hasattr(data, 'reaction_id')}
    
    train_val_overlap = train_reaction_ids.intersection(val_reaction_ids)
    train_test_overlap = train_reaction_ids.intersection(test_reaction_ids)
    val_test_overlap = val_reaction_ids.intersection(test_reaction_ids)
    
    has_overlap = False
    
    if train_val_overlap:
        logger.warning(f"WARNING: {len(train_val_overlap)} reaction_ids overlap between train and validation sets")
        has_overlap = True
    
    if train_test_overlap:
        logger.warning(f"WARNING: {len(train_test_overlap)} reaction_ids overlap between train and test sets")
        has_overlap = True
    
    if val_test_overlap:
        logger.warning(f"WARNING: {len(val_test_overlap)} reaction_ids overlap between validation and test sets")
        has_overlap = True
    
    return not has_overlap

def load_dataset(root, csv_file, target_fields=None, file_keywords=None, input_features=None, force_reload=False,
                 id_field='ID', dir_field='R_dir', reaction_field='smiles'):
    force_reload = force_reload or (target_fields and len(target_fields) > 1)
    
    try:
        dataset = ReactionXYZDataset(
            root=root, 
            csv_file=csv_file,
            target_fields=target_fields,
            file_keywords=file_keywords,
            input_features=input_features,
            force_reload=force_reload,
            id_field=id_field,
            dir_field=dir_field,
            reaction_field=reaction_field
        )
    except Exception as e:
        logger = logging.getLogger('deepreaction')
        logger.warning(f"Error loading dataset: {e}, forcing reload...")
        try:
            dataset = ReactionXYZDataset(
                root=root, 
                csv_file=csv_file,
                target_fields=target_fields,
                file_keywords=file_keywords,
                input_features=input_features,
                force_reload=True,
                id_field=id_field,
                dir_field=dir_field,
                reaction_field=reaction_field
            )
        except Exception as reload_error:
            logger.error(f"Failed to reload dataset: {reload_error}")
            raise reload_error
        
    return dataset

def create_data_split(dataset, indices, ensure_2d=True):
    data_list = []
    failed_indices = []
    
    for i in indices:
        try:
            data = dataset[i]
            if data is None:
                failed_indices.append(i)
                continue
                
            if ensure_2d and hasattr(data, 'y') and data.y is not None and len(data.y.shape) == 1:
                data.y = data.y.unsqueeze(0)
            data_list.append(data)
        except Exception as e:
            logger = logging.getLogger('deepreaction')
            logger.error(f"Error accessing item at index {i}: {e}")
            failed_indices.append(i)
    
    if failed_indices:
        logger = logging.getLogger('deepreaction')
        logger.warning(f"Failed to load {len(failed_indices)} samples: {failed_indices[:10]}{'...' if len(failed_indices) > 10 else ''}")
        
        if len(failed_indices) > len(indices) * 0.1:
            raise RuntimeError(f"Too many failed samples ({len(failed_indices)}/{len(indices)}). Data may be corrupted.")
    
    return data_list

def prepare_data_splits(dataset, train_indices, val_indices, test_indices, use_scaler=False):
    train_data = create_data_split(dataset, train_indices)
    val_data = create_data_split(dataset, val_indices)
    test_data = create_data_split(dataset, test_indices)
    
    if len(train_data) == 0:
        raise ValueError("No training data could be loaded")
    
    logger = logging.getLogger('deepreaction')
    logger.info(f"Data splits: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")
    
    scalers = train_scaler(train_data) if use_scaler else None
    
    if scalers:
        logger.info(f"Trained {len(scalers)} scalers")
    
    train_scaled = scale_reaction_dataset(train_data, scalers)
    val_scaled = scale_reaction_dataset(val_data, scalers)
    test_scaled = scale_reaction_dataset(test_data, scalers)
    
    return train_scaled, val_scaled, test_scaled, scalers

def load_reaction(
    random_seed,
    root,
    dataset_csv='DA_dataset_cleaned.csv',
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    use_scaler=False,
    target_fields=None,
    file_keywords=None,
    input_features=None,
    cv_folds=0,
    val_csv=None,
    test_csv=None,
    cv_test_fold=-1,
    cv_stratify=False,
    cv_grouped=True,
    id_field='ID',
    dir_field='R_dir',
    reaction_field='smiles'
):
    logger = logging.getLogger('deepreaction')
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    if cv_folds == 0 and not (val_csv and test_csv):
        split_sum = train_ratio + val_ratio + test_ratio
        if abs(split_sum - 1.0) > 1e-6:
            raise ValueError(f"Train, validation, and test ratios must sum to 1.0, got {split_sum}")
    
    if cv_folds > 0:
        logger.info(f"Setting up {cv_folds}-fold cross-validation")
        dataset = load_dataset(
            root=root, 
            csv_file=dataset_csv,
            target_fields=target_fields,
            file_keywords=file_keywords,
            input_features=input_features,
            id_field=id_field,
            dir_field=dir_field,
            reaction_field=reaction_field
        )
        
        raise NotImplementedError("Cross-validation not implemented in this version")
    
    if val_csv and test_csv:
        logger.info(f"Loading separate train/val/test datasets from CSVs")
        
        try:
            train_dataset = load_dataset(
                root=root, 
                csv_file=dataset_csv,
                target_fields=target_fields,
                file_keywords=file_keywords,
                input_features=input_features,
                id_field=id_field,
                dir_field=dir_field,
                reaction_field=reaction_field
            )
            
            val_dataset = load_dataset(
                root=root, 
                csv_file=val_csv,
                target_fields=target_fields,
                file_keywords=file_keywords,
                input_features=input_features,
                id_field=id_field,
                dir_field=dir_field,
                reaction_field=reaction_field
            )
            
            test_dataset = load_dataset(
                root=root, 
                csv_file=test_csv,
                target_fields=target_fields,
                file_keywords=file_keywords,
                input_features=input_features,
                id_field=id_field,
                dir_field=dir_field,
                reaction_field=reaction_field
            )
        except Exception as e:
            logger.error(f"Failed to load separate datasets: {e}")
            raise
        
        train_data = create_data_split(train_dataset, range(len(train_dataset)))
        val_data = create_data_split(val_dataset, range(len(val_dataset)))
        test_data = create_data_split(test_dataset, range(len(test_dataset)))
        
        logger.info(f"Loaded {len(train_data)} training, {len(val_data)} validation, and {len(test_data)} test samples from separate CSVs")
        
        check_reaction_id_overlap(train_data, val_data, test_data)
        
        scalers = train_scaler(train_data) if use_scaler else None
        
        train_scaled = scale_reaction_dataset(train_data, scalers)
        val_scaled = scale_reaction_dataset(val_data, scalers)
        test_scaled = scale_reaction_dataset(test_data, scalers)
        
        return train_scaled, val_scaled, test_scaled, scalers
    
    logger.info(f"Loading single dataset with automatic train/val/test split")
    dataset = load_dataset(
        root=root, 
        csv_file=dataset_csv,
        target_fields=target_fields,
        file_keywords=file_keywords,
        input_features=input_features,
        id_field=id_field,
        dir_field=dir_field,
        reaction_field=reaction_field
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


def load_reaction_for_inference(
        random_seed,
        root,
        dataset_csv,
        file_keywords=None,
        input_features=None,
        target_fields=None,
        scaler=None,
        force_reload=False,
        id_field='ID',
        dir_field='R_dir',
        reaction_field='smiles'
):
    logger = logging.getLogger('deepreaction')
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    logger.info(f"Loading inference dataset from {dataset_csv}")
    
    if not osp.exists(dataset_csv):
        raise FileNotFoundError(f"Inference CSV file not found: {dataset_csv}")
    
    try:
        dataset = load_inference_dataset(
            root=root,
            csv_file=dataset_csv,
            file_keywords=file_keywords,
            input_features=input_features,
            target_fields=None,
            force_reload=force_reload,
            id_field=id_field,
            dir_field=dir_field,
            reaction_field=reaction_field
        )
    except Exception as e:
        logger.error(f"Failed to load inference dataset: {e}")
        raise
    
    data_list = create_data_split(dataset, range(len(dataset)))
    
    if scaler is not None:
        data_list = scale_inference_dataset(data_list, scaler)
    
    logger.info(f"Loaded {len(data_list)} samples for inference")
    
    return data_list


def load_inference_dataset(root, csv_file, file_keywords=None, input_features=None, target_fields=None, force_reload=False,
                          id_field='ID', dir_field='R_dir', reaction_field='smiles'):
    try:
        dataset = ReactionXYZDataset(
            root=root,
            csv_file=csv_file,
            target_fields=None,
            file_keywords=file_keywords,
            input_features=input_features,
            force_reload=force_reload,
            inference_mode=True,
            id_field=id_field,
            dir_field=dir_field,
            reaction_field=reaction_field
        )
    except Exception as e:
        logger = logging.getLogger('deepreaction')
        logger.warning(f"Error loading dataset: {e}, forcing reload...")
        try:
            dataset = ReactionXYZDataset(
                root=root,
                csv_file=csv_file,
                target_fields=None,
                file_keywords=file_keywords,
                input_features=input_features,
                force_reload=True,
                inference_mode=True,
                id_field=id_field,
                dir_field=dir_field,
                reaction_field=reaction_field
            )
        except Exception as reload_error:
            logger.error(f"Failed to reload inference dataset: {reload_error}")
            raise reload_error

    return dataset


def scale_inference_dataset(ds_list, scalers):
    if not ds_list or not scalers:
        return ds_list

    new_data_list = []
    for data in ds_list:
        try:
            d = Data(
                z0=data.z0,
                z1=data.z1,
                z2=data.z2,
                pos0=data.pos0,
                pos1=data.pos1,
                pos2=data.pos2,
                y=data.y,
                num_nodes=data.num_nodes
            )

            if hasattr(data, 'xtb_features'):
                xtb_features_value = data.xtb_features.numpy()
                xtb_features_scaled = np.zeros_like(xtb_features_value)

                num_features = xtb_features_value.shape[1]
                for j in range(num_features):
                    feature_column = xtb_features_value[:, j].reshape(-1, 1)
                    if j < len(scalers) and scalers[j] is not None:
                        try:
                            scaled_column = scalers[j].transform(feature_column)
                        except Exception as e:
                            logger = logging.getLogger('deepreaction')
                            logger.warning(f"Failed to scale inference feature {j}: {e}, using original values")
                            scaled_column = feature_column
                    else:
                        scaled_column = feature_column

                    xtb_features_scaled[:, j] = scaled_column.flatten()

                d.xtb_features = torch.tensor(xtb_features_scaled, dtype=torch.float)

            for attr in ['feature_names', 'reaction_id', 'id', 'smiles']:
                if hasattr(data, attr):
                    setattr(d, attr, getattr(data, attr))

            new_data_list.append(d)
            
        except Exception as e:
            logger = logging.getLogger('deepreaction')
            logger.error(f"Error scaling inference data item: {e}")
            continue

    return new_data_list