# deepreaction/data/dataset.py
import os
import torch
from torch_geometric.data import Data
import pandas as pd
import numpy as np
from typing import List, Optional, Union, Tuple, Dict, Any


class ReactionDataset:
    def __init__(
            self,
            root: str = "./dataset",
            csv_file: str = "./dataset/data.csv",
            target_fields: Optional[List[str]] = None,
            file_patterns: Optional[List[str]] = None,
            input_features: Optional[List[str]] = None,
            use_scaler: bool = True,
            random_seed: int = 42,
            train_ratio: float = 0.8,
            val_ratio: float = 0.1,
            test_ratio: float = 0.1,
            cv_folds: int = 0,
            cv_test_fold: int = -1,
            cv_stratify: bool = False,
            cv_grouped: bool = True,
            id_field: str = 'ID',
            dir_field: str = 'R_dir',
            reaction_field: str = 'reaction',
            force_reload: bool = False,
            inference_mode: bool = False,
            config: Optional[Any] = None
    ):
        """
        Initialize the ReactionDataset.
        
        Args:
            root: Root directory containing XYZ files
            csv_file: CSV file with reaction data
            target_fields: List of target property fields
            file_patterns: List of file patterns for XYZ files
            input_features: List of input feature columns
            use_scaler: Whether to scale the target values
            random_seed: Random seed for reproducibility
            train_ratio: Ratio of training data
            val_ratio: Ratio of validation data
            test_ratio: Ratio of test data
            cv_folds: Number of cross-validation folds (0 for standard split)
            cv_test_fold: Which fold to use as test set in CV
            cv_stratify: Whether to stratify the CV splits
            cv_grouped: Whether to group reactions in CV
            id_field: Column name for reaction IDs
            dir_field: Column name for subdirectory names
            reaction_field: Column name for reaction SMILES
            force_reload: Whether to force reload the dataset
            inference_mode: Whether to run in inference mode
            config: Optional Config object to load parameters from
        """
        from deepreaction.data.load_Reaction import load_reaction, load_reaction_for_inference
        
        # If config is provided, use its parameters
        if config is not None:
            from deepreaction.config.config import Config
            if hasattr(config, 'reaction'):
                self.root = config.reaction.dataset_root
                self.csv_file = config.reaction.dataset_csv
                self.target_fields = config.reaction.target_fields
                self.file_patterns = config.reaction.file_patterns
                self.input_features = config.reaction.input_features
                self.use_scaler = config.reaction.use_scaler
                self.random_seed = config.reaction.random_seed
                self.train_ratio = config.reaction.train_ratio
                self.val_ratio = config.reaction.val_ratio
                self.test_ratio = config.reaction.test_ratio
                self.cv_folds = config.reaction.cv_folds
                self.cv_test_fold = config.reaction.cv_test_fold
                self.cv_stratify = config.reaction.cv_stratify
                self.cv_grouped = config.reaction.cv_grouped
                self.id_field = config.reaction.id_field
                self.dir_field = config.reaction.dir_field
                self.reaction_field = config.reaction.reaction_field
                self.force_reload = config.reaction.force_reload
                self.inference_mode = getattr(config.reaction, 'inference_mode', inference_mode)
            else:
                # Assume it's a dict-like object
                self.root = getattr(config, 'dataset_root', root)
                self.csv_file = getattr(config, 'dataset_csv', csv_file)
                self.target_fields = getattr(config, 'target_fields', target_fields)
                self.file_patterns = getattr(config, 'file_patterns', file_patterns)
                self.input_features = getattr(config, 'input_features', input_features)
                self.use_scaler = getattr(config, 'use_scaler', use_scaler)
                self.random_seed = getattr(config, 'random_seed', random_seed)
                self.train_ratio = getattr(config, 'train_ratio', train_ratio)
                self.val_ratio = getattr(config, 'val_ratio', val_ratio)
                self.test_ratio = getattr(config, 'test_ratio', test_ratio)
                self.cv_folds = getattr(config, 'cv_folds', cv_folds)
                self.cv_test_fold = getattr(config, 'cv_test_fold', cv_test_fold)
                self.cv_stratify = getattr(config, 'cv_stratify', cv_stratify)
                self.cv_grouped = getattr(config, 'cv_grouped', cv_grouped)
                self.id_field = getattr(config, 'id_field', id_field)
                self.dir_field = getattr(config, 'dir_field', dir_field)
                self.reaction_field = getattr(config, 'reaction_field', reaction_field)
                self.force_reload = getattr(config, 'force_reload', force_reload)
                self.inference_mode = getattr(config, 'inference_mode', inference_mode)
        else:
            # Use provided parameters
            self.root = root
            self.csv_file = csv_file
            self.target_fields = target_fields or ["G(TS)", "DrG"]
            self.file_patterns = file_patterns or ['*_reactant.xyz', '*_ts.xyz', '*_product.xyz']
            self.input_features = input_features or []
            self.use_scaler = use_scaler
            self.random_seed = random_seed
            self.train_ratio = train_ratio
            self.val_ratio = val_ratio
            self.test_ratio = test_ratio
            self.cv_folds = cv_folds
            self.cv_test_fold = cv_test_fold
            self.cv_stratify = cv_stratify
            self.cv_grouped = cv_grouped
            self.id_field = id_field
            self.dir_field = dir_field
            self.reaction_field = reaction_field
            self.force_reload = force_reload
            self.inference_mode = inference_mode

        self.is_inference = inference_mode

        # Ensure target_fields is a list
        if isinstance(self.target_fields, str):
            self.target_fields = [self.target_fields]

        # Load the dataset based on inference mode
        if self.is_inference:
            # Use load_reaction_for_inference for inference mode
            self.train_data = []
            self.val_data = []
            self.test_data = load_reaction_for_inference(
                random_seed=self.random_seed,
                root=self.root,
                dataset_csv=self.csv_file,
                file_patterns=self.file_patterns,
                file_dir_pattern=None,  # Add if needed in your implementation
                input_features=self.input_features,
                scaler=None,  # Pass scalers if needed
                force_reload=self.force_reload,
                id_field=self.id_field,
                dir_field=self.dir_field,
                reaction_field=self.reaction_field
            )
            self.scalers = None
            self.fold_datasets = None
        elif self.cv_folds > 0:
            # Use cross-validation
            self.fold_datasets = load_reaction(
                random_seed=self.random_seed,
                root=self.root,
                dataset_csv=self.csv_file,
                train_ratio=self.train_ratio,
                val_ratio=self.val_ratio,
                test_ratio=self.test_ratio,
                use_scaler=self.use_scaler,
                target_fields=self.target_fields,
                file_patterns=self.file_patterns,
                input_features=self.input_features,
                cv_folds=self.cv_folds,
                cv_test_fold=self.cv_test_fold,
                cv_stratify=self.cv_stratify,
                cv_grouped=self.cv_grouped,
                id_field=self.id_field,
                dir_field=self.dir_field,
                reaction_field=self.reaction_field
            )
            self.train_data = self.fold_datasets[0]['train']
            self.val_data = self.fold_datasets[0]['val']
            self.test_data = self.fold_datasets[0]['test']
            self.scalers = self.fold_datasets[0]['scalers']
            self.current_fold = 0
        else:
            # Standard train/val/test split
            result = load_reaction(
                random_seed=self.random_seed,
                root=self.root,
                dataset_csv=self.csv_file,
                train_ratio=self.train_ratio,
                val_ratio=self.val_ratio,
                test_ratio=self.test_ratio,
                use_scaler=self.use_scaler,
                target_fields=self.target_fields,
                file_patterns=self.file_patterns,
                input_features=self.input_features,
                id_field=self.id_field,
                dir_field=self.dir_field,
                reaction_field=self.reaction_field
            )

            # Handle the result format
            if isinstance(result, tuple) and len(result) == 4:
                self.train_data, self.val_data, self.test_data, self.scalers = result
            else:
                # Handle unexpected result format
                self.train_data = []
                self.val_data = []
                self.test_data = result if isinstance(result, list) else []
                self.scalers = None
            self.fold_datasets = None

    # Rest of the class methods remain the same...
    def get_train_data(self):
        return self.train_data

    def get_val_data(self):
        return self.val_data

    def get_test_data(self):
        return self.test_data

    def get_scalers(self):
        return self.scalers

    def set_fold(self, fold_idx):
        if self.fold_datasets is None or fold_idx >= len(self.fold_datasets):
            raise ValueError(
                f"Invalid fold index: {fold_idx}. Dataset has {len(self.fold_datasets) if self.fold_datasets else 0} folds")

        self.current_fold = fold_idx
        self.train_data = self.fold_datasets[fold_idx]['train']
        self.val_data = self.fold_datasets[fold_idx]['val']
        self.test_data = self.fold_datasets[fold_idx]['test']
        self.scalers = self.fold_datasets[fold_idx]['scalers']

    def get_current_fold(self):
        return self.current_fold

    def get_num_folds(self):
        return len(self.fold_datasets) if self.fold_datasets else 0

    def get_data_loaders(self, batch_size=32, num_workers=4):
        from torch_geometric.loader import DataLoader

        follow_batch = ['z0', 'z1', 'z2', 'pos0', 'pos1', 'pos2']

        if self.is_inference and self.test_data:
            # For inference mode, only create test loader with all data
            test_loader = DataLoader(
                self.test_data,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                follow_batch=follow_batch
            )
            return None, None, test_loader

        train_loader = None
        if self.train_data:
            train_loader = DataLoader(
                self.train_data,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                follow_batch=follow_batch
            )

        val_loader = None
        if self.val_data:
            val_loader = DataLoader(
                self.val_data,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                follow_batch=follow_batch
            )

        test_loader = None
        if self.test_data:
            test_loader = DataLoader(
                self.test_data,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                follow_batch=follow_batch
            )

        return train_loader, val_loader, test_loader

    def get_data_stats(self):
        stats = {
            "train_size": len(self.train_data) if self.train_data else 0,
            "val_size": len(self.val_data) if self.val_data else 0,
            "test_size": len(self.test_data) if self.test_data else 0,
            "target_fields": self.target_fields,
            "input_features": self.input_features,
            "num_folds": self.get_num_folds(),
            "current_fold": self.current_fold if self.fold_datasets else None,
            "is_inference": self.is_inference
        }

        if self.scalers is not None:
            if isinstance(self.scalers, list):
                stats["scaler_info"] = []
                for i, scaler in enumerate(self.scalers):
                    if hasattr(scaler, 'mean_') and hasattr(scaler, 'scale_'):
                        stats["scaler_info"].append({
                            "target": self.target_fields[i] if self.target_fields and i < len(
                                self.target_fields) else f"target_{i}",
                            "mean": float(scaler.mean_[0]) if hasattr(scaler.mean_, '__iter__') else float(
                                scaler.mean_),
                            "scale": float(scaler.scale_[0]) if hasattr(scaler.scale_, '__iter__') else float(
                                scaler.scale_)
                        })
            else:
                if hasattr(self.scalers, 'mean_') and hasattr(self.scalers, 'scale_'):
                    stats["scaler_info"] = {
                        "mean": float(self.scalers.mean_[0]) if hasattr(self.scalers.mean_, '__iter__') else float(
                            self.scalers.mean_),
                        "scale": float(self.scalers.scale_[0]) if hasattr(self.scalers.scale_, '__iter__') else float(
                            self.scalers.scale_)
                    }

        return stats