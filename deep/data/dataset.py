import os
import torch
from torch_geometric.data import Data
import pandas as pd
import numpy as np
from typing import List, Optional, Union, Tuple

class ReactionDataset:
    def __init__(
        self,
        root: str,
        csv_file: str,
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
        force_reload: bool = False
    ):
        from ..data.load_Reaction import load_reaction
        
        self.root = root
        self.csv_file = csv_file
        self.is_inference = target_fields is None or len(target_fields) == 0
            
        self.target_fields = target_fields or ["dummy_target"]
        self.file_patterns = file_patterns or ['*_reactant.xyz', '*_ts.xyz', '*_product.xyz']
        self.input_features = input_features if input_features is not None else []
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
        
        if cv_folds > 0:
            self.fold_datasets = load_reaction(
                random_seed=random_seed,
                root=root,
                dataset_csv=csv_file,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                use_scaler=use_scaler,
                target_fields=self.target_fields,
                file_patterns=file_patterns,
                input_features=input_features,
                cv_folds=cv_folds,
                cv_test_fold=cv_test_fold,
                cv_stratify=cv_stratify,
                cv_grouped=cv_grouped,
                id_field=id_field,
                dir_field=dir_field,
                reaction_field=reaction_field
            )
            self.train_data = self.fold_datasets[0]['train']
            self.val_data = self.fold_datasets[0]['val']
            self.test_data = self.fold_datasets[0]['test']
            self.scalers = self.fold_datasets[0]['scalers']
            self.current_fold = 0
        else:
            result = load_reaction(
                random_seed=random_seed,
                root=root,
                dataset_csv=csv_file,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                use_scaler=use_scaler,
                target_fields=self.target_fields,
                file_patterns=file_patterns,
                input_features=input_features,
                id_field=id_field,
                dir_field=dir_field,
                reaction_field=reaction_field
            )
            
            if isinstance(result, tuple) and len(result) == 4:
                self.train_data, self.val_data, self.test_data, self.scalers = result
            else:
                # Handle the case where result is just test data for inference
                self.train_data = []
                self.val_data = []
                self.test_data = result
                self.scalers = None
                
            self.fold_datasets = None
    
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
            raise ValueError(f"Invalid fold index: {fold_idx}. Dataset has {len(self.fold_datasets) if self.fold_datasets else 0} folds")
        
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
                            "target": self.target_fields[i] if self.target_fields and i < len(self.target_fields) else f"target_{i}",
                            "mean": float(scaler.mean_[0]) if hasattr(scaler.mean_, '__iter__') else float(scaler.mean_),
                            "scale": float(scaler.scale_[0]) if hasattr(scaler.scale_, '__iter__') else float(scaler.scale_)
                        })
            else:
                if hasattr(self.scalers, 'mean_') and hasattr(self.scalers, 'scale_'):
                    stats["scaler_info"] = {
                        "mean": float(self.scalers.mean_[0]) if hasattr(self.scalers.mean_, '__iter__') else float(self.scalers.mean_),
                        "scale": float(self.scalers.scale_[0]) if hasattr(self.scalers.scale_, '__iter__') else float(self.scalers.scale_)
                    }
        
        return stats