import os
import torch
import numpy as np
import random
from typing import Optional, Tuple, List, Dict, Any
import logging

from .config import Config
from ..data.load_Reaction import load_reaction


class ReactionDataset:
    def __init__(self, config: Config):
        self.config = config
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.scalers = None
        self.fold_datasets = None
        self._setup_logging()
        self._load_data()
    
    def _setup_logging(self):
        self.logger = logging.getLogger('deepreaction')
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(getattr(logging, self.config.system.log_level.upper()))
    
    def _load_data(self):
        torch.manual_seed(self.config.training.random_seed)
        np.random.seed(self.config.training.random_seed)
        random.seed(self.config.training.random_seed)
        
        self.logger.info("Loading reaction dataset...")
        
        try:
            result = load_reaction(
                random_seed=self.config.training.random_seed,
                root=self.config.dataset.dataset_root,
                dataset_csv=self.config.dataset.dataset_csv,
                train_ratio=self.config.dataset.train_ratio,
                val_ratio=self.config.dataset.val_ratio,
                test_ratio=self.config.dataset.test_ratio,
                use_scaler=self.config.dataset.use_scaler,
                target_fields=self.config.dataset.target_fields,
                file_suffixes=['_reactant.xyz', '_ts.xyz', '_product.xyz'],
                input_features=self.config.dataset.input_features,
                cv_folds=self.config.dataset.cv_folds,
                val_csv=self.config.dataset.val_csv,
                test_csv=self.config.dataset.test_csv,
                cv_test_fold=self.config.dataset.cv_test_fold,
                cv_stratify=self.config.dataset.cv_stratify,
                cv_grouped=self.config.dataset.cv_grouped
            )
            
            if self.config.dataset.cv_folds > 0:
                self.fold_datasets = result
                self.logger.info(f"Loaded {len(self.fold_datasets)} CV folds")
            else:
                self.train_data, self.val_data, self.test_data, self.scalers = result
                self.logger.info(f"Loaded train: {len(self.train_data)}, val: {len(self.val_data)}, test: {len(self.test_data)}")
                
        except Exception as e:
            self.logger.error(f"Error loading dataset: {e}")
            raise
    
    def get_num_folds(self) -> int:
        return self.config.dataset.cv_folds if self.fold_datasets else 0
    
    def get_fold_data(self, fold_idx: int) -> Dict[str, Any]:
        if not self.fold_datasets or fold_idx >= len(self.fold_datasets):
            raise ValueError(f"Invalid fold index: {fold_idx}")
        return self.fold_datasets[fold_idx]
    
    def get_data_splits(self) -> Tuple[List, List, List, Optional[List]]:
        if self.fold_datasets:
            fold_data = self.fold_datasets[0]
            return fold_data['train'], fold_data['val'], fold_data['test'], fold_data['scalers']
        return self.train_data, self.val_data, self.test_data, self.scalers
    
    def get_num_targets(self) -> int:
        return len(self.config.dataset.target_fields)
    
    def get_num_features(self) -> int:
        return len(self.config.dataset.input_features)