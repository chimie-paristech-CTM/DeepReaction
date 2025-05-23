import os
import torch
import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Optional
from torch_geometric.loader import DataLoader as GeometricDataLoader

from .config import Config
from ..module.pl_wrap import Estimator
from ..data.load_Reaction import load_reaction_for_inference


class ReactionPredictor:
    def __init__(self, config: Config, checkpoint_path: str):
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.logger = self._setup_logging()
        self._load_model()
    
    def _setup_logging(self):
        logger = logging.getLogger('deepreaction')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(getattr(logging, self.config.system.log_level.upper()))
        return logger
    
    def _load_model(self):
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file does not exist: {self.checkpoint_path}")
        
        self.logger.info(f"Loading model from checkpoint: {self.checkpoint_path}")
        try:
            self.model = Estimator.load_from_checkpoint(self.checkpoint_path)
        except Exception as e:
            self.logger.error(f"Failed to load model from checkpoint: {e}")
            raise
        
        if self.config.system.cuda and torch.cuda.is_available():
            device = torch.device(f"cuda:{self.config.system.gpu_id}")
            self.model = self.model.to(device)
        else:
            self.model = self.model.to('cpu')
        
        self.model.eval()
        
        if hasattr(self.model, 'target_field_names') and self.model.target_field_names:
            self.logger.info(f"Model target fields: {self.model.target_field_names}")
        if hasattr(self.model, 'scaler') and self.model.scaler:
            self.logger.info(f"Model has {len(self.model.scaler)} scalers")
    
    def _create_dataloader(self, dataset, batch_size):
        num_workers = min(self.config.system.num_workers, 2)
        
        loader_kwargs = {
            'batch_size': batch_size,
            'num_workers': num_workers,
            'worker_init_fn': None,
            'shuffle': False,
            'follow_batch': ['z0', 'z1', 'z2', 'pos0', 'pos1', 'pos2'],
            'pin_memory': self.config.system.cuda,
        }
        return GeometricDataLoader(dataset, **loader_kwargs)
    
    def _map_features_to_scaler_indices(self, input_features, model_target_fields):
        feature_to_scaler_map = {}
        
        self.logger.info(f"Mapping input features {input_features} to model targets {model_target_fields}")
        
        for i, feature in enumerate(input_features):
            if feature.endswith('_xtb'):
                key = feature[:-4]
            elif feature.endswith('_dft'):
                key = feature[:-4]
            elif feature.endswith('_calc'):
                key = feature[:-5]
            else:
                key = feature
            
            scaler_idx = i
            if model_target_fields and key in model_target_fields:
                scaler_idx = model_target_fields.index(key)
                self.logger.info(f"Mapped feature '{feature}' (key: '{key}') to scaler index {scaler_idx}")
            else:
                self.logger.warning(f"No exact match for feature '{feature}' (key: '{key}'), using default index {i}")
            
            feature_to_scaler_map[i] = scaler_idx
        
        return feature_to_scaler_map
    
    def _get_output_field_names(self, input_features):
        output_fields = []
        for feature in input_features:
            if feature.endswith('_xtb'):
                output_fields.append(feature[:-4])
            elif feature.endswith('_dft'):
                output_fields.append(feature[:-4])
            elif feature.endswith('_calc'):
                output_fields.append(feature[:-5])
            else:
                output_fields.append(feature)
        return output_fields
    
    def predict_from_csv(self, csv_path: str, output_dir: str = None) -> pd.DataFrame:
        if output_dir is None:
            output_dir = os.path.dirname(csv_path)
        
        self.logger.info(f"Loading inference data from: {csv_path}")
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        try:
            inference_data = load_reaction_for_inference(
                random_seed=self.config.training.random_seed,
                root=self.config.dataset.dataset_root,
                dataset_csv=csv_path,
                file_suffixes=self.config.dataset.file_suffixes,
                input_features=self.config.dataset.input_features,
                target_fields=None,
                scaler=self.model.scaler
            )
        except Exception as e:
            self.logger.error(f"Failed to load inference data: {e}")
            raise
        
        return self.predict(inference_data, output_dir)
    
    def predict(self, dataset: List, output_dir: str = None) -> pd.DataFrame:
        if output_dir is None:
            output_dir = './predictions'
        
        os.makedirs(output_dir, exist_ok=True)
        
        if not dataset:
            raise ValueError("Dataset is empty")
        
        dataloader = self._create_dataloader(dataset, self.config.training.batch_size)
        
        device = next(self.model.parameters()).device
        
        all_predictions = []
        all_reaction_ids = []
        all_reaction_data = []
        
        self.logger.info(f"Running predictions on {len(dataset)} samples...")
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                try:
                    batch = batch.to(device)
                    
                    pos0, pos1, pos2 = batch.pos0, batch.pos1, batch.pos2
                    z0, z1, z2, batch_mapping = batch.z0, batch.z1, batch.z2, batch.batch
                    xtb_features = getattr(batch, 'xtb_features', None)
                    
                    if xtb_features is None:
                        self.logger.warning(f"Batch {batch_idx}: No XTB features found")
                    
                    _, _, predictions = self.model(pos0, pos1, pos2, z0, z1, z2, batch_mapping, xtb_features)
                    
                    all_predictions.append(predictions.cpu().numpy())
                    
                    batch_size = predictions.size(0)
                    for i in range(batch_size):
                        reaction_id = getattr(batch, 'reaction_id', None)
                        if reaction_id is not None and hasattr(reaction_id, '__getitem__') and i < len(reaction_id):
                            rid = reaction_id[i] if isinstance(reaction_id, (list, tuple)) else reaction_id
                        else:
                            rid = f"sample_{batch_idx}_{i}"
                        all_reaction_ids.append(rid)
                        
                        reaction_data = {}
                        for attr in ['id', 'smiles']:
                            if hasattr(batch, attr):
                                value = getattr(batch, attr)
                                if isinstance(value, (list, tuple)) and i < len(value):
                                    reaction_data[attr] = value[i]
                                elif not isinstance(value, (list, tuple)):
                                    reaction_data[attr] = value
                                else:
                                    reaction_data[attr] = None
                        
                        all_reaction_data.append(reaction_data)
                        
                except Exception as e:
                    self.logger.error(f"Error processing batch {batch_idx}: {e}")
                    continue
        
        if not all_predictions:
            raise RuntimeError("No predictions were generated")
        
        predictions = np.vstack(all_predictions)
        self.logger.info(f"Predictions shape: {predictions.shape}")
        
        model_target_fields = None
        if hasattr(self.model, 'target_field_names') and self.model.target_field_names:
            model_target_fields = self.model.target_field_names
        elif hasattr(self.config.dataset, 'target_fields') and self.config.dataset.target_fields:
            model_target_fields = self.config.dataset.target_fields
        
        feature_to_scaler_map = self._map_features_to_scaler_indices(
            self.config.dataset.input_features, model_target_fields
        )
        
        output_field_names = self._get_output_field_names(self.config.dataset.input_features)
        self.logger.info(f"Output field names: {output_field_names}")
        
        results = {}
        for i, output_field in enumerate(output_field_names):
            target_preds = predictions[:, i].reshape(-1, 1)
            
            scaler_idx = feature_to_scaler_map.get(i, i)
            if (hasattr(self.model, 'scaler') and self.model.scaler is not None 
                and isinstance(self.model.scaler, list) and scaler_idx < len(self.model.scaler) 
                and self.model.scaler[scaler_idx] is not None):
                try:
                    target_preds = self.model.scaler[scaler_idx].inverse_transform(target_preds)
                    self.logger.info(f"Applied inverse scaling (scaler {scaler_idx}) for output field '{output_field}'")
                except Exception as e:
                    self.logger.warning(f"Failed to apply inverse scaling for output field '{output_field}': {e}")
            else:
                self.logger.warning(f"No scaler available for output field '{output_field}' (index {i})")
            
            results[output_field] = target_preds.flatten()
        
        results_df = pd.DataFrame()
        
        if all_reaction_ids:
            results_df['ID'] = all_reaction_ids
        
        for i in range(min(len(all_reaction_data), len(results_df) if len(results_df) > 0 else len(all_reaction_data))):
            if i >= len(results_df):
                results_df = pd.concat([results_df, pd.DataFrame([{}])], ignore_index=True)
            data = all_reaction_data[i]
            for key, value in data.items():
                if key not in results_df.columns:
                    results_df[key] = None
                if value is not None:
                    results_df.at[i, key] = value
        
        for output_field, preds in results.items():
            results_df[f'{output_field}_predicted'] = preds
        
        output_path = os.path.join(output_dir, 'predictions.csv')
        try:
            results_df.to_csv(output_path, index=False)
            np.save(os.path.join(output_dir, 'predictions.npy'), predictions)
        except Exception as e:
            self.logger.error(f"Failed to save predictions: {e}")
            raise
        
        self.logger.info(f"Predictions saved to {output_path}")
        self.logger.info(f"Result statistics:")
        for output_field in output_field_names:
            pred_col = f'{output_field}_predicted'
            if pred_col in results_df.columns and len(results_df[pred_col]) > 0:
                mean_val = results_df[pred_col].mean()
                std_val = results_df[pred_col].std()
                self.logger.info(f"  {output_field}: mean={mean_val:.4f}, std={std_val:.4f}")
        
        return results_df