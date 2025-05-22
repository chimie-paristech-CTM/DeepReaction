import os
import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union
import pytorch_lightning as pl


class ReactionPredictor:
    def __init__(
            self,
            checkpoint_path: str,
            output_dir: str = './predictions',
            batch_size: int = 32,
            gpu: bool = True,
            num_workers: int = 4,
            use_scaler: bool = False
    ):
        self.checkpoint_path = checkpoint_path
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.gpu = gpu
        self.num_workers = num_workers
        self.use_scaler = use_scaler
        self.model = None
        self.device = torch.device('cuda' if gpu and torch.cuda.is_available() else 'cpu')
        self.target_field_names = None

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Load the model
        self._load_model()

    def _load_model(self):
        try:
            # Try local import first
            from ..module.pl_wrap import Estimator
        except ImportError:
            # Fallback to global import
            from deepreaction.module.pl_wrap import Estimator
            
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file does not exist: {self.checkpoint_path}")
            
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # Create model from checkpoint's hyperparameters
        self.model = Estimator(**checkpoint['hyper_parameters'])
        self.model.load_state_dict(checkpoint['state_dict'])
        
        # Set to evaluation mode
        self.model.eval()
        
        # Disable scaling if requested
        if not self.use_scaler and hasattr(self.model, 'scaler') and self.model.scaler is not None:
            print("Disabling scaler for predictions")
            self.model.scaler = None

        # Move model to appropriate device
        self.model = self.model.to(self.device)
            
        # Get target field names
        if hasattr(self.model, 'target_field_names') and self.model.target_field_names:
            self.target_field_names = self.model.target_field_names
            print(f"Using target field names from model: {self.target_field_names}")
        else:
            num_targets = self.model.num_targets if hasattr(self.model, 'num_targets') else 1
            self.target_field_names = [f"target_{i}" for i in range(num_targets)]
            print(f"Using default target field names: {self.target_field_names}")

    def predict(self, data_loader=None, dataset=None, csv_output_path=None):
        if self.model is None:
            raise ValueError("Model has not been loaded. Check the checkpoint path.")

        if data_loader is None and dataset is not None:
            # Get data loaders from dataset
            _, _, data_loader = dataset.get_data_loaders(
                batch_size=self.batch_size,
                num_workers=self.num_workers
            )

        if data_loader is None:
            raise ValueError("Either data_loader or dataset must be provided.")

        # Make predictions
        all_predictions = []
        all_reaction_ids = []
        all_reaction_data = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                print(f"Processing batch {batch_idx+1}/{len(data_loader)}")
                batch = batch.to(self.device)
                pos0, pos1, pos2 = batch.pos0, batch.pos1, batch.pos2
                z0, z1, z2, batch_mapping = batch.z0, batch.z1, batch.z2, batch.batch
                xtb_features = getattr(batch, 'xtb_features', None)

                # Direct call to the internal model's forward method to bypass Lightning wrapper
                _, embeddings, predictions = self.model.model(pos0, pos1, pos2, z0, z1, z2, batch_mapping, xtb_features)
                all_predictions.append(predictions.cpu().numpy())

                for i in range(len(predictions)):
                    reaction_id = batch.reaction_id[i] if hasattr(batch, 'reaction_id') else f"sample_{i}"
                    all_reaction_ids.append(reaction_id)

                    reaction_data = {}
                    for attr in ['id', 'reaction']:
                        if hasattr(batch, attr):
                            value = getattr(batch, attr)
                            if isinstance(value, list):
                                reaction_data[attr] = value[i] if i < len(value) else None
                            else:
                                reaction_data[attr] = value
                    all_reaction_data.append(reaction_data)

        predictions = np.vstack(all_predictions) if all_predictions else np.array([])
        print(f"Made predictions for {len(all_reaction_ids)} samples with shape {predictions.shape}")

        # Get target field names from model if available
        target_fields = self.target_field_names or [f"target_{i}" for i in range(predictions.shape[1])]

        # Apply inverse scaling if available
        results = {}
        for i, target_name in enumerate(target_fields):
            if i >= predictions.shape[1]:
                continue
                
            target_preds = predictions[:, i].reshape(-1, 1)
            if hasattr(self.model, 'scaler') and self.model.scaler is not None:
                if isinstance(self.model.scaler, list) and i < len(self.model.scaler):
                    target_preds = self.model.scaler[i].inverse_transform(target_preds)
                elif not isinstance(self.model.scaler, list):
                    target_preds = self.model.scaler.inverse_transform(target_preds)
            results[target_name] = target_preds.flatten()

        # Create output DataFrame
        results_df = pd.DataFrame()
        results_df['ID'] = all_reaction_ids

        for i, data in enumerate(all_reaction_data):
            for key, value in data.items():
                if key not in results_df.columns:
                    results_df[key] = None
                if value is not None:
                    results_df.at[i, key] = value

        for target_name, preds in results.items():
            results_df[f'{target_name}_predicted'] = preds

        # Save predictions
        if csv_output_path is None:
            csv_output_path = os.path.join(self.output_dir, 'predictions.csv')

        results_df.to_csv(csv_output_path, index=False)
        np.save(os.path.join(self.output_dir, 'predictions.npy'), predictions)
        print(f"Saved predictions to {csv_output_path}")

        return results_df

    def predict_from_dataset(self, dataset, csv_output_path=None):
        # If dataset doesn't have target fields but the model does, set them
        if hasattr(dataset, 'target_fields') and not dataset.target_fields and self.target_field_names:
            dataset.target_fields = self.target_field_names
            print(f"Set dataset target fields to: {self.target_field_names}")

        _, _, test_loader = dataset.get_data_loaders(
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )
        
        print(f"Created data loader with {len(test_loader)} batches")
        return self.predict(data_loader=test_loader, csv_output_path=csv_output_path)