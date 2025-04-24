import os
import os.path as osp
import csv
import torch
import hashlib
from torch_geometric.data import InMemoryDataset, Data
from rdkit import Chem
from sklearn.utils import shuffle
import pandas as pd
from tqdm import tqdm
import json
import re
import glob
from typing import List, Dict, Tuple, Optional, Union, Any, Callable

def read_xyz(file_path):
    atomic_symbols = []
    coords = []

    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()

        if len(lines) < 3:
            raise ValueError(f"File {file_path} format error: not enough lines")

        natoms = int(lines[0].strip())

        for line in lines[2:]:
            parts = line.split()
            if len(parts) < 4:
                continue
            atomic_symbols.append(parts[0])
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])

        coords = torch.tensor(coords, dtype=torch.float)
    except Exception as e:
        print(f"Error reading xyz file {file_path}: {e}")
        return None, None

    return atomic_symbols, coords

def symbols_to_atomic_numbers(symbols):
    pt = Chem.GetPeriodicTable()
    atomic_nums = []

    for s in symbols:
        try:
            atomic_nums.append(pt.GetAtomicNumber(s))
        except Exception as e:
            print(f"Error converting symbol {s} to atomic number: {e}")
            return None

    return torch.tensor(atomic_nums, dtype=torch.long)

class ReactionXYZDataset(InMemoryDataset):
    SCHEMA_VERSION = "v3"

    def __init__(self, 
                 root, 
                 csv_file='dataset.csv', 
                 transform=None, 
                 pre_transform=None, 
                 pre_filter=None,
                 target_fields=None, 
                 file_patterns=None, 
                 file_dir_pattern=None,
                 input_features=None, 
                 force_reload=False, 
                 inference_mode=False,
                 id_field='ID',
                 dir_field='R_dir',
                 reaction_field='reaction'):
        
        self.csv_file = csv_file
        self.target_fields = target_fields if isinstance(target_fields, list) else (
            [target_fields] if target_fields else None)
        
        self.file_patterns = file_patterns or ['*_reactant.xyz', '*_ts.xyz', '*_product.xyz']
        self.file_dir_pattern = file_dir_pattern or 'reaction_*'
        self.id_field = id_field
        self.dir_field = dir_field
        self.reaction_field = reaction_field
        self.input_features = input_features
        self.force_reload = force_reload
        self.inference_mode = inference_mode
    
        if self.input_features is not None and not isinstance(self.input_features, list):
            self.input_features = [self.input_features]
    
        super(ReactionXYZDataset, self).__init__(root, transform, pre_transform, pre_filter)
    
        if self.check_if_reprocessing_needed():
            self.process()
    
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
    
    def check_if_reprocessing_needed(self):
        if self.force_reload:
            print("Force reload enabled, reprocessing data")
            return True

        processed_path = self.processed_paths[0]

        if not osp.exists(processed_path):
            print(f"Processed file not found at {processed_path}, processing dataset")
            return True

        metadata_path = osp.join(self.processed_dir, 'metadata.json')
        if not osp.exists(metadata_path):
            print("Metadata file not found, reprocessing dataset")
            return True

        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            if metadata.get('schema_version') != self.SCHEMA_VERSION:
                print(f"Schema version changed from {metadata.get('schema_version')} to {self.SCHEMA_VERSION}")
                return True

            if metadata.get('input_features') != self.input_features:
                print(f"Input features changed from {metadata.get('input_features')} to {self.input_features}")
                return True

            if metadata.get('target_fields') != self.target_fields:
                print(f"Target fields changed from {metadata.get('target_fields')} to {self.target_fields}")
                return True

            if metadata.get('file_patterns') != self.file_patterns:
                print(f"File patterns changed from {metadata.get('file_patterns')} to {self.file_patterns}")
                return True
                
            if metadata.get('inference_mode', False) != self.inference_mode:
                print(f"Inference mode changed from {metadata.get('inference_mode', False)} to {self.inference_mode}")
                return True

            try:
                saved_data, slices = torch.load(processed_path, weights_only=False)

                for attr in ['z0', 'z1', 'z2', 'pos0', 'pos1', 'pos2', 'y']:
                    if not hasattr(saved_data, attr):
                        print(f"Missing expected attribute {attr} in saved data")
                        return True

                for i in range(min(3, len(self.slices['z0']) - 1)):
                    _ = self[i]

                return False

            except Exception as e:
                print(f"Error checking saved data: {e}")
                return True

        except Exception as e:
            print(f"Error reading metadata: {e}")
            return True

    @property
    def raw_dir(self):
        return self.root

    @property
    def processed_dir(self):
        processed_dir = osp.join(self.root, 'processed')
        os.makedirs(processed_dir, exist_ok=True)
        return processed_dir

    @property
    def raw_file_names(self):
        return [osp.basename(self.csv_file)]

    @property
    def processed_file_names(self):
        features_str = '_'.join(sorted(self.input_features)) if self.input_features else 'no_features'
        targets_str = '_'.join(sorted(self.target_fields)) if self.target_fields else 'default'
        patterns_str = '_'.join([p.replace('*', 'X') for p in self.file_patterns])
        combined_str = f"{features_str}_{targets_str}_{patterns_str}_{self.SCHEMA_VERSION}"
        if self.inference_mode:
            combined_str += "_inference"
        features_hash = hashlib.md5(combined_str.encode()).hexdigest()[:12]
        return [f'data_{features_hash}.pt']

    def download(self):
        pass

    def save_metadata(self):
        metadata = {
            'schema_version': self.SCHEMA_VERSION,
            'input_features': self.input_features,
            'target_fields': self.target_fields,
            'file_patterns': self.file_patterns,
            'file_dir_pattern': self.file_dir_pattern,
            'inference_mode': self.inference_mode,
            'id_field': self.id_field,
            'dir_field': self.dir_field, 
            'reaction_field': self.reaction_field,
            'created_at': pd.Timestamp.now().isoformat()
        }

        metadata_path = osp.join(self.processed_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved metadata to {metadata_path}")
    
    def get_structure_files(self, folder_path, prefix):
        files = []
        
        for pattern in self.file_patterns:
            if '*' in pattern:
                search_pattern = pattern.replace('*', prefix)
                matching_files = glob.glob(os.path.join(folder_path, search_pattern))
                if matching_files:
                    files.append(matching_files[0])
                else:
                    return None
            else:
                file_path = os.path.join(folder_path, f"{prefix}{pattern}")
                if os.path.exists(file_path):
                    files.append(file_path)
                else:
                    return None
        
        if len(files) != 3:
            return None
            
        return files

    def extract_prefix(self, dir_name):
        if self.file_dir_pattern and '*' in self.file_dir_pattern:
            pattern = self.file_dir_pattern.replace('*', '(.*)')
            match = re.match(pattern, dir_name)
            if match:
                return match.group(1)
            
        return dir_name

    def process(self):
            for file_path in self.processed_paths:
                if osp.exists(file_path):
                    try:
                        os.remove(file_path)
                    except Exception as e:
                        print(f"Warning: Failed to remove old processed file: {e}")
    
            if not osp.exists(self.csv_file):
                raise FileNotFoundError(f"CSV file {self.csv_file} does not exist")
    
            with open(self.csv_file, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
    
            if not rows:
                raise ValueError("Empty CSV file")
    
            sample_row = rows[0]
    
            target_field_names = self.target_fields
            if not target_field_names and not self.inference_mode:
                possible_target_fields = ['G(TS)', 'DrG', 'dG(ts)', 'dG(TS)', 'energy']
                detected_fields = []
                for field in possible_target_fields:
                    if field in sample_row:
                        detected_fields.append(field)
                
                if detected_fields:
                    target_field_names = detected_fields
                else:
                    raise ValueError(f"Could not auto-detect target fields. Available fields: {list(sample_row.keys())}")
            
            if self.inference_mode and not target_field_names:
                target_field_names = ['target']
                print("Inference mode: Using dummy target field")
    
            print(f"Using target fields: {target_field_names}")
            print(f"Using input features: {self.input_features}")
            print(f"Using file patterns: {self.file_patterns}")
    
            data_list = []
            skipped_count = 0
            for row in tqdm(rows, desc="Processing reactions"):
                reaction_id = row.get(self.id_field, '').strip()
                dir_name = row.get(self.dir_field, '').strip()
                reaction_str = row.get(self.reaction_field, '').strip()
    
                if not reaction_id or not dir_name:
                    skipped_count += 1
                    continue
    
                folder_path = osp.join(self.raw_dir, dir_name)
                if not osp.isdir(folder_path):
                    print(f"Warning: Folder {folder_path} does not exist, skipping reaction_id {reaction_id}")
                    skipped_count += 1
                    continue
    
                target_values = []
                skip_record = False
    
                if not self.inference_mode:
                    for target_field in target_field_names:
                        target_value_str = row.get(target_field, '').strip()
                        if not target_value_str:
                            skip_record = True
                            break
    
                        try:
                            target_values.append(float(target_value_str))
                        except ValueError:
                            skip_record = True
                            break
    
                    if skip_record:
                        skipped_count += 1
                        continue
                else:
                    target_values = [0.0] * len(target_field_names) if target_field_names else [0.0]
    
                feature_values = []
                skip_record = False
                if self.input_features and len(self.input_features) > 0:
                    for feature_name in self.input_features:
                        feature_str = row.get(feature_name, '').strip()
                        if not feature_str:
                            skip_record = True
                            break
            
                        try:
                            feature_values.append(float(feature_str))
                        except ValueError:
                            skip_record = True
                            break
    
                    if skip_record:
                        skipped_count += 1
                        continue
    
                prefix = self.extract_prefix(dir_name)
                
                structure_files = self.get_structure_files(folder_path, prefix)
                if not structure_files:
                    print(f"Warning: Could not find structure files for {reaction_id} in {folder_path}")
                    skipped_count += 1
                    continue
                    
                reactant_file, ts_file, product_file = structure_files
    
                symbols0, pos0 = read_xyz(reactant_file)
                symbols1, pos1 = read_xyz(ts_file)
                symbols2, pos2 = read_xyz(product_file)
    
                if None in (symbols0, pos0, symbols1, pos1, symbols2, pos2):
                    print(f"Warning: Failed to read XYZ files for {reaction_id}, skipping")
                    skipped_count += 1
                    continue
    
                z0 = symbols_to_atomic_numbers(symbols0)
                z1 = symbols_to_atomic_numbers(symbols1)
                z2 = symbols_to_atomic_numbers(symbols2)
    
                if None in (z0, z1, z2):
                    print(f"Warning: Failed to convert atomic symbols for {reaction_id}, skipping")
                    skipped_count += 1
                    continue
    
                if len({pos0.size(0), pos1.size(0), pos2.size(0), z0.size(0), z1.size(0), z2.size(0)}) > 1:
                    print(f"Warning: Inconsistent atom count in {reaction_id}, skipping")
                    skipped_count += 1
                    continue
    
                y = torch.tensor([target_values], dtype=torch.float)
            
                data = Data(
                    z0=z0, z1=z1, z2=z2,
                    pos0=pos0, pos1=pos1, pos2=pos2,
                    y=y,
                    num_nodes=z0.size(0)
                )
            
                if self.input_features and feature_values and len(feature_values) > 0:
                    data.xtb_features = torch.tensor([feature_values], dtype=torch.float)
                    data.feature_names = self.input_features
            
                data.reaction_id = reaction_id
                data.id = dir_name
                data.reaction = reaction_str
            
                data_list.append(data)
    
            if not data_list:
                raise RuntimeError("No reaction data processed, please check the CSV and XYZ file formats.")
    
            print(f"Processed {len(data_list)} reactions, skipped {skipped_count} reactions")
    
            data, slices = self.collate(data_list)
            data.input_features = self.input_features
            data.target_fields = target_field_names
            data.inference_mode = self.inference_mode
    
            torch.save((data, slices), self.processed_paths[0])
            self.save_metadata()
            print(f"Processed {len(data_list)} reactions, saved to {self.processed_paths[0]}")

    def get_idx_split(self, train_size, valid_size, seed):
        group_to_indices = {}
        for idx, data in enumerate(self):
            if not hasattr(data, 'reaction_id'):
                raise ValueError(f"No reaction_id attribute found in dataset")
    
            key = data.reaction_id
            if key not in group_to_indices:
                group_to_indices[key] = []
            group_to_indices[key].append(idx)
    
        group_keys = list(group_to_indices.keys())
        group_keys = shuffle(group_keys, random_state=seed)
    
        train_idx = []
        valid_idx = []
        test_idx = []
        n_assigned = 0
    
        for key in group_keys:
            indices = group_to_indices[key]
    
            if n_assigned < train_size:
                train_idx.extend(indices)
            elif n_assigned < train_size + valid_size:
                valid_idx.extend(indices)
            else:
                test_idx.extend(indices)
    
            n_assigned += len(indices)
    
        train_idx = torch.tensor(train_idx, dtype=torch.long)
        valid_idx = torch.tensor(valid_idx, dtype=torch.long)
        test_idx = torch.tensor(test_idx, dtype=torch.long)
    
        print(f"Dataset split: train {len(train_idx)}, validation {len(valid_idx)}, test {len(test_idx)} samples")
        
        train_set = set(train_idx.tolist())
        valid_set = set(valid_idx.tolist())
        test_set = set(test_idx.tolist())
    
        assert len(train_set.intersection(valid_set)) == 0, "Train and validation sets overlap!"
        assert len(train_set.intersection(test_set)) == 0, "Train and test sets overlap!"
        assert len(valid_set.intersection(test_set)) == 0, "Validation and test sets overlap!"
    
        return {'train': train_idx, 'valid': valid_idx, 'test': test_idx}

    def __getitem__(self, idx):
        try:
            data = super().__getitem__(idx)
            if len(data.y.shape) == 1:
                data.y = data.y.unsqueeze(0)
            return data
        except Exception as e:
            print(f"Error accessing item at index {idx}: {e}")
            raise