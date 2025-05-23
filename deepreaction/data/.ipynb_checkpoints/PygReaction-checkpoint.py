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
        except Exception:
            print(f"Error converting symbol {s} to atomic number")
            return None

    return torch.tensor(atomic_nums, dtype=torch.long)


class ReactionXYZDataset(InMemoryDataset):
    SCHEMA_VERSION = "v2"

    def __init__(self, root, csv_file='DA_dataset.csv', transform=None, pre_transform=None, pre_filter=None,
                 target_fields=None, file_suffixes=None, input_features=None, force_reload=False, inference_mode=False):
        if osp.isabs(csv_file) or csv_file.startswith('./') or csv_file.startswith('../'):
            self.csv_file = csv_file
        else:
            self.csv_file = osp.join(root, csv_file)
        self.target_fields = target_fields if isinstance(target_fields, list) else (
            [target_fields] if target_fields else None)
        self.file_suffixes = file_suffixes or ['_reactant.xyz', '_ts.xyz', '_product.xyz']
        self.input_features = input_features or ['DG_act_xtb', 'DrG_xtb']
        self.force_reload = force_reload
        self.inference_mode = inference_mode

        if not isinstance(self.input_features, list):
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

            if metadata.get('file_suffixes') != self.file_suffixes:
                print(f"File suffixes changed from {metadata.get('file_suffixes')} to {self.file_suffixes}")
                return True
                
            if metadata.get('inference_mode', False) != self.inference_mode:
                print(f"Inference mode changed from {metadata.get('inference_mode', False)} to {self.inference_mode}")
                return True

            try:
                saved_data, saved_slices = torch.load(processed_path, weights_only=False)

                for attr in ['z0', 'z1', 'z2', 'pos0', 'pos1', 'pos2', 'y']:
                    if not hasattr(saved_data, attr):
                        print(f"Missing expected attribute {attr} in saved data")
                        return True

                if saved_slices and 'z0' in saved_slices and len(saved_slices['z0']) > 1:
                    for i in range(min(3, len(saved_slices['z0']) - 1)):
                        try:
                            test_data = super(ReactionXYZDataset, self).__getitem__(i)
                            if test_data is None:
                                print(f"Null data at index {i}")
                                return True
                        except Exception as e:
                            print(f"Error accessing sample {i}: {e}")
                            return True

                return False

            except Exception as e:
                print(f"Error checking saved data: {e}")
                return True

        except Exception as e:
            print(f"Error reading metadata: {e}")
            return True

    def _cleanup_corrupted_files(self):
        for file_path in self.processed_paths:
            if osp.exists(file_path):
                try:
                    torch.load(file_path, weights_only=False)
                except Exception:
                    print(f"Removing corrupted file: {file_path}")
                    try:
                        os.remove(file_path)
                    except Exception as remove_error:
                        print(f"Warning: Failed to remove corrupted file: {remove_error}")
        
        metadata_path = osp.join(self.processed_dir, 'metadata.json')
        if osp.exists(metadata_path):
            try:
                os.remove(metadata_path)
                print(f"Removed metadata file: {metadata_path}")
            except Exception as e:
                print(f"Warning: Failed to remove metadata file: {e}")

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
        features_str = '_'.join(sorted(self.input_features))
        targets_str = '_'.join(sorted(self.target_fields)) if self.target_fields else 'default'
        combined_str = f"{features_str}_{targets_str}_{self.SCHEMA_VERSION}"
        if self.inference_mode:
            combined_str += "_inference"
        features_hash = hashlib.md5(combined_str.encode()).hexdigest()[:8]
        return [f'data_{features_hash}.pt']

    def download(self):
        pass

    def save_metadata(self):
        metadata = {
            'schema_version': self.SCHEMA_VERSION,
            'input_features': self.input_features,
            'target_fields': self.target_fields,
            'file_suffixes': self.file_suffixes,
            'inference_mode': self.inference_mode,
            'created_at': pd.Timestamp.now().isoformat()
        }

        metadata_path = osp.join(self.processed_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved metadata to {metadata_path}")

    def process(self):
        for file_path in self.processed_paths:
            if osp.exists(file_path):
                print(f"Removing old processed file: {file_path}")
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
            possible_target_fields = ['DG_act', 'dG(ts)', 'G(TS)', 'G(ts)', 'dG(TS)', 'DrG']
            for field in possible_target_fields:
                if field in sample_row:
                    target_field_names = [field]
                    break

        if not target_field_names and not self.inference_mode:
            raise ValueError(f"Could not find target field in CSV. Available fields: {list(sample_row.keys())}")
        
        if self.inference_mode and not target_field_names:
            target_field_names = ['target']
            print("Inference mode: Using dummy target field")

        print(f"Using target fields: {target_field_names}")
        print(f"Using input features: {self.input_features}")
        reactant_suffix, ts_suffix, product_suffix = self.file_suffixes
        print(f"Using file suffixes: reactant='{reactant_suffix}', ts='{ts_suffix}', product='{product_suffix}'")

        data_list = []
        for row in tqdm(rows, desc="Processing reactions"):
            reaction_id = row.get('ID', '').strip()
            R_dir = row.get('R_dir', '').strip()
            reaction_str = row.get('smiles', '').strip()

            if not reaction_id or not R_dir:
                print(f"Warning: Missing required fields, skipping record: {row}")
                continue

            folder_path = osp.join(self.raw_dir, R_dir)
            if not osp.isdir(folder_path):
                print(f"Warning: Folder {folder_path} does not exist, skipping reaction_id {reaction_id}")
                continue

            target_values = []
            skip_record = False

            if not self.inference_mode:
                for target_field in target_field_names:
                    target_value_str = row.get(target_field, '').strip()
                    if not target_value_str:
                        print(f"Warning: Missing target field {target_field}, skipping record: {reaction_id}")
                        skip_record = True
                        break

                    try:
                        target_values.append(float(target_value_str))
                    except ValueError:
                        print(f"Error parsing target value in reaction_id {reaction_id}, field {target_field}")
                        skip_record = True
                        break

                if skip_record:
                    continue
            else:
                target_values = [0.0] * len(target_field_names)

            feature_values = []
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

            prefix = R_dir
            if prefix.startswith("reaction_"):
                prefix = prefix[len("reaction_"):]

            reactant_file = osp.join(folder_path, f"{prefix}{reactant_suffix}")
            ts_file = osp.join(folder_path, f"{prefix}{ts_suffix}")
            product_file = osp.join(folder_path, f"{prefix}{product_suffix}")

            if not (osp.exists(reactant_file) and osp.exists(ts_file) and osp.exists(product_file)):
                print(f"Warning: One or more xyz files are missing in {folder_path}, skipping reaction_id {reaction_id}")
                continue

            symbols0, pos0 = read_xyz(reactant_file)
            symbols1, pos1 = read_xyz(ts_file)
            symbols2, pos2 = read_xyz(product_file)

            if None in (symbols0, pos0, symbols1, pos1, symbols2, pos2):
                print(f"Warning: Failed to read XYZ files for {reaction_id}, skipping")
                continue

            z0 = symbols_to_atomic_numbers(symbols0)
            z1 = symbols_to_atomic_numbers(symbols1)
            z2 = symbols_to_atomic_numbers(symbols2)

            if None in (z0, z1, z2):
                print(f"Warning: Failed to convert atomic symbols for {reaction_id}, skipping")
                continue

            if len({pos0.size(0), pos1.size(0), pos2.size(0), z0.size(0), z1.size(0), z2.size(0)}) > 1:
                print(f"Warning: Inconsistent atom count in {reaction_id}, skipping")
                continue

            y = torch.tensor([target_values], dtype=torch.float)

            data = Data(
                z0=z0, z1=z1, z2=z2,
                pos0=pos0, pos1=pos1, pos2=pos2,
                y=y,
                xtb_features=torch.tensor([feature_values], dtype=torch.float),
                feature_names=self.input_features,
                reaction_id=reaction_id,
                id=R_dir,
                smiles=reaction_str,
                num_nodes=z0.size(0)
            )

            data_list.append(data)

        if not data_list:
            raise RuntimeError("No reaction data processed, please check the CSV and xyz file formats.")

        data, slices = self.collate(data_list)
        data.input_features = self.input_features
        data.target_fields = target_field_names
        data.inference_mode = self.inference_mode

        try:
            torch.save((data, slices), self.processed_paths[0])
            self.save_metadata()
            print(f"Processed {len(data_list)} reactions, saved to {self.processed_paths[0]}")
        except Exception as e:
            print(f"Error saving processed data: {e}")
            backup_dir = osp.join(self.processed_dir, 'backup')
            os.makedirs(backup_dir, exist_ok=True)
            backup_path = osp.join(backup_dir, osp.basename(self.processed_paths[0]))
            try:
                torch.save((data, slices), backup_path)
                print(f"Saved to backup location: {backup_path}")
                self.processed_paths[0] = backup_path
                self.save_metadata()
            except Exception as backup_error:
                print(f"Failed to save to backup location: {backup_error}")
                raise e

    def get_idx_split(self, train_size, valid_size, seed):
        group_to_indices = {}
        for idx in range(len(self)):
            try:
                data = self[idx]
                if not hasattr(data, 'reaction_id'):
                    raise ValueError(f"No reaction_id attribute found in dataset")

                key = data.reaction_id
                if key not in group_to_indices:
                    group_to_indices[key] = []
                group_to_indices[key].append(idx)
            except Exception as e:
                print(f"Warning: Failed to process index {idx}: {e}")
                continue

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

    def __len__(self):
        if self.slices is None or 'z0' not in self.slices:
            return 0
        return max(0, len(self.slices['z0']) - 1)

    def __getitem__(self, idx):
        try:
            if self.slices is None or 'z0' not in self.slices:
                raise ValueError("Dataset not properly loaded")
                
            max_idx = len(self.slices['z0']) - 1
            if idx < 0 or idx >= max_idx:
                raise IndexError(f"Index {idx} out of range (max: {max_idx - 1})")
            
            data = super().__getitem__(idx)
            
            if data is None:
                raise ValueError(f"Data at index {idx} is None")
            
            if hasattr(data, 'y') and data.y is not None:
                if len(data.y.shape) == 1:
                    data.y = data.y.unsqueeze(0)
            else:
                data.y = torch.zeros((1, 1), dtype=torch.float)
            
            required_attrs = ['z0', 'z1', 'z2', 'pos0', 'pos1', 'pos2', 'y']
            for attr in required_attrs:
                if not hasattr(data, attr) or getattr(data, attr) is None:
                    raise ValueError(f"Data at index {idx} missing attribute: {attr}")
            
            return data
            
        except Exception as e:
            print(f"Error accessing item at index {idx}: {e}")
            
            if idx == 0:
                print("First item access failed, data may be corrupted. Forcing reprocessing...")
                self._cleanup_corrupted_files()
                raise RuntimeError(f"Data corruption detected at index {idx}. Please restart to reprocess data.")
            
            raise