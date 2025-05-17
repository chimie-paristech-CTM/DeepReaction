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
        except Exception as e:
            print(f"Error converting symbol {s} to atomic number: {e}")
            return None

    return torch.tensor(atomic_nums, dtype=torch.long)


class ReactionXYZDataset(InMemoryDataset):
    SCHEMA_VERSION = "v2"  # Update when changing data structure

    def __init__(self, root, csv_file='DA_dataset.csv', transform=None, pre_transform=None, pre_filter=None,
                 target_fields=None, file_suffixes=None, input_features=None, force_reload=False):
        self.csv_file = osp.join(root, csv_file)
        self.target_fields = target_fields if isinstance(target_fields, list) else (
            [target_fields] if target_fields else None)
        self.file_suffixes = file_suffixes or ['_reactant.xyz', '_ts.xyz', '_product.xyz']
        self.input_features = input_features or ['G(TS)_xtb', 'DrG_xtb']
        self.force_reload = force_reload

        if not isinstance(self.input_features, list):
            self.input_features = [self.input_features]

        super(ReactionXYZDataset, self).__init__(root, transform, pre_transform, pre_filter)

        if self.check_if_reprocessing_needed():
            self.process()

        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    def check_if_reprocessing_needed(self):
        """Check if the dataset needs reprocessing due to changes in structure or parameters."""
        if self.force_reload:
            print("Force reload enabled, reprocessing data")
            return True

        processed_path = self.processed_paths[0]

        if not osp.exists(processed_path):
            print(f"Processed file not found at {processed_path}, processing dataset")
            return True

        # Check metadata file
        metadata_path = osp.join(self.processed_dir, 'metadata.json')
        if not osp.exists(metadata_path):
            print("Metadata file not found, reprocessing dataset")
            return True

        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            # Basic validation
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

            # Verify data integrity
            try:
                saved_data, slices = torch.load(processed_path, weights_only=False)

                # Check expected attributes
                for attr in ['z0', 'z1', 'z2', 'pos0', 'pos1', 'pos2', 'y']:
                    if not hasattr(saved_data, attr):
                        print(f"Missing expected attribute {attr} in saved data")
                        return True

                # Test sample access
                for i in range(min(3, len(self.slices['z0']) - 1)):
                    _ = self[i]

                return False  # No reprocessing needed

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
        features_str = '_'.join(sorted(self.input_features))
        targets_str = '_'.join(sorted(self.target_fields)) if self.target_fields else 'default'
        combined_str = f"{features_str}_{targets_str}_{self.SCHEMA_VERSION}"
        features_hash = hashlib.md5(combined_str.encode()).hexdigest()[:8]
        return [f'data_{features_hash}.pt']

    def download(self):
        pass

    def save_metadata(self):
        """Save processing metadata to detect changes later."""
        metadata = {
            'schema_version': self.SCHEMA_VERSION,
            'input_features': self.input_features,
            'target_fields': self.target_fields,
            'file_suffixes': self.file_suffixes,
            'created_at': pd.Timestamp.now().isoformat()
        }

        metadata_path = osp.join(self.processed_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved metadata to {metadata_path}")

    def process(self):
        """Process raw data into processed data."""
        # Clean up old processed file if it exists
        for file_path in self.processed_paths:
            if osp.exists(file_path):
                print(f"Removing old processed file: {file_path}")
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"Warning: Failed to remove old processed file: {e}")

        if not osp.exists(self.csv_file):
            raise FileNotFoundError(f"CSV file {self.csv_file} does not exist")

        # Read CSV file
        with open(self.csv_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            raise ValueError("Empty CSV file")

        sample_row = rows[0]

        # Determine target fields if not specified
        target_field_names = self.target_fields
        if not target_field_names:
            possible_target_fields = ['dG(ts)', 'G(TS)', 'G(ts)', 'dG(TS)']
            for field in possible_target_fields:
                if field in sample_row:
                    target_field_names = [field]
                    break

        if not target_field_names:
            raise ValueError(f"Could not find target field in CSV. Available fields: {list(sample_row.keys())}")

        print(f"Using target fields: {target_field_names}")
        print(f"Using input features: {self.input_features}")
        reactant_suffix, ts_suffix, product_suffix = self.file_suffixes
        print(f"Using file suffixes: reactant='{reactant_suffix}', ts='{ts_suffix}', product='{product_suffix}'")

        data_list = []
        for row in tqdm(rows, desc="Processing reactions"):
            reaction_id = row.get('ID', '').strip()
            R_dir = row.get('R_dir', '').strip()
            reaction_str = row.get('reaction', '').strip()

            # Validation checks
            if not reaction_id or not R_dir:
                print(f"Warning: Missing required fields, skipping record: {row}")
                continue

            folder_path = osp.join(self.raw_dir, R_dir)
            if not osp.isdir(folder_path):
                print(f"Warning: Folder {folder_path} does not exist, skipping reaction_id {reaction_id}")
                continue

            # Get target values for all target fields
            target_values = []
            skip_record = False

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

            # Get input feature values
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

            # Prepare file paths
            prefix = R_dir
            if prefix.startswith("reaction_"):
                prefix = prefix[len("reaction_"):]

            reactant_file = osp.join(folder_path, f"{prefix}{reactant_suffix}")
            ts_file = osp.join(folder_path, f"{prefix}{ts_suffix}")
            product_file = osp.join(folder_path, f"{prefix}{product_suffix}")

            if not (osp.exists(reactant_file) and osp.exists(ts_file) and osp.exists(product_file)):
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

            # Create data object
            y = torch.tensor([target_values], dtype=torch.float)

            data = Data(
                z0=z0, z1=z1, z2=z2,
                pos0=pos0, pos1=pos1, pos2=pos2,
                y=y,
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

        data, slices = self.collate(data_list)
        data.input_features = self.input_features
        data.target_fields = target_field_names

        torch.save((data, slices), self.processed_paths[0])
        self.save_metadata()
        print(f"Processed {len(data_list)} reactions, saved to {self.processed_paths[0]}")

    def get_idx_split(self, train_size, valid_size, seed):
        # Group indices by reaction_id to maintain reaction integrity in splits
        group_to_indices = {}
        for idx, data in enumerate(self):
            if not hasattr(data, 'reaction_id'):
                raise ValueError(f"No reaction_id attribute found in dataset")

            key = data.reaction_id
            if key not in group_to_indices:
                group_to_indices[key] = []
            group_to_indices[key].append(idx)

        # Shuffle reaction groups
        group_keys = list(group_to_indices.keys())
        group_keys = shuffle(group_keys, random_state=seed)

        train_idx = []
        valid_idx = []
        test_idx = []
        n_assigned = 0

        # Assign reactions to splits
        for key in group_keys:
            indices = group_to_indices[key]

            if n_assigned < train_size:
                train_idx.extend(indices)
            elif n_assigned < train_size + valid_size:
                valid_idx.extend(indices)
            else:
                test_idx.extend(indices)

            n_assigned += len(indices)

        # Convert to tensors
        train_idx = torch.tensor(train_idx, dtype=torch.long)
        valid_idx = torch.tensor(valid_idx, dtype=torch.long)
        test_idx = torch.tensor(test_idx, dtype=torch.long)

        print(f"Dataset split: train {len(train_idx)}, validation {len(valid_idx)}, test {len(test_idx)} samples")

        # Validate no overlap between splits
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