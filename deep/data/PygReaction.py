# dataset/PygReaction.py
import os
import os.path as osp
import csv
import re
from tqdm import tqdm
import torch
from torch_geometric.data import InMemoryDataset, Data
from rdkit import Chem
from rdkit.Chem import PeriodicTable
from sklearn.utils import shuffle
import pandas as pd
import torch


def read_xyz(file_path):
    """
    Read an xyz file and return a list of atomic symbols and a coordinate tensor.

    The xyz file format is:
      - First line: number of atoms
      - Second line: comment
      - Subsequent lines: element_symbol  x  y  z
    """
    atomic_symbols = []
    coords = []
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        if len(lines) < 3:
            raise ValueError(f"File {file_path} format error: not enough lines")
        natoms = int(lines[0].strip())
        if natoms != len(lines) - 2:
            print(f"Warning: Declared atom count does not match actual line count in {file_path}")
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
    """
    Convert atomic symbols to atomic numbers using RDKit's PeriodicTable.
    """
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
    r"""
    The dataset root directory contains a CSV file (e.g., DATASET_DA.csv), with each row recording a reaction.

    The CSV file fields include:
      - ID: Reaction identifier, stored in data.reaction_id
      - Rnber: Sequence number (optional)
      - R_dir: Name of the xyz folder (e.g., reaction_R0), stored in data.id
      - reaction: Reaction SMILES or other description
      - dG(ts): Target value (prediction y) - Note: May be stored as "G(TS)" or "dG(ts)" in the CSV

    For each CSV record, the program will search for three xyz files in the corresponding folder under the dataset root:
      - {prefix}{reactant_suffix} (input 0), where the "reaction_" prefix is removed from the folder name
      - {prefix}{ts_suffix} (input 1)
      - {prefix}{product_suffix} (input 2)

    Atomic symbols are extracted from {prefix}{reactant_suffix} (converted to atomic numbers z),
    and atomic coordinates are extracted from all three files (pos0, pos1, pos2).
    Additionally, the CSV field ID is saved to data.reaction_id and R_dir is saved to data.id.
    """

    def __init__(self, root, csv_file='DA_dataset.csv', transform=None, pre_transform=None, pre_filter=None,
                 energy_field=None, file_suffixes=None):
        self.csv_file = osp.join(root, csv_file)
        self.energy_field = energy_field
        self.file_suffixes = file_suffixes or ['_reactant.xyz', '_ts.xyz', '_product.xyz']
        super(ReactionXYZDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return self.root

    @property
    def processed_dir(self):
        return osp.join(self.root, 'processed')

    @property
    def raw_file_names(self):
        # The raw files consist of the CSV file and corresponding folders for each reaction.
        # Here we return the CSV file name.
        return [osp.basename(self.csv_file)]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # If necessary, implement download logic here.
        pass

    def process(self):
        data_list = []
        if not osp.exists(self.csv_file):
            raise FileNotFoundError(f"CSV file {self.csv_file} does not exist")

        with open(self.csv_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Check for field name variations
        sample_row = rows[0] if rows else {}

        # Use the provided energy field name or detect it
        energy_field_name = self.energy_field
        if not energy_field_name:
            possible_energy_fields = ['dG(ts)', 'G(TS)', 'G(ts)', 'dG(TS)']
            for field in possible_energy_fields:
                if field in sample_row:
                    energy_field_name = field
                    break

        if not energy_field_name:
            print(f"Warning: Could not find energy field in CSV. Available fields: {list(sample_row.keys())}")
            if 'DrG' in sample_row:
                print("Found 'DrG' field, but this is typically the reaction energy, not the transition state energy.")
            return

        print(f"Using '{energy_field_name}' as the energy field")
        # Unpack the file suffixes
        reactant_suffix, ts_suffix, product_suffix = self.file_suffixes
        print(f"Using file suffixes: reactant='{reactant_suffix}', ts='{ts_suffix}', product='{product_suffix}'")

        for row in tqdm(rows, desc="Processing reactions"):
            # Read fields from the CSV
            reaction_id = row.get('ID', '').strip()  # Reaction identifier
            R_dir = row.get('R_dir', '').strip()  # Folder name
            reaction_str = row.get('reaction', '').strip()
            dG_ts_str = row.get(energy_field_name, '').strip()

            if not reaction_id or not R_dir or not dG_ts_str:
                print(f"Warning: Missing required fields, skipping record: {row}")
                continue
            try:
                dG_ts = float(dG_ts_str)
            except Exception as e:
                print(f"Error parsing {energy_field_name} value {dG_ts_str} in reaction_id {reaction_id}: {e}")
                continue

            # Construct the path for the xyz folder
            folder_path = osp.join(self.raw_dir, R_dir)
            if not osp.isdir(folder_path):
                print(f"Warning: Folder {folder_path} does not exist, skipping reaction_id {reaction_id}")
                continue

            # Remove the "reaction_" prefix from the folder name (if present) to build file names
            prefix = R_dir
            if prefix.startswith("reaction_"):
                prefix = prefix[len("reaction_"):]

            # Construct xyz file paths using the provided suffixes
            reactant_file = osp.join(folder_path, f"{prefix}{reactant_suffix}")
            ts_file = osp.join(folder_path, f"{prefix}{ts_suffix}")
            product_file = osp.join(folder_path, f"{prefix}{product_suffix}")

            if not (osp.exists(reactant_file) and osp.exists(ts_file) and osp.exists(product_file)):
                print(
                    f"Warning: One or more xyz files are missing in {folder_path}, skipping reaction_id {reaction_id}")
                print(f"Looked for: {reactant_file}, {ts_file}, {product_file}")
                continue

            # Read the reactant file and extract atomic symbols and coordinates
            symbols, pos0 = read_xyz(reactant_file)
            if symbols is None or pos0 is None:
                print(f"Warning: Failed to read {reactant_file}, skipping reaction_id {reaction_id}")
                continue
            z = symbols_to_atomic_numbers(symbols)
            if z is None:
                print(f"Warning: Conversion of atomic symbols failed, skipping reaction_id {reaction_id}")
                continue

            # Read ts and product files to extract coordinates
            _, pos1 = read_xyz(ts_file)
            _, pos2 = read_xyz(product_file)
            if pos1 is None or pos2 is None:
                print(f"Warning: Failed to read ts or product file, skipping reaction_id {reaction_id}")
                continue
            # Check that the number of atoms is consistent across files
            if not (pos0.size(0) == pos1.size(0) == pos2.size(0) == z.size(0)):
                print(f"Warning: Inconsistent atom count, skipping reaction_id {reaction_id}")
                continue

            y = torch.tensor([dG_ts], dtype=torch.float)
            data = Data(z=z, pos0=pos0, pos1=pos1, pos2=pos2, y=y)
            # Save additional attributes: CSV field ID is saved to reaction_id, R_dir is saved to id.
            data.reaction_id = reaction_id
            data.id = R_dir
            data.reaction = reaction_str  # Optionally save reaction description
            data.num_nodes = z.size(0)
            data_list.append(data)

        if len(data_list) == 0:
            raise RuntimeError("No reaction data processed, please check the CSV and xyz file formats.")

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self, train_size, valid_size, seed):
        """
        Split the dataset based on data.id (i.e., R_dir) to ensure that samples from the same reaction
        (typically from the same folder) are assigned to the same split, while approximately meeting the desired sample counts.

        Parameters:
          - train_size: Number of training samples (by sample count, not group count)
          - valid_size: Number of validation samples
          - seed: Random seed
        Returns:
          A dictionary containing 'train', 'valid', and 'test' keys with corresponding index tensors.
        """
        # Group samples by data.id (i.e., R_dir)
        group_to_indices = {}
        for idx, data in enumerate(self):
            key = data.id
            if key not in group_to_indices:
                group_to_indices[key] = []
            group_to_indices[key].append(idx)

        # Shuffle group keys using the provided seed
        group_keys = list(group_to_indices.keys())
        group_keys = shuffle(group_keys, random_state=seed)

        train_idx = []
        valid_idx = []
        test_idx = []
        n_assigned = 0

        # Assign entire groups sequentially to train, valid, and test until the desired counts are met
        for key in group_keys:
            group = group_to_indices[key]
            if n_assigned < train_size:
                train_idx.extend(group)
            elif n_assigned < train_size + valid_size:
                valid_idx.extend(group)
            else:
                test_idx.extend(group)
            n_assigned += len(group)

        train_idx = torch.tensor(train_idx, dtype=torch.long)
        valid_idx = torch.tensor(valid_idx, dtype=torch.long)
        test_idx = torch.tensor(test_idx, dtype=torch.long)
        return {'train': train_idx, 'valid': valid_idx, 'test': test_idx}

