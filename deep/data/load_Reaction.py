import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler

from .PygReaction import ReactionXYZDataset


def train_scaler(ds_list):
    """
    Fit and return a sklearn StandardScaler based on the target values (y)
    from a list of Reaction data objects (each element is a Data object).
    """
    ys = np.array([data.y.item() for data in ds_list]).reshape(-1, 1)
    scaler = StandardScaler().fit(ys)
    return scaler


def scale_reaction_dataset(ds_list, scaler):
    """
    Standardize the target values (y) of a list of Reaction data objects (each element is a Data object).
    Note that only the scalar target y is scaled, while pos0, pos1, and pos2 (the coordinates) remain unchanged.
    Returns a new list where each Data object is a copy with the target value replaced by the scaled value.
    """
    ys = np.array([data.y.item() for data in ds_list]).reshape(-1, 1)
    ys_scaled = scaler.transform(ys) if scaler else ys

    new_data_list = []
    for i, data in enumerate(ds_list):
        d = Data(
            # Retain atomic numbers
            z=data.z,
            # Retain the three sets of coordinates
            pos0=data.pos0,
            pos1=data.pos1,
            pos2=data.pos2,
            # Replace the target value with the scaled version
            y=torch.tensor(ys_scaled[i], dtype=torch.float),
            # Required number of nodes for PyG
            num_nodes=data.num_nodes
        )
        # If you wish to preserve additional fields like reaction_id, id (R_dir), reaction, etc., copy them manually.
        if hasattr(data, 'reaction_id'):
            d.reaction_id = data.reaction_id
        if hasattr(data, 'id'):
            d.id = data.id
        if hasattr(data, 'reaction'):
            d.reaction = data.reaction
        new_data_list.append(d)

    return new_data_list


def load_reaction(
        random_seed: int,
        root: str,
        csv_file: str = 'DA_dataset_cleaned.csv',
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        use_scaler: bool = False,
):
    """
    Similar to the load_qm9.py function, this function loads the Reaction dataset,
    splits it into train/validation/test sets, and standardizes the target values.

    Parameters:
    --------
    random_seed : int
        The random seed used for data splitting.
    root : str
        The root directory of the dataset, which should contain the csv_file and corresponding xyz folders
        (see the documentation in PygReaction.py).
    csv_file : str
        The CSV file name, default is 'DA_dataset_cleaned.csv'.
    train_ratio : float
        The proportion of the dataset to be used for training.
    val_ratio : float
        The proportion of the dataset to be used for validation.
        (The remainder is used for testing.)

    Returns:
    --------
    (train_scaled, val_scaled, test_scaled, scaler)
      - train_scaled, val_scaled, test_scaled: list[Data]
        Lists of Data objects for the training, validation, and test sets (with standardized y values).
      - scaler: The StandardScaler fitted on the training set, which can be used to invert the scaling.
    """

    # Optionally fix or assert the random seed
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # 1) Load the custom ReactionXYZDataset
    dataset = ReactionXYZDataset(root=root, csv_file=csv_file)

    # 2) Calculate train_size and valid_size based on the provided ratios
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    # The rest of the samples belong to the test set

    # 3) Split the dataset using the get_idx_split method provided in PygReaction.py
    splits = dataset.get_idx_split(train_size, val_size, seed=random_seed)
    train_indices = splits['train']
    val_indices = splits['valid']
    test_indices = splits['test']

    # 4) Extract data corresponding to the indices
    train_data = [dataset[i] for i in train_indices]
    val_data = [dataset[i] for i in val_indices]
    test_data = [dataset[i] for i in test_indices]

    # 5) Fit a StandardScaler on the target values (y) from the training set
    scaler = train_scaler(train_data) if use_scaler else None

    # 6) Scale the target values for train, validation, and test sets
    train_scaled = scale_reaction_dataset(train_data, scaler)
    val_scaled = scale_reaction_dataset(val_data, scaler)
    test_scaled = scale_reaction_dataset(test_data, scaler)

    return train_scaled, val_scaled, test_scaled, scaler
