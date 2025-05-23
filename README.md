# Deep Reaction

<div align="center">
  <img src="./assets/reaction.jpg" width="100px" alt="Deep Reaction Logo" />
  <p><strong>Efficient Prediction of Chemical reactions</strong></p>
</div>

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.0-orange.svg)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-latest-red.svg)](https://pytorch-geometric.readthedocs.io/)

This repository corresponds to the DeepReaction project, designed for accurate prediction of chemical reaction properties using graph neural networks.

## 📋 Table of Contents

- [Installation](#installation)
- [Data Format](#data-format)
- [Dataset](#dataset)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Advanced Usage](#advanced-usage)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## 🔧 Installation

### Method 1: Using pip (Recommended)

```bash
# Clone this repository:
git clone https://github.com/chimie-paristech-CTM/DeepReaction.git
cd DeepReaction

# Install in development mode
pip install -e .

# (Optional) For Jupyter notebook support
pip install jupyterlab
```

### Method 2: Using conda environment

```bash
# Clone this repository:
git clone https://github.com/chimie-paristech-CTM/DeepReaction.git
cd DeepReaction

# Create the conda environment from the environment.yml file
conda env create -f environment.yml

# Activate the environment
conda activate reaction

# (Optional) For Jupyter notebook support
pip install jupyterlab
```

> ⚠️ **Note:** The version of **PyTorch Geometric (PyG)** and its related packages must be selected according to your hardware configuration (e.g., CUDA version).
> Visit the official [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) to find the correct command for your system.
> ⚠️ **Note:** Due to the computational complexity of graph neural network architectures built with PyG (PyTorch Geometric), it is recommended to run them on a GPU for better performance and efficiency.

## 📊 Data Format

DeepReaction requires a specific data format for training and prediction. The key components are:

### CSV Input Format

Your main dataset file should be a CSV with the following essential columns:

| Column       | Description |
|--------------|-------------|
| `ID`         | Unique identifier for each reaction |
| `R_dir`      | Directory name containing XYZ files (e.g., "reaction_R0") |
| `smiles`     | SMILES representation of the reaction |
| `DG_act`     |  Target property: Gibbs free activation energy (kcal/mol) |
| `DrG`        |  Target property: Gibbs free reaction energy (kcal/mol) |
| `DG_act_xtb` |  Input feature: XTB-computed approximation of DG_act |
| `DrG_xtb`    | Input feature: XTB-computed approximation of DrG |

#### Example CSV row:
```
ID63623,reaction_R0,[C:1](=[C:2]([C:3](=[C:4]([H:11])[H:12])[H:10])[H:9])([H:7])[H:8].[C:5](=[C:6]([H:15])[H:16])([H:13])[H:14]>>[C:1]1([H:7])([H:8])[C:2]([H:9])=[C:3]([H:10])[C:4]([H:11])([H:12])[C:5]([H:13])([H:14])[C:6]1([H:15])[H:16],35.16,-22.54,21.70,-44.40
```

### XYZ File Structure

For each reaction in your dataset, you need to provide three XYZ files representing the:
1. Reactant(s)
2. Transition state (TS)
3. Product(s)

The XYZ files should be organized in directories named according to the `R_dir` column in your CSV:

```
dataset_root/
└── reaction_R0/
    ├── R0_reactant.xyz
    ├── R0_ts.xyz
    └── R0_product.xyz
└── reaction_R1/
    ├── R1_reactant.xyz
    ├── R1_ts.xyz
    └── R1_product.xyz
...
```

#### XYZ File Format:
```
[Number of atoms]
[Optional comment line]
[Element] [X coordinate] [Y coordinate] [Z coordinate]
[Element] [X coordinate] [Y coordinate] [Z coordinate]
...
```

### Important Configuration Parameters

When setting up your configuration, make sure to specify:

- `file_patterns`: Patterns to identify XYZ files (default: `['*_reactant.xyz', '*_ts.xyz', '*_product.xyz']`)
- `target_fields`:  Target properties to predict (default: `['DG_act', 'DrG']`)
- `input_features`:  Features used as input (default: `['DG_act_xtb', 'DrG_xtb']`)
- `id_field`: Column name for reaction IDs (default: `'ID'`)
- `dir_field`: Column name for directory names (default: `'R_dir'`)
- `reaction_field`: Column name for reaction SMILES (default: `'reaction'`)

## 🔍 Dataset

### Diels-Alder Reaction Dataset

The models in DeepReaction were developed and tested using a comprehensive Diels-Alder reaction dataset:

**Dataset link:** [Diels-Alder Reaction Space for Self-Healing Polymer](https://figshare.com/articles/dataset/Diels-Alder_reaction_space_for_self-healing_polymer/29118509?file=54702098)

This dataset contains:
- 1,580+ Diels-Alder reactions with complete 3D structures
- Quantum chemical calculations (DFT and XTB) for transition states and energetics
- Reaction energies, activation energies, and structural information
-  Computed properties including DG_act and DrG values

### Download and Use

1. Download the dataset archive from the Figshare link above
2. Extract the contents to your desired location (recommended: `./dataset/DATASET_DA_F/`)
3. Ensure the dataset has the correct structure as described in the [XYZ File Structure](#xyz-file-structure) section
4. Update the dataset paths in your configuration if needed

## 📁 Dataset Preparation

Place your reaction dataset in the appropriate location:

```
./Dataset
```

Alternatively, modify the paths in the configuration file or command-line arguments.

## 💻 Training

### Using Command Line Interface

To train the model with the dataset using our specialized training script:

```
# Basic training with default parameters
cp ./example/train.py .
python train.py
```

### Available command line options

- `--readout`: Readout function type (set_transformer, sum, mean, max, attention)
- `--batch`: Batch size for training
- `--epochs`: Maximum number of training epochs
- `--lr`: Learning rate
- `--node-dim`: Dimension of node latent representations
- `--output`: Output directory for results
- `--reaction-root`: Custom path to reaction dataset root, i.e., the location of the xyz files of reactants, products and TSs
- `--reaction-csv`: Custom path to reaction dataset CSV

## 📈 Evaluation

To evaluate a trained model:

```
cp ./example/predict.py .
python predict.py
```

The prediction notebook allows you to:
- Load a trained model checkpoint
- Make predictions on new data
- Visualize prediction results
- Compare predictions with actual values (if available)

## 🔧 Advanced Usage

### Hyperparameter Optimization

```bash
# Run hyperparameter optimization
cp ./example/hyper.py .
python hyper.py
```

## 📝 Citation

If you use DeepReaction or the Diels-Alder dataset in your research, please cite:

```
[Placeholder]
```

## 🙏 Acknowledgements

This implementation is built upon several open-source projects:

- [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric)
- [PyTorch Lightning](https://github.com/Lightning-AI/lightning)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.