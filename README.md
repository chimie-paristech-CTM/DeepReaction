# Deep Reaction

<div align="center">
  <img src="deepreaction/assets/reaction.jpg" width="100px" alt="Deep Reaction Logo" />
  <p><strong>Efficient Prediction of Chemical reactions</strong></p>
</div>

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.0-orange.svg)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-latest-red.svg)](https://pytorch-geometric.readthedocs.io/)

This repository corresponds to the DeepReaction project.



### Installation

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
> If needed, you can manually install PyG with:
>




### Dataset Preparation

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
./example/train_reaction.py
```

### Using Jupyter Notebook (Interactive)

We provide an interactive Jupyter notebook for easier experimentation and visualization:

```bash
# Start Jupyter notebook server
jupyter lab

# Navigate to example/train_reaction.ipynb
```

The notebook `example/train_reaction.ipynb` offers a streamlined interface for:
- Loading and exploring the reaction dataset
- Configuring model architecture and training parameters
- Training with real-time visualization of progress
- Evaluating model performance with interactive visualizations

This is particularly useful for quick experimentation and educational purposes, allowing you to:
- Modify hyperparameters and immediately see their effects
- Visualize the training process and results in real-time
- Interact with model predictions and understand performance characteristics

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
# Basic evaluation
./example/predict_reaction.ipynb
```

## 🔧 Advanced Usage


### Hyperparameter Optimization

```bash
# Run hyperparameter optimization
./example/hyperopt_grid.py
```

## 📝 Citation

```
[Placeholder]
```

## 🙏 Acknowledgements

This implementation is built upon several open-source projects:

- [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric)
- [PyTorch Lightning](https://github.com/Lightning-AI/lightning)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
