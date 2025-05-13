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

## 🚀 Getting Started

### Installation

```bash
# Clone this repository:
git clone https://github.com/chimie-paristech-CTM/DeepReaction.git
cd DeepReaction

# Create and activate the environment
conda create -n reaction python=3.10
conda activate reaction

# Install PyTorch with CUDA support
conda install pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 -c pytorch

# Install PyTorch Geometric
conda install pyg -c pyg

# Install other dependencies
conda env create -f environment.yml

# For Jupyter notebook functionality
pip install jupyterlab
```

### Dataset Preparation

Place your reaction dataset in the appropriate location:

```
./Dataset
```

Alternatively, modify the paths in the configuration file or command-line arguments.

## 💻 Training

### Using Command Line Interface

To train the model with the dataset using our specialized training script:

```bash
# Basic training with default parameters
./example/train.sh
```

### Using Jupyter Notebook (Interactive)

We provide an interactive Jupyter notebook for easier experimentation and visualization:

```bash
# Start Jupyter notebook server
jupyter lab

# Navigate to example/train_demo.ipynb
```

The notebook `example/train_demo.ipynb` offers a streamlined interface for:
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

```bash
# Basic evaluation
./example/inference.sh
```

## 🔧 Advanced Usage

### Fine-tuning Pre-trained Models

```bash
# Fine-tune a pre-trained model on new data
./example/finetune.sh 
```

### Hyperparameter Optimization

```bash
# Run hyperparameter optimization (grid_search)
./example/hyperopt.sh 
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
