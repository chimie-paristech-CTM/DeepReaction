<div align="center">
  <h1>Deep Reaction</h1>
</div>
<div align="center">
  <img src="./deep/assets/reaction.jpg" width="100px" />

[//]: # (  <h3>Efficient Prediction of Molecular Properties</h3>)
  <div>

[//]: # (    [Name])
  </div>
</div>

---
# DeepReaction

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

This repository implements an advanced molecular prediction framework using various readout functions. Our implementation enhances the state-of-the-art performance for predicting reaction datasets.

## Features

- **Architecture**: Graph neural network for efficient molecular representation
- **Multiple Readout Functions**: Support for various readout functions including:
- **PyTorch Lightning Framework**: Clean, modular implementation with easy training and evaluation
- **Reaction Dataset Support**: Specialized handling for the reaction dataset
- **Flexible Configuration**: Extensive hyperparameter customization via YAML config or command-line arguments
- **Interactive Notebook Interface**: User-friendly Jupyter notebook for training and experimentation

## Main Results

### Benchmark on Reaction Dataset

| Model           | Readout         | MAE   | RMSE  | R²    | #Params | Training Time |
|:---------------:|:---------------:|:-----:|:-----:|:-----:|:-------:|:------------:|
| Small | Set Transformer | x.xx  | x.xx  | x.xx  | xx.xM   | xxx min       |
| Base  | Set Transformer | x.xx  | x.xx  | x.xx  | xx.xM   | xxx min       |
| Large | Set Transformer | x.xx  | x.xx  | x.xx  | xx.xM   | xxx min       |
| Base  | Sum             | x.xx  | x.xx  | x.xx  | xx.xM   | xxx min       |
| Base  | Attention       | x.xx  | x.xx  | x.xx  | xx.xM   | xxx min       |

### Visualization of Prediction Performance
[Placeholder for performance visualization]


## Getting Started

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
pip install -r requirements.txt

# For Jupyter notebook functionality
pip install jupyter notebook matplotlib seaborn
```

### Dataset Preparation
Place your reaction dataset in the appropriate location:
```
./Dataset
```

Alternatively, modify the paths in the configuration file or command-line arguments.

## Training

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
example/train_demo.ipynb

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

### Available Training Options

- `--readout`: Readout function type (set_transformer, sum, mean, max, attention)
- `--batch`: Batch size for training
- `--epochs`: Maximum number of training epochs
- `--lr`: Learning rate
- `--node-dim`: Dimension of node latent representations
- `--output`: Output directory for results
- `--reaction-root`: Custom path to reaction dataset root
- `--reaction-csv`: Custom path to reaction dataset CSV
...



## Evaluation

To evaluate a trained model:

```bash
# Basic evaluation
./example/inference.sh

```

[//]: # (## Project Structure)

[//]: # ()
[//]: # (```)

[//]: # (├── deep/                # Main model code)

[//]: # (│   ├── cli/             # Command-line interface)

[//]: # (│   │   ├── config.py    # Configuration handling)

[//]: # (│   │   ├── train.py     # Training script)

[//]: # (│   │   ├── finetune.py  # Fine-tuning script)

[//]: # (│   │   ├── inference.py # Inference script)

[//]: # (│   │   └── hyperopt.py  # Hyperparameter optimization)

[//]: # (│   ├── data/            # Data loading utilities)

[//]: # (│   ├── model/           # Model definitions)

[//]: # (│   │   └── model.py     # Model implementation)

[//]: # (│   ├── module/          # PyTorch Lightning modules)

[//]: # (│   │   └── pl_wrap.py   # Lightning wrapper for models)

[//]: # (│   └── utils/           # Utility functions)

[//]: # (│       ├── metrics.py   # Evaluation metrics)

[//]: # (│       ├── model_utils.py  # Model utilities)

[//]: # (│       └── visualization.py  # Visualization tools)

[//]: # (├── example/             # Example scripts and notebooks)

[//]: # (│   ├── train.sh         # Training script for XTB dataset)

[//]: # (│   ├── inference.sh     # Inference script)

[//]: # (│   └── train_demo.ipynb # Interactive Jupyter notebook for training)

[//]: # (├── configs/             # Configuration files)

[//]: # (└── README.md            # This file)

[//]: # (```)

## Advanced Usage

### Fine-tuning Pre-trained Models

```bash
# Fine-tune a pre-trained model on new data
./example/finetune.sh 
```

### Hyperparameter Optimization

```bash
# Run hyperparameter optimization
./example/hyperopt.sh 
```

## Citation


```
[Placeholder for your citation information]
```

## Acknowledgements

This implementation is built upon several open-source projects:

- [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric)
- [PyTorch Lightning](https://github.com/Lightning-AI/lightning)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.