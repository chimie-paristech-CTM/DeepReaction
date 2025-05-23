#!/usr/bin/env python
import os
import subprocess
import sys
from setuptools import setup, find_packages
from setuptools.command.install import install
from setuptools.command.develop import develop

def get_version():
    """Extract version from __init__.py without importing the package."""
    version_file = os.path.join(os.path.dirname(__file__), "deepreaction", "__init__.py")
    with open(version_file, "r") as f:
        for line in f:
            if line.startswith("__version__"):
                return line.strip().split("=")[1].strip(" \"'")
    return "0.1.0"

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

def get_torch_version():
    """Get the installed PyTorch version."""
    try:
        import torch
        version = torch.__version__.split('+')[0]
        # Ensure compatibility with PyG (max version 2.4.x)
        major, minor = map(int, version.split('.')[:2])
        if major > 2 or (major == 2 and minor > 4):
            print(f"Warning: PyTorch {version} detected. PyG currently supports PyTorch <= 2.4.x")
            print("Consider downgrading PyTorch: pip install 'torch<=2.4.1'")
        return version
    except ImportError:
        return None

def get_cuda_version():
    """Get CUDA version compatible with PyTorch."""
    try:
        import torch
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            if cuda_version:
                # Format CUDA version for PyG URLs (e.g., '11.8' -> 'cu118')
                return f"cu{cuda_version.replace('.', '')}"
        return "cpu"
    except (ImportError, AttributeError):
        return "cpu"

def install_pyg_dependencies():
    """Install PyG dependencies with correct versions."""
    print("Checking PyTorch installation...")
    torch_version = get_torch_version()
    
    if not torch_version:
        print("PyTorch not found. Installing compatible version...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "torch<=2.4.1", "torchvision", "torchaudio"
        ])
        torch_version = get_torch_version()
    
    cuda_version = get_cuda_version()
    print(f"Detected PyTorch: {torch_version}, CUDA: {cuda_version}")
    
    # PyG packages to install
    pyg_packages = [
        "torch-scatter",
        "torch-sparse", 
        "torch-cluster",
        "torch-spline-conv"
    ]
    
    # Construct the find-links URL
    find_links_url = f"https://data.pyg.org/whl/torch-{torch_version}+{cuda_version}.html"
    
    print(f"Installing PyG dependencies from: {find_links_url}")
    
    # Install PyG packages with find-links
    cmd = [
        sys.executable, "-m", "pip", "install",
        "--find-links", find_links_url
    ] + pyg_packages
    
    try:
        subprocess.check_call(cmd)
        print("PyG dependencies installed successfully.")
        
        # Install torch-geometric separately (often works better)
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "torch-geometric"
        ])
        print("torch-geometric installed successfully.")
        
    except subprocess.CalledProcessError as e:
        print(f"Error installing PyG dependencies: {e}")
        print("Trying alternative installation method...")
        
        # Try installing each package individually
        for package in pyg_packages:
            try:
                cmd_individual = [
                    sys.executable, "-m", "pip", "install",
                    "--find-links", find_links_url,
                    package
                ]
                subprocess.check_call(cmd_individual)
                print(f"Successfully installed {package}")
            except subprocess.CalledProcessError:
                print(f"Failed to install {package}")
        
        # Try installing torch-geometric
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "torch-geometric"
            ])
        except subprocess.CalledProcessError:
            print("Failed to install torch-geometric")
            
        print("\nIf installation failed, please try manually:")
        print(f"pip install torch-scatter torch-sparse torch-cluster torch-spline-conv --find-links {find_links_url}")
        print("pip install torch-geometric")

class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        install.run(self)
        install_pyg_dependencies()

class PostDevelopCommand(develop):
    """Post-installation for development mode."""
    def run(self):
        develop.run(self)
        install_pyg_dependencies()

# Core dependencies (without PyG)
REQUIREMENTS = [
    "torch<=2.4.1",  # Ensure compatibility with PyG
    "pytorch-lightning>=2.0.0,<3.0.0",
    "numpy>=1.20.0",
    "pandas>=1.3.0",
    "scipy>=1.7.0",
    "scikit-learn>=1.0.0",
    "matplotlib>=3.5.0",
    "rdkit>=2022.3.5",
    "pyyaml>=6.0",
    "tqdm>=4.62.0",
    "sympy>=1.10.0",
    "tensorboard>=2.10.0",
    "seaborn>=0.11.0",
    "ipywidgets>=8.0.0",
]

EXTRAS_REQUIRE = {
    "dev": [
        "black",
        "flake8", 
        "isort",
        "pytest",
        "pytest-cov",
        "mypy",
    ],
    "notebook": [
        "jupyterlab>=3.5.0",
        "notebook>=6.0.0",
    ],
    "full": [
        "wandb>=0.13.0",
        "optuna>=3.0.0",
        "ray[tune]>=2.0.0",
        "plotly>=5.0.0",
    ],
}

EXTRAS_REQUIRE['all'] = [item for sublist in EXTRAS_REQUIRE.values() for item in sublist]

if __name__ == "__main__":
    setup(
        name="deepreaction",
        version=get_version(),
        author="Bowen Deng",
        author_email="a18608202465@gmail.com",
        description="Deep learning framework for chemical reaction property prediction",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/chimie-paristech-CTM/DeepReaction",
        packages=find_packages(),
        include_package_data=True,
        package_data={
            "deepreaction": ["assets/*"],
        },
        install_requires=REQUIREMENTS,
        extras_require=EXTRAS_REQUIRE,
        cmdclass={
            'install': PostInstallCommand,
            'develop': PostDevelopCommand,
        },
        entry_points={
            "console_scripts": [
                "deepreaction-train=deepreaction.cli.train:main",
            ],
        },
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Science/Research", 
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.10",
            "Topic :: Scientific/Engineering :: Chemistry",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
        python_requires=">=3.10",
    )