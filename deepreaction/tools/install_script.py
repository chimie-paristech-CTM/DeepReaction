#!/usr/bin/env python
"""Installation helper script for DeepReaction package."""

import argparse
import subprocess
import sys
import os


def get_torch_version():
    """Get the PyTorch version if it's installed."""
    try:
        import torch
        return torch.__version__.split('+')[0]
    except ImportError:
        return None


def get_cuda_version():
    """Get CUDA version if PyTorch with CUDA is installed."""
    try:
        import torch
        if torch.cuda.is_available():
            # Get CUDA version, e.g., '11.7'
            cuda_version = torch.version.cuda
            # Format to 'cu117' style
            return f"cu{cuda_version.replace('.', '')}"
        else:
            return "cpu"
    except (ImportError, AttributeError):
        return "cpu"


def install_pyg_dependencies(verbose=True, force=False):
    """Install PyG dependencies based on detected torch and CUDA versions."""
    torch_version = get_torch_version()
    cuda_version = get_cuda_version()

    if not torch_version:
        print("PyTorch not found. Please install PyTorch first.")
        print("Visit https://pytorch.org/get-started/locally/ for installation instructions.")
        return False

    pyg_deps = ["torch-scatter", "torch-sparse", "torch-cluster", "torch-spline-conv"]
    url = f"https://data.pyg.org/whl/torch-{torch_version}+{cuda_version}.html"

    if verbose:
        print(f"Installing PyTorch Geometric dependencies for torch-{torch_version}+{cuda_version}...")

    # Check if dependencies are already installed
    if not force:
        try:
            for dep in pyg_deps:
                __import__(dep.replace('-', '_'))
            if verbose:
                print("PyTorch Geometric dependencies are already installed.")
            return True
        except ImportError:
            pass  # Continue with installation

    cmd = [sys.executable, "-m", "pip", "install"] + pyg_deps + ["-f", url]

    try:
        subprocess.check_call(cmd)
        if verbose:
            print("Successfully installed PyTorch Geometric dependencies.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing PyG dependencies: {e}")
        print(f"You may need to install them manually with:")
        print(f"pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f {url}")
        return False


def install_full_dependencies(verbose=True):
    """Install all optional dependencies."""
    cmd = [sys.executable, "-m", "pip", "install", "deepreaction[all]"]

    try:
        subprocess.check_call(cmd)
        if verbose:
            print("Successfully installed all optional dependencies.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing optional dependencies: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Install DeepReaction with dependencies")
    parser.add_argument("--pyg-only", action="store_true", help="Only install PyG dependencies")
    parser.add_argument("--full", action="store_true", help="Install all optional dependencies")
    parser.add_argument("--force", action="store_true", help="Force reinstallation of dependencies")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    if args.pyg_only:
        return install_pyg_dependencies(verbose=args.verbose, force=args.force)

    if args.full:
        success = install_full_dependencies(verbose=args.verbose)
        if success:
            return install_pyg_dependencies(verbose=args.verbose, force=args.force)
        return False

    # Print torch and CUDA information
    torch_version = get_torch_version()
    cuda_version = get_cuda_version()

    print("\nDeepReaction Installation Helper")
    print("==============================")
    print(f"Detected PyTorch: {torch_version or 'Not installed'}")
    print(f"Detected CUDA: {cuda_version}")

    if torch_version:
        print("\nRecommended installation commands:")
        print(
            f"pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-{torch_version}+{cuda_version}.html")
        print("\nTo install all optional dependencies:")
        print("pip install deepreaction[all]")
    else:
        print("\nPlease install PyTorch first:")
        print("Visit https://pytorch.org/get-started/locally/ for installation instructions.")

    install = input("\nWould you like to install PyG dependencies now? [y/N]: ")
    if install.lower() in ["y", "yes"]:
        install_pyg_dependencies(verbose=True, force=args.force)

    install_full = input("Would you like to install all optional dependencies? [y/N]: ")
    if install_full.lower() in ["y", "yes"]:
        install_full_dependencies(verbose=True)

    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)