"""
Visualization utilities for model training, evaluation and analysis.

This module provides functions for visualizing training results, predictions,
feature importance, attention weights, and other model outputs.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
import pandas as pd
from scipy.stats import gaussian_kde
import tensorboard.backend.event_processing.event_accumulator as ea
import json


def setup_plot_style() -> None:
    """
    Set up matplotlib plot style for consistent visualizations.
    """
    # Set seaborn style
    sns.set_style("whitegrid")
    
    # Set matplotlib params
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['lines.linewidth'] = 2


def plot_predictions(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    output_dir: str,
    title: str = "Predicted vs. True Values",
    save_name: str = "predictions.png",
    show_density: bool = True,
    include_metrics: bool = True
) -> None:
    """
    Plot predicted vs. true values with optional density estimation and metrics.
    
    Args:
        y_true (np.ndarray): Ground truth values
        y_pred (np.ndarray): Predicted values
        output_dir (str): Directory to save the plot
        title (str, optional): Plot title. Defaults to "Predicted vs. True Values".
        save_name (str, optional): Filename to save plot. Defaults to "predictions.png".
        show_density (bool, optional): Whether to show density estimation. Defaults to True.
        include_metrics (bool, optional): Whether to include metrics in the plot. Defaults to True.
    """
    # Set up plot style
    setup_plot_style()
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Flatten arrays if needed
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    # Calculate metrics if requested
    if include_metrics:
        from .metrics import compute_regression_metrics
        metrics = compute_regression_metrics(y_true, y_pred)
        rmse = metrics.get('rmse', 0)
        mae = metrics.get('mae', 0)
        r2 = metrics.get('r2', 0)
        
        # Add metrics to title
        title += f"\nRMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}"
    
    # Plot density if requested
    if show_density and len(y_true) > 10:
        try:
            # Create a 2D histogram
            h, xedges, yedges = np.histogram2d(y_true, y_pred, bins=50, density=True)
            
            # Plot the 2D histogram as a heatmap
            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
            im = ax.imshow(h.T, extent=extent, origin='lower', aspect='auto', 
                         cmap='viridis', alpha=0.7)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Density')
            
            # Add scatter plot with transparency
            ax.scatter(y_true, y_pred, c='white', alpha=0.3, s=10, edgecolors='none')
        except Exception as e:
            # Fall back to scatter plot if density estimation fails
            print(f"Density estimation failed: {e}. Falling back to scatter plot.")
            ax.scatter(y_true, y_pred, alpha=0.5, s=20)
    else:
        # Simple scatter plot
        ax.scatter(y_true, y_pred, alpha=0.5, s=20)
    
    # Add diagonal line (perfect predictions)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    margin = 0.1 * (max_val - min_val)
    ax.plot([min_val - margin, max_val + margin], 
            [min_val - margin, max_val + margin], 
            'r--', label='Perfect Prediction')
    
    # Set plot limits with margins
    ax.set_xlim(min_val - margin, max_val + margin)
    ax.set_ylim(min_val - margin, max_val + margin)
    
    # Add labels and title
    ax.set_xlabel('True Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title(title)
    
    # Add legend
    ax.legend()
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Save the figure
    plt.tight_layout()
    save_path = os.path.join(output_dir, save_name)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Create additional visualization: Residual plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate residuals
    residuals = y_pred - y_true
    
    # Plot residuals
    ax.scatter(y_true, residuals, alpha=0.5, s=20)
    ax.axhline(y=0, color='r', linestyle='--')
    
    # Add labels
    ax.set_xlabel('True Values')
    ax.set_ylabel('Residuals (Predicted - True)')
    ax.set_title('Residual Plot')
    
    # Save the residual plot
    plt.tight_layout()
    residuals_path = os.path.join(output_dir, f"residuals_{save_name}")
    plt.savefig(residuals_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Create error distribution plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot error distribution
    sns.histplot(residuals, kde=True, ax=ax)
    ax.axvline(x=0, color='r', linestyle='--')
    
    # Add labels
    ax.set_xlabel('Prediction Error')
    ax.set_ylabel('Frequency')
    ax.set_title('Error Distribution')
    
    # Save the error distribution plot
    plt.tight_layout()
    error_dist_path = os.path.join(output_dir, f"error_distribution_{save_name}")
    plt.savefig(error_dist_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_loss_curves(
    tensorboard_dir: str, 
    output_dir: str,
    metrics: List[str] = None
) -> None:
    """
    Plot training and validation loss curves from TensorBoard logs.
    
    Args:
        tensorboard_dir (str): Directory containing TensorBoard logs
        output_dir (str): Directory to save the plots
        metrics (List[str], optional): List of metrics to plot. Defaults to None.
    """
    # Set up plot style
    setup_plot_style()
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Default metrics if none provided
    if metrics is None:
        metrics = ['loss', 'rmse', 'mae']
    
    # Find the most recent event file
    event_files = [os.path.join(root, file) 
                   for root, _, files in os.walk(tensorboard_dir) 
                   for file in files if file.startswith('events.out')]
    
    if not event_files:
        print(f"No TensorBoard event files found in {tensorboard_dir}")
        return
    
    # Sort by modification time (newest first)
    event_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    latest_event_file = event_files[0]
    
    # Load event file
    event_acc = ea.EventAccumulator(latest_event_file)
    event_acc.Reload()
    
    # Get available tags
    available_tags = event_acc.Tags()['scalars']
    
    # Filter for relevant metrics
    for metric in metrics:
        # Find relevant tags for this metric
        train_tag = None
        val_tag = None
        
        for tag in available_tags:
            # Check for train metric
            if f'train_{metric}' in tag or (tag.startswith('train/') and metric in tag):
                train_tag = tag
            # Check for val metric
            if f'val_{metric}' in tag or (tag.startswith('val/') and metric in tag):
                val_tag = tag
        
        if train_tag is None and val_tag is None:
            print(f"No tags found for metric: {metric}")
            continue
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Get train data if available
        if train_tag:
            train_events = event_acc.Scalars(train_tag)
            train_steps = [event.step for event in train_events]
            train_values = [event.value for event in train_events]
            ax.plot(train_steps, train_values, label=f'Training {metric}')
        
        # Get validation data if available
        if val_tag:
            val_events = event_acc.Scalars(val_tag)
            val_steps = [event.step for event in val_events]
            val_values = [event.value for event in val_events]
            ax.plot(val_steps, val_values, label=f'Validation {metric}')
        
        # Add labels and title
        ax.set_xlabel('Steps')
        ax.set_ylabel(metric.upper())
        ax.set_title(f'{metric.upper()} during Training')
        
        # Add legend
        ax.legend()
        
        # Save the figure
        plt.tight_layout()
        save_path = os.path.join(output_dir, f"{metric}_curve.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)


def plot_feature_importance(
    feature_names: List[str],
    importance_scores: np.ndarray,
    output_dir: str,
    title: str = "Feature Importance",
    save_name: str = "feature_importance.png",
    top_n: int = 20
) -> None:
    """
    Plot feature importance scores.
    
    Args:
        feature_names (List[str]): Names of features
        importance_scores (np.ndarray): Importance scores for each feature
        output_dir (str): Directory to save the plot
        title (str, optional): Plot title. Defaults to "Feature Importance".
        save_name (str, optional): Filename to save plot. Defaults to "feature_importance.png".
        top_n (int, optional): Number of top features to show. Defaults to 20.
    """
    # Set up plot style
    setup_plot_style()
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a DataFrame for better handling
    df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance_scores
    })
    
    # Sort by importance
    df = df.sort_values('Importance', ascending=False)
    
    # Get top N features
    if len(df) > top_n:
        df = df.head(top_n)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, max(6, len(df) * 0.3)))
    
    # Plot horizontal bar chart
    bars = ax.barh(df['Feature'], df['Importance'])
    
    # Add color gradient
    norm = Normalize(vmin=df['Importance'].min(), vmax=df['Importance'].max())
    colors = cm.viridis(norm(df['Importance']))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    # Add value labels
    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(row['Importance'] + 0.01, i, f"{row['Importance']:.4f}", 
               va='center', ha='left', fontsize=10)
    
    # Add labels and title
    ax.set_xlabel('Importance Score')
    ax.set_title(title)
    
    # Adjust y-axis
    ax.invert_yaxis()  # Highest importance on top
    
    # Save the figure
    plt.tight_layout()
    save_path = os.path.join(output_dir, save_name)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_attention_weights(
    output_dir: str,
    attention_file: str = 'attention_weights.npz',
    plot_dir: str = None,
    max_molecules: int = 10,
    title_prefix: str = "Attention Weights"
) -> None:
    """
    Plot attention weights for molecules.
    
    Args:
        output_dir (str): Directory containing attention weights file
        attention_file (str, optional): Filename of attention weights. Defaults to 'attention_weights.npz'.
        plot_dir (str, optional): Directory to save the plots. Defaults to output_dir/attention_plots.
        max_molecules (int, optional): Maximum number of molecules to plot. Defaults to 10.
        title_prefix (str, optional): Prefix for plot titles. Defaults to "Attention Weights".
    """
    # Set up plot style
    setup_plot_style()
    
    # Set plot directory if not specified
    if plot_dir is None:
        plot_dir = os.path.join(output_dir, 'attention_plots')
    
    # Ensure plot directory exists
    os.makedirs(plot_dir, exist_ok=True)
    
    # Path to attention weights file
    attention_path = os.path.join(output_dir, attention_file)
    
    # Check if file exists
    if not os.path.exists(attention_path):
        print(f"Attention weights file not found: {attention_path}")
        return
    
    try:
        # Load attention weights
        data = np.load(attention_path, allow_pickle=True)
        
        # Extract data
        if 'attention_weights' in data:
            attention_weights = data['attention_weights']
            molecule_ids = data.get('molecule_ids', np.arange(len(attention_weights)))
            atom_types = data.get('atom_types', None)
        else:
            # Try to load as a dictionary
            attention_weights = data
            molecule_ids = np.arange(len(attention_weights))
            atom_types = None
        
        # Limit number of molecules to plot
        if len(attention_weights) > max_molecules:
            indices = np.random.choice(len(attention_weights), max_molecules, replace=False)
            attention_weights = attention_weights[indices]
            molecule_ids = molecule_ids[indices] if len(molecule_ids) > max_molecules else molecule_ids
            if atom_types is not None:
                atom_types = [atom_types[i] for i in indices] if len(atom_types) > max_molecules else atom_types
        
        # Plot attention weights for each molecule
        for i, weights in enumerate(attention_weights):
            if weights.ndim < 2:
                # Skip if weights are not a matrix
                continue
            
            # Get molecule ID
            mol_id = molecule_ids[i] if i < len(molecule_ids) else i
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Plot heatmap
            im = ax.imshow(weights, cmap='viridis')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Attention Weight')
            
            # Add labels for atoms if available
            if atom_types is not None and i < len(atom_types):
                atom_labels = atom_types[i]
                # Set tick labels if number of atoms is reasonable
                if len(atom_labels) <= 20:
                    ax.set_xticks(np.arange(len(atom_labels)))
                    ax.set_yticks(np.arange(len(atom_labels)))
                    ax.set_xticklabels(atom_labels)
                    ax.set_yticklabels(atom_labels)
                    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            
            # Add title
            ax.set_title(f"{title_prefix} - Molecule {mol_id}")
            
            # Save the figure
            plt.tight_layout()
            save_path = os.path.join(plot_dir, f"attention_weights_mol_{mol_id}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
    
    except Exception as e:
        print(f"Error plotting attention weights: {e}")


def plot_learning_rate(
    tensorboard_dir: str, 
    output_dir: str,
    save_name: str = "learning_rate.png"
) -> None:
    """
    Plot learning rate changes during training from TensorBoard logs.
    
    Args:
        tensorboard_dir (str): Directory containing TensorBoard logs
        output_dir (str): Directory to save the plot
        save_name (str, optional): Filename to save plot. Defaults to "learning_rate.png".
    """
    # Set up plot style
    setup_plot_style()
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Find the most recent event file
    event_files = [os.path.join(root, file) 
                   for root, _, files in os.walk(tensorboard_dir) 
                   for file in files if file.startswith('events.out')]
    
    if not event_files:
        print(f"No TensorBoard event files found in {tensorboard_dir}")
        return
    
    # Sort by modification time (newest first)
    event_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    latest_event_file = event_files[0]
    
    # Load event file
    event_acc = ea.EventAccumulator(latest_event_file)
    event_acc.Reload()
    
    # Get available tags
    available_tags = event_acc.Tags()['scalars']
    
    # Find learning rate tag
    lr_tag = None
    for tag in available_tags:
        if 'lr' in tag or 'learning_rate' in tag:
            lr_tag = tag
            break
    
    if lr_tag is None:
        print("No learning rate data found in TensorBoard logs")
        return
    
    # Get learning rate data
    lr_events = event_acc.Scalars(lr_tag)
    steps = [event.step for event in lr_events]
    lr_values = [event.value for event in lr_events]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot learning rate
    ax.plot(steps, lr_values, marker='o', markersize=3)
    
    # Use log scale for y-axis if range is large
    if max(lr_values) / (min(lr_values) + 1e-10) > 10:
        ax.set_yscale('log')
    
    # Add labels and title
    ax.set_xlabel('Steps')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    
    # Add grid
    ax.grid(True, which='both', linestyle='--', alpha=0.7)
    
    # Save the figure
    plt.tight_layout()
    save_path = os.path.join(output_dir, save_name)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_uncertainty(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    output_dir: str,
    z_score: float = 1.96,  # 95% confidence interval
    save_name: str = "prediction_uncertainty.png"
) -> None:
    """
    Plot predictions with uncertainty bounds.
    
    Args:
        y_true (np.ndarray): Ground truth values
        y_pred (np.ndarray): Predicted values
        y_std (np.ndarray): Standard deviation of predictions
        output_dir (str): Directory to save the plot
        z_score (float, optional): Z-score for confidence interval. Defaults to 1.96 (95% CI).
        save_name (str, optional): Filename to save plot. Defaults to "prediction_uncertainty.png".
    """
    # Set up plot style
    setup_plot_style()
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Flatten arrays if needed
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    y_std = y_std.flatten()
    
    # Sort by true values
    sort_idx = np.argsort(y_true)
    y_true = y_true[sort_idx]
    y_pred = y_pred[sort_idx]
    y_std = y_std[sort_idx]
    
    # Compute confidence intervals
    lower_bound = y_pred - z_score * y_std
    upper_bound = y_pred + z_score * y_std
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot predictions with confidence intervals
    ax.plot(y_true, y_true, 'r--', label='Perfect Prediction')
    ax.plot(y_true, y_pred, 'b-', label='Prediction')
    ax.fill_between(y_true, lower_bound, upper_bound, alpha=0.3, color='b', label=f'{z_score:.2f}σ Confidence Interval')
    
    # Add labels and title
    ax.set_xlabel('True Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title('Predictions with Uncertainty')
    
    # Add legend
    ax.legend()
    
    # Save the figure
    plt.tight_layout()
    save_path = os.path.join(output_dir, save_name)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Create calibration plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Compute coverage at different confidence levels
    z_scores = np.linspace(0, 3, 100)
    coverages = []
    
    for z in z_scores:
        lower = y_pred - z * y_std
        upper = y_pred + z * y_std
        coverage = np.mean((y_true >= lower) & (y_true <= upper))
        coverages.append(coverage)
    
    # Compute expected coverage
    expected_coverage = 2 * norm.cdf(z_scores) - 1
    
    # Plot calibration curve
    ax.plot(z_scores, coverages, label='Empirical Coverage')
    ax.plot(z_scores, expected_coverage, 'r--', label='Expected Coverage')
    
    # Add labels and title
    ax.set_xlabel('Z-score')
    ax.set_ylabel('Coverage Probability')
    ax.set_title('Uncertainty Calibration Plot')
    
    # Add legend
    ax.legend()
    
    # Save the figure
    plt.tight_layout()
    calibration_path = os.path.join(output_dir, f"calibration_{save_name}")
    plt.savefig(calibration_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_cross_validation_results(
    cv_metrics_path: str,
    output_dir: str,
    save_name: str = "cv_results.png"
) -> None:
    """
    Plot cross-validation results.
    
    Args:
        cv_metrics_path (str): Path to CV metrics JSON file
        output_dir (str): Directory to save the plot
        save_name (str, optional): Filename to save plot. Defaults to "cv_results.png".
    """
    # Set up plot style
    setup_plot_style()
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load CV metrics
    with open(cv_metrics_path, 'r') as f:
        cv_metrics = json.load(f)
    
    # Extract data
    avg_metrics = cv_metrics.get('avg', {})
    std_metrics = cv_metrics.get('std', {})
    fold_metrics = cv_metrics.get('folds', [])
    
    # Filter metrics (remove 'epoch' and other non-numeric metrics)
    metric_names = []
    metric_values = []
    metric_stds = []
    
    for key, value in avg_metrics.items():
        if isinstance(value, (int, float)) and key not in ['epoch', 'step']:
            metric_names.append(key)
            metric_values.append(value)
            metric_stds.append(std_metrics.get(key, 0))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot bar chart with error bars
    x = range(len(metric_names))
    bars = ax.bar(x, metric_values, yerr=metric_stds, capsize=5, alpha=0.7)
    
    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + metric_stds[i],
                f'{metric_values[i]:.4f}±{metric_stds[i]:.4f}',
                ha='center', va='bottom', rotation=0, fontsize=10)
    
    # Add labels and title
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Value')
    ax.set_title('Cross-Validation Results')
    
    # Set x-ticks
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation=45, ha='right')
    
    # Save the figure
    plt.tight_layout()
    save_path = os.path.join(output_dir, save_name)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Plot fold comparison for each metric
    for metric in metric_names:
        # Skip if metric is not present in all folds
        if not all(metric in fold for fold in fold_metrics):
            continue
        
        # Extract values for this metric
        fold_values = [fold[metric] for fold in fold_metrics]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot fold values
        ax.bar(range(1, len(fold_values) + 1), fold_values, alpha=0.7)
        
        # Add reference line for mean
        ax.axhline(y=avg_metrics[metric], color='r', linestyle='--', 
                  label=f'Mean: {avg_metrics[metric]:.4f}')
        
        # Add labels and title
        ax.set_xlabel('Fold')
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} Across Folds')
        
        # Set x-ticks
        ax.set_xticks(range(1, len(fold_values) + 1))
        
        # Add legend
        ax.legend()
        
        # Save the figure
        plt.tight_layout()
        metric_path = os.path.join(output_dir, f"{metric}_folds.png")
        plt.savefig(metric_path, dpi=300, bbox_inches='tight')
        plt.close(fig)


def plot_ensemble_comparison(
    ensemble_metrics_path: str,
    individual_metrics_paths: List[str],
    output_dir: str,
    save_name: str = "ensemble_comparison.png"
) -> None:
    """
    Plot comparison between ensemble model and individual models.
    
    Args:
        ensemble_metrics_path (str): Path to ensemble metrics JSON file
        individual_metrics_paths (List[str]): Paths to individual model metrics JSON files
        output_dir (str): Directory to save the plot
        save_name (str, optional): Filename to save plot. Defaults to "ensemble_comparison.png".
    """
    # Set up plot style
    setup_plot_style()
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load ensemble metrics
    with open(ensemble_metrics_path, 'r') as f:
        ensemble_metrics = json.load(f)
    
    # Load individual model metrics
    individual_metrics = []
    for path in individual_metrics_paths:
        with open(path, 'r') as f:
            individual_metrics.append(json.load(f))
    
    # Check if metrics exist
    if not individual_metrics:
        print("No individual model metrics found")
        return
    
    # Extract test metrics
    ensemble_test = ensemble_metrics
    individual_test = [m.get('test', m) for m in individual_metrics]
    
    # Find common metrics
    common_metrics = set(ensemble_test.keys())
    for metrics in individual_test:
        common_metrics &= set(metrics.keys())
    
    # Filter metrics (remove 'epoch' and other non-numeric metrics)
    metric_names = []
    for key in common_metrics:
        if isinstance(ensemble_test[key], (int, float)) and key not in ['epoch', 'step']:
            metric_names.append(key)
    
    # Create comparison data
    comparison_data = {}
    for metric in metric_names:
        comparison_data[metric] = {
            'ensemble': ensemble_test[metric],
            'individual': [m[metric] for m in individual_test]
        }
    
    # Plot comparison for each metric
    for metric, data in comparison_data.items():
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot ensemble value
        ax.axhline(y=data['ensemble'], color='r', linestyle='-', linewidth=2,
                  label=f'Ensemble: {data["ensemble"]:.4f}')
        
        # Plot individual values
        x = range(1, len(data['individual']) + 1)
        ax.bar(x, data['individual'], alpha=0.7, label='Individual Models')
        
        # Add value labels
        for i, value in enumerate(data['individual']):
            ax.text(i + 1, value, f'{value:.4f}', ha='center', va='bottom', fontsize=10)
        
        # Add labels and title
        ax.set_xlabel('Model')
        ax.set_ylabel(metric)
        ax.set_title(f'Ensemble vs Individual Models: {metric}')
        
        # Set x-ticks
        ax.set_xticks(x)
        
        # Add legend
        ax.legend()
        
        # Save the figure
        plt.tight_layout()
        metric_path = os.path.join(output_dir, f"{metric}_ensemble_comparison.png")
        plt.savefig(metric_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    # Create overall comparison plot
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Set up metrics and models
    metrics_to_plot = ['rmse', 'mae', 'r2'] if all(m in metric_names for m in ['rmse', 'mae', 'r2']) else metric_names[:3]
    model_labels = ['Ensemble'] + [f'Model {i+1}' for i in range(len(individual_metrics))]
    
    # Set up bar positions
    bar_width = 0.8 / len(metrics_to_plot)
    positions = np.arange(len(model_labels))
    
    # Plot bars for each metric
    for i, metric in enumerate(metrics_to_plot):
        # Get values
        values = [comparison_data[metric]['ensemble']] + comparison_data[metric]['individual']
        
        # Adjust sign for metrics where lower is better
        if metric.lower() in ['rmse', 'mae', 'loss']:
            # For these metrics, lower is better, so invert for visualization
            values = [-v for v in values]
            label = f'{metric.upper()} (lower is better)'
        else:
            label = f'{metric.upper()} (higher is better)'
        
        # Plot bars
        offset = (i - len(metrics_to_plot)/2 + 0.5) * bar_width
        bars = ax.bar(positions + offset, values, bar_width, label=label, alpha=0.7)
        
        # Add value labels
        for bar, value in zip(bars, values):
            actual_value = -value if metric.lower() in ['rmse', 'mae', 'loss'] else value
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{actual_value:.4f}', ha='center', va='bottom', rotation=90, fontsize=9)
    
    # Add labels and title
    ax.set_xlabel('Model')
    ax.set_ylabel('Metric Value (scaled for comparison)')
    ax.set_title('Ensemble vs Individual Models Comparison')
    
    # Set x-ticks
    ax.set_xticks(positions)
    ax.set_xticklabels(model_labels)
    
    # Add legend
    ax.legend()
    
    # Save the figure
    plt.tight_layout()
    save_path = os.path.join(output_dir, save_name)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)