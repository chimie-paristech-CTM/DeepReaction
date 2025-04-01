"""
Metrics utility module for evaluating model performance.

This module contains classes and functions for computing various regression
and classification metrics to evaluate model performance.
"""

import numpy as np
from typing import Dict, Any, List, Union, Optional, Tuple
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    matthews_corrcoef
)


class RegressionMetrics:
    """
    Class for computing and storing regression metrics.
    """
    
    @staticmethod
    def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute mean squared error.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            
        Returns:
            float: Mean squared error
        """
        return float(mean_squared_error(y_true, y_pred))
    
    @staticmethod
    def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute root mean squared error.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            
        Returns:
            float: Root mean squared error
        """
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))
    
    @staticmethod
    def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute mean absolute error.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            
        Returns:
            float: Mean absolute error
        """
        return float(mean_absolute_error(y_true, y_pred))
    
    @staticmethod
    def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute R^2 score.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            
        Returns:
            float: R^2 score
        """
        return float(r2_score(y_true, y_pred))
    
    @staticmethod
    def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute mean absolute percentage error.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            
        Returns:
            float: Mean absolute percentage error
        """
        # Avoid division by zero
        mask = y_true != 0
        if not np.any(mask):
            return float('nan')
        return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    
    @staticmethod
    def max_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute maximum error.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            
        Returns:
            float: Maximum error
        """
        return float(np.max(np.abs(y_true - y_pred)))
    
    @staticmethod
    def median_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute median absolute error.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            
        Returns:
            float: Median absolute error
        """
        return float(np.median(np.abs(y_true - y_pred)))
    
    @staticmethod
    def explained_variance_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute explained variance score.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            
        Returns:
            float: Explained variance score
        """
        y_diff = y_true - y_pred
        return float(1 - np.var(y_diff) / np.var(y_true)) if np.var(y_true) > 0 else 0.0


class ClassificationMetrics:
    """
    Class for computing and storing classification metrics.
    """
    
    @staticmethod
    def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute accuracy score.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            
        Returns:
            float: Accuracy score
        """
        return float(accuracy_score(y_true, y_pred))
    
    @staticmethod
    def precision(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'macro') -> float:
        """
        Compute precision score.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            average: Averaging method ('micro', 'macro', 'weighted', 'samples')
            
        Returns:
            float: Precision score
        """
        return float(precision_score(y_true, y_pred, average=average, zero_division=0))
    
    @staticmethod
    def recall(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'macro') -> float:
        """
        Compute recall score.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            average: Averaging method ('micro', 'macro', 'weighted', 'samples')
            
        Returns:
            float: Recall score
        """
        return float(recall_score(y_true, y_pred, average=average, zero_division=0))
    
    @staticmethod
    def f1(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'macro') -> float:
        """
        Compute F1 score.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            average: Averaging method ('micro', 'macro', 'weighted', 'samples')
            
        Returns:
            float: F1 score
        """
        return float(f1_score(y_true, y_pred, average=average, zero_division=0))
    
    @staticmethod
    def roc_auc(y_true: np.ndarray, y_score: np.ndarray, average: str = 'macro', multi_class: str = 'ovr') -> float:
        """
        Compute ROC AUC score.
        
        Args:
            y_true: Ground truth labels
            y_score: Predicted probabilities
            average: Averaging method ('micro', 'macro', 'weighted')
            multi_class: Approach for multi-class ROC AUC ('ovr', 'ovo')
            
        Returns:
            float: ROC AUC score
        """
        try:
            if y_score.ndim == 1:
                # Binary classification
                return float(roc_auc_score(y_true, y_score))
            else:
                # Multi-class classification
                return float(roc_auc_score(y_true, y_score, average=average, multi_class=multi_class))
        except ValueError:
            # Return NaN if ROC AUC cannot be computed (e.g., only one class in y_true)
            return float('nan')
    
    @staticmethod
    def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute confusion matrix.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            
        Returns:
            np.ndarray: Confusion matrix
        """
        return confusion_matrix(y_true, y_pred)
    
    @staticmethod
    def matthews_corrcoef(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute Matthews correlation coefficient.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            
        Returns:
            float: Matthews correlation coefficient
        """
        return float(matthews_corrcoef(y_true, y_pred))


def compute_regression_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    metrics: List[str] = None
) -> Dict[str, float]:
    """
    Compute regression metrics.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        metrics: List of metrics to compute
            Options: 'mse', 'rmse', 'mae', 'r2', 'mape', 
                    'max_error', 'median_ae', 'explained_variance'
        
    Returns:
        Dict[str, float]: Dictionary of computed metrics
    """
    # If metrics is None, use default metrics
    if metrics is None:
        metrics = ['mse', 'rmse', 'mae', 'r2']
    
    # Ensure inputs are numpy arrays
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    # Initialize results dictionary
    results = {}
    
    # Compute requested metrics
    for metric in metrics:
        if metric == 'mse':
            results['mse'] = RegressionMetrics.mean_squared_error(y_true, y_pred)
        elif metric == 'rmse':
            results['rmse'] = RegressionMetrics.root_mean_squared_error(y_true, y_pred)
        elif metric == 'mae':
            results['mae'] = RegressionMetrics.mean_absolute_error(y_true, y_pred)
        elif metric == 'r2':
            results['r2'] = RegressionMetrics.r2_score(y_true, y_pred)
        elif metric == 'mape':
            results['mape'] = RegressionMetrics.mean_absolute_percentage_error(y_true, y_pred)
        elif metric == 'max_error':
            results['max_error'] = RegressionMetrics.max_error(y_true, y_pred)
        elif metric == 'median_ae':
            results['median_ae'] = RegressionMetrics.median_absolute_error(y_true, y_pred)
        elif metric == 'explained_variance':
            results['explained_variance'] = RegressionMetrics.explained_variance_score(y_true, y_pred)
    
    return results


def compute_classification_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    y_score: Optional[np.ndarray] = None,
    metrics: List[str] = None,
    average: str = 'macro'
) -> Dict[str, float]:
    """
    Compute classification metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_score: Predicted probabilities
        metrics: List of metrics to compute
            Options: 'accuracy', 'precision', 'recall', 'f1', 
                    'roc_auc', 'matthews_corrcoef'
        average: Averaging method for multi-class metrics
        
    Returns:
        Dict[str, float]: Dictionary of computed metrics
    """
    # If metrics is None, use default metrics
    if metrics is None:
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        if y_score is not None:
            metrics.append('roc_auc')
    
    # Ensure inputs are numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_score is not None:
        y_score = np.asarray(y_score)
    
    # Initialize results dictionary
    results = {}
    
    # Compute requested metrics
    for metric in metrics:
        if metric == 'accuracy':
            results['accuracy'] = ClassificationMetrics.accuracy(y_true, y_pred)
        elif metric == 'precision':
            results['precision'] = ClassificationMetrics.precision(y_true, y_pred, average=average)
        elif metric == 'recall':
            results['recall'] = ClassificationMetrics.recall(y_true, y_pred, average=average)
        elif metric == 'f1':
            results['f1'] = ClassificationMetrics.f1(y_true, y_pred, average=average)
        elif metric == 'roc_auc' and y_score is not None:
            results['roc_auc'] = ClassificationMetrics.roc_auc(y_true, y_score, average=average)
        elif metric == 'matthews_corrcoef':
            results['matthews_corrcoef'] = ClassificationMetrics.matthews_corrcoef(y_true, y_pred)
    
    return results


def compute_metrics_by_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresholds: List[float] = None
) -> Dict[float, Dict[str, float]]:
    """
    Compute classification metrics at different thresholds.
    
    Args:
        y_true: Ground truth binary labels
        y_score: Predicted probabilities
        thresholds: List of thresholds to evaluate
            
    Returns:
        Dict[float, Dict[str, float]]: Dictionary of metrics at each threshold
    """
    # If thresholds is None, use default thresholds
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 9)
    
    # Ensure inputs are numpy arrays
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    
    # Initialize results dictionary
    results = {}
    
    # Compute metrics at each threshold
    for threshold in thresholds:
        # Convert scores to binary predictions
        y_pred = (y_score >= threshold).astype(int)
        
        # Compute metrics
        metrics = compute_classification_metrics(
            y_true=y_true,
            y_pred=y_pred,
            metrics=['accuracy', 'precision', 'recall', 'f1', 'matthews_corrcoef']
        )
        
        # Store metrics for this threshold
        results[float(threshold)] = metrics
    
    return results


def confidence_interval(
    values: np.ndarray,
    confidence: float = 0.95
) -> Tuple[float, float, float]:
    """
    Calculate confidence interval for a sample of values.
    
    Args:
        values: Sample values
        confidence: Confidence level (0-1)
        
    Returns:
        Tuple[float, float, float]: (mean, lower_bound, upper_bound)
    """
    import scipy.stats as stats
    
    values = np.asarray(values)
    mean = np.mean(values)
    se = stats.sem(values)
    
    # Calculate confidence interval
    h = se * stats.t.ppf((1 + confidence) / 2, len(values) - 1)
    
    return mean, mean - h, mean + h


def bootstrap_metric(
    metric_func: callable,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_state: Optional[int] = None
) -> Tuple[float, float, float]:
    """
    Calculate bootstrap confidence interval for a metric.
    
    Args:
        metric_func: Metric function (y_true, y_pred) -> float
        y_true: Ground truth values
        y_pred: Predicted values
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (0-1)
        random_state: Random state for reproducibility
        
    Returns:
        Tuple[float, float, float]: (metric_value, lower_bound, upper_bound)
    """
    # Set random state
    np.random.seed(random_state)
    
    # Compute metric on original data
    metric_value = metric_func(y_true, y_pred)
    
    # Bootstrap samples
    bootstrap_values = []
    n_samples = len(y_true)
    
    for _ in range(n_bootstrap):
        # Generate bootstrap sample indices
        indices = np.random.randint(0, n_samples, n_samples)
        
        # Compute metric on bootstrap sample
        bootstrap_values.append(
            metric_func(y_true[indices], y_pred[indices])
        )
    
    # Calculate confidence interval
    lower_percentile = (1 - confidence) / 2 * 100
    upper_percentile = (1 + confidence) / 2 * 100
    
    lower_bound = np.percentile(bootstrap_values, lower_percentile)
    upper_bound = np.percentile(bootstrap_values, upper_percentile)
    
    return metric_value, lower_bound, upper_bound
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
from typing import Dict, List, Union, Optional, Tuple, Any

def compute_regression_metrics(y_true, y_pred, metrics=None):
    """
    Compute regression metrics between true and predicted values.

    Args:
        y_true: True values (numpy array)
        y_pred: Predicted values (numpy array)
        metrics: List of metrics to compute. Defaults to ['mae', 'rmse', 'r2']

    Returns:
        Dictionary with computed metrics
    """
    if metrics is None:
        metrics = ['mae', 'rmse', 'r2']

    results = {}

    # Convert to numpy if tensors
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Ensure arrays are flattened if needed
    if len(y_true.shape) > 1 and y_true.shape[1] == 1:
        y_true = y_true.flatten()
    if len(y_pred.shape) > 1 and y_pred.shape[1] == 1:
        y_pred = y_pred.flatten()

    # Check shapes
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shapes of y_true {y_true.shape} and y_pred {y_pred.shape} don't match")

    # Compute metrics
    if 'mae' in metrics:
        results['mae'] = float(mean_absolute_error(y_true, y_pred))

    if 'rmse' in metrics:
        results['rmse'] = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    if 'mse' in metrics:
        results['mse'] = float(mean_squared_error(y_true, y_pred))

    if 'r2' in metrics:
        results['r2'] = float(r2_score(y_true, y_pred))

    return results

def compute_multi_target_metrics(y_true, y_pred, metrics=None):
    """
    Compute metrics for multi-target regression.

    Args:
        y_true: True values with shape [n_samples, n_targets]
        y_pred: Predicted values with shape [n_samples, n_targets]
        metrics: List of metrics to compute. Defaults to ['mae', 'rmse', 'r2']

    Returns:
        List of dictionaries with metrics for each target and an 'avg' key with average metrics
    """
    if metrics is None:
        metrics = ['mae', 'rmse', 'r2']

    # Convert to numpy if tensors
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Check shapes
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shapes of y_true {y_true.shape} and y_pred {y_pred.shape} don't match")

    if len(y_true.shape) < 2:
        # Single target case
        return {'target_0': compute_regression_metrics(y_true, y_pred, metrics), 'avg': compute_regression_metrics(y_true, y_pred, metrics)}

    num_targets = y_true.shape[1]

    all_metrics = {}
    avg_metrics = {metric: 0.0 for metric in metrics}

    # Compute metrics for each target
    for i in range(num_targets):
        target_metrics = compute_regression_metrics(y_true[:, i], y_pred[:, i], metrics)
        all_metrics[f'target_{i}'] = target_metrics

        # Accumulate for average
        for metric in metrics:
            avg_metrics[metric] += target_metrics[metric]

    # Compute averages
    for metric in metrics:
        avg_metrics[metric] /= num_targets

    all_metrics['avg'] = avg_metrics

    return all_metrics

def compute_ensemble_metrics(predictions_list, y_true, metrics=None):
    """
    Compute metrics for ensemble predictions.

    Args:
        predictions_list: List of prediction arrays, each with shape [n_samples, n_targets]
        y_true: True values with shape [n_samples, n_targets]
        metrics: List of metrics to compute. Defaults to ['mae', 'rmse', 'r2']

    Returns:
        Dictionary with metrics for ensemble predictions and individual model metrics
    """
    if metrics is None:
        metrics = ['mae', 'rmse', 'r2']

    # Convert to numpy if tensors
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()

    predictions_np = []
    for pred in predictions_list:
        if isinstance(pred, torch.Tensor):
            predictions_np.append(pred.detach().cpu().numpy())
        else:
            predictions_np.append(pred)

    # Compute ensemble predictions (average)
    ensemble_pred = np.mean(predictions_np, axis=0)

    # Compute ensemble metrics
    ensemble_metrics = compute_multi_target_metrics(y_true, ensemble_pred, metrics)

    # Compute individual model metrics
    individual_metrics = []
    for i, pred in enumerate(predictions_np):
        model_metrics = compute_multi_target_metrics(y_true, pred, metrics)
        individual_metrics.append(model_metrics)

    return {
        'ensemble': ensemble_metrics,
        'individual': individual_metrics
    }

def compute_uncertainty_metrics(predictions, prediction_variances, y_true, metrics=None):
    """
    Compute metrics for uncertainty-aware predictions.

    Args:
        predictions: Mean predictions with shape [n_samples, n_targets]
        prediction_variances: Prediction variances with shape [n_samples, n_targets]
        y_true: True values with shape [n_samples, n_targets]
        metrics: List of metrics to compute. Defaults to ['mae', 'rmse', 'r2', 'nll']

    Returns:
        Dictionary with uncertainty-aware metrics
    """
    if metrics is None:
        metrics = ['mae', 'rmse', 'r2', 'nll']

    # Convert to numpy if tensors
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(prediction_variances, torch.Tensor):
        prediction_variances = prediction_variances.detach().cpu().numpy()

    # Ensure prediction_variances are positive
    prediction_variances = np.maximum(prediction_variances, 1e-8)

    # Compute standard metrics
    standard_metrics = compute_multi_target_metrics(y_true, predictions, ['mae', 'rmse', 'r2'])

    # Compute negative log likelihood (NLL) if requested
    if 'nll' in metrics:
        nll = compute_gaussian_nll(y_true, predictions, prediction_variances)

        for target_key in standard_metrics:
            if target_key != 'avg':
                standard_metrics[target_key]['nll'] = nll[int(target_key.split('_')[1])]

        standard_metrics['avg']['nll'] = np.mean(list(nll))

    return standard_metrics

def compute_gaussian_nll(y_true, means, variances):
    """
    Compute Gaussian negative log likelihood.

    Args:
        y_true: True values with shape [n_samples, n_targets]
        means: Predicted means with shape [n_samples, n_targets]
        variances: Predicted variances with shape [n_samples, n_targets]

    Returns:
        Array of NLL values for each target
    """
    n_targets = y_true.shape[1]
    nll_values = np.zeros(n_targets)

    for i in range(n_targets):
        # Ensure variance is positive
        target_var = np.maximum(variances[:, i], 1e-8)

        # Compute NLL for this target
        target_nll = 0.5 * np.log(2 * np.pi * target_var) + 0.5 * ((y_true[:, i] - means[:, i])**2) / target_var
        nll_values[i] = np.mean(target_nll)

    return nll_values