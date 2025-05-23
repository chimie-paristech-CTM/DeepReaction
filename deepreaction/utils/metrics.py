import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, List, Union


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                               metrics: List[str] = ['mae', 'mse', 'rmse', 'r2']) -> Dict[str, float]:
    """
    Compute regression metrics between true and predicted values.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        metrics: List of metrics to compute

    Returns:
        Dictionary containing computed metrics
    """
    results = {}

    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    if 'mae' in metrics:
        results['mae'] = mean_absolute_error(y_true_flat, y_pred_flat)

    if 'mse' in metrics:
        results['mse'] = mean_squared_error(y_true_flat, y_pred_flat)

    if 'rmse' in metrics:
        results['rmse'] = np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))

    if 'r2' in metrics:
        try:
            results['r2'] = r2_score(y_true_flat, y_pred_flat)
        except:
            results['r2'] = 0.0

    return results