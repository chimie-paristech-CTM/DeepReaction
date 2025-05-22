import numpy as np
from typing import Dict, List, Union, Optional
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error

def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Compute regression metrics for model evaluation.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        metrics: List of metrics to compute. Defaults to 
                ['mae', 'rmse', 'r2', 'mpae', 'max_ae', 'median_ae']
    
    Returns:
        Dictionary containing all computed metrics
    """
    if metrics is None:
        metrics = ['mae', 'rmse', 'r2', 'mpae', 'max_ae', 'median_ae']
    
    # Ensure numpy arrays
    if not isinstance(y_true, np.ndarray):
        y_true = np.array(y_true)
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.array(y_pred)
    
    # Reshape if needed
    if len(y_true.shape) == 1:
        y_true = y_true.reshape(-1, 1)
    if len(y_pred.shape) == 1:
        y_pred = y_pred.reshape(-1, 1)
    
    # Initialize results dictionary
    results = {}
    
    # Compute metrics
    if 'mae' in metrics:
        results['mae'] = float(mean_absolute_error(y_true, y_pred))
    
    if 'rmse' in metrics:
        results['rmse'] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    
    if 'r2' in metrics:
        results['r2'] = float(r2_score(y_true, y_pred))
    
    if 'mpae' in metrics:
        # Mean percentage absolute error
        with np.errstate(divide='ignore', invalid='ignore'):
            mpae = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            if np.isnan(mpae) or np.isinf(mpae):
                results['mpae'] = float('nan')
            else:
                results['mpae'] = float(mpae)
    
    if 'max_ae' in metrics:
        # Maximum absolute error
        results['max_ae'] = float(np.max(np.abs(y_true - y_pred)))
    
    if 'median_ae' in metrics:
        # Median absolute error
        results['median_ae'] = float(median_absolute_error(y_true, y_pred))
    
    return results