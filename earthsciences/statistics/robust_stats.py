"""
Robust statistical methods 

MAD, trimmed means, M-estimators, and robust regression.
"""

import numpy as np
from scipy import stats
from scipy.optimize import minimize
from typing import Dict, Tuple, Optional, Union


def mad(data: np.ndarray, scale_factor: float = 1.4826) -> float:
    """
    Median Absolute Deviation (MAD).
    
    Parameters
    ----------
    data : array_like
        Input data
    scale_factor : float
        Scale factor for consistency with std (default=1.4826 for normal)
        
    Returns
    -------
    float
        MAD value
    """
    data = np.asarray(data)
    median = np.median(data)
    mad_value = np.median(np.abs(data - median))
    return scale_factor * mad_value


def trimmed_mean(data: np.ndarray, trim_percent: float = 0.1) -> float:
    """
    Trimmed mean (remove extreme values).
    
    Parameters
    ----------
    data : array_like
        Input data
    trim_percent : float
        Fraction to trim from each end (default=0.1 = 10%)
        
    Returns
    -------
    float
        Trimmed mean
    """
    return stats.trim_mean(data, trim_percent)


def winsorized_mean(data: np.ndarray, limits: Tuple[float, float] = (0.1, 0.1)) -> float:
    """
    Winsorized mean (replace extreme values).
    
    Parameters
    ----------
    data : array_like
        Input data
    limits : tuple
        (lower_limit, upper_limit) fractions to winsorize
        
    Returns
    -------
    float
        Winsorized mean
    """
    from scipy.stats.mstats import winsorize
    winsorized_data = winsorize(data, limits=limits)
    return np.mean(winsorized_data)


def huber_location(data: np.ndarray, c: float = 1.5, tol: float = 1e-6) -> float:
    """
    Huber M-estimator for location.
    
    Parameters
    ----------
    data : array_like
        Input data
    c : float
        Tuning constant (default=1.5)
    tol : float
        Convergence tolerance
        
    Returns
    -------
    float
        Huber location estimate
    """
    data = np.asarray(data)
    
    mu = np.median(data)
    sigma = mad(data)
    
    for _ in range(100):
        residuals = (data - mu) / sigma
        weights = np.where(np.abs(residuals) <= c, 1.0, c / np.abs(residuals))
        
        mu_new = np.sum(weights * data) / np.sum(weights)
        
        if np.abs(mu_new - mu) < tol:
            break
        mu = mu_new
    
    return mu


def ransac_regression(x: np.ndarray, y: np.ndarray,
                      max_trials: int = 100,
                      residual_threshold: Optional[float] = None,
                      min_samples: Optional[int] = None) -> Dict:
    """
    RANSAC robust regression.
    
    Parameters
    ----------
    x, y : array_like
        Data points
    max_trials : int
        Maximum number of iterations
    residual_threshold : float, optional
        Threshold for inliers (default=auto)
    min_samples : int, optional
        Minimum samples for fitting (default=2)
        
    Returns
    -------
    dict
        Dictionary with slope, intercept, inliers
    """
    x = np.asarray(x)
    y = np.asarray(y)
    n = len(x)
    
    if min_samples is None:
        min_samples = 2
    
    if residual_threshold is None:
        residual_threshold = 2 * np.std(y)
    
    best_inliers = []
    best_slope = 0
    best_intercept = 0
    
    for _ in range(max_trials):
        sample_idx = np.random.choice(n, min_samples, replace=False)
        x_sample = x[sample_idx]
        y_sample = y[sample_idx]
        
        slope = (y_sample[1] - y_sample[0]) / (x_sample[1] - x_sample[0] + 1e-10)
        intercept = y_sample[0] - slope * x_sample[0]
        
        y_pred = slope * x + intercept
        residuals = np.abs(y - y_pred)
        inliers = residuals < residual_threshold
        
        if np.sum(inliers) > len(best_inliers):
            best_inliers = inliers
            x_inliers = x[inliers]
            y_inliers = y[inliers]
            
            slope_fit = np.cov(x_inliers, y_inliers)[0, 1] / np.var(x_inliers)
            intercept_fit = np.mean(y_inliers) - slope_fit * np.mean(x_inliers)
            
            best_slope = slope_fit
            best_intercept = intercept_fit
    
    return {
        'slope': best_slope,
        'intercept': best_intercept,
        'inliers': best_inliers,
        'n_inliers': np.sum(best_inliers)
    }


def theil_sen_regression(x: np.ndarray, y: np.ndarray) -> Dict:
    """
    Theil-Sen robust regression (median of slopes).
    
    Parameters
    ----------
    x, y : array_like
        Data points
        
    Returns
    -------
    dict
        Dictionary with slope and intercept
    """
    from scipy.stats import theilslopes
    
    x = np.asarray(x)
    y = np.asarray(y)
    
    result = theilslopes(y, x)
    
    return {
        'slope': result[0],
        'intercept': result[1],
        'low_slope': result[2],
        'high_slope': result[3]
    }


def detect_outliers_iqr(data: np.ndarray, k: float = 1.5) -> np.ndarray:
    """
    Detect outliers using IQR method.
    
    Parameters
    ----------
    data : array_like
        Input data
    k : float
        IQR multiplier (default=1.5)
        
    Returns
    -------
    ndarray
        Boolean array of outliers
    """
    data = np.asarray(data)
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    
    lower_bound = q1 - k * iqr
    upper_bound = q3 + k * iqr
    
    outliers = (data < lower_bound) | (data > upper_bound)
    return outliers


def detect_outliers_zscore(data: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    """
    Detect outliers using Z-score method.
    
    Parameters
    ----------
    data : array_like
        Input data
    threshold : float
        Z-score threshold (default=3.0)
        
    Returns
    -------
    ndarray
        Boolean array of outliers
    """
    data = np.asarray(data)
    z_scores = np.abs(stats.zscore(data))
    outliers = z_scores > threshold
    return outliers


def detect_outliers_mad(data: np.ndarray, threshold: float = 3.5) -> np.ndarray:
    """
    Detect outliers using MAD method.
    
    Parameters
    ----------
    data : array_like
        Input data
    threshold : float
        MAD multiplier threshold (default=3.5)
        
    Returns
    -------
    ndarray
        Boolean array of outliers
    """
    data = np.asarray(data)
    median = np.median(data)
    mad_value = mad(data)
    
    modified_z_scores = 0.6745 * np.abs(data - median) / mad_value
    outliers = modified_z_scores > threshold
    return outliers


def biweight_location(data: np.ndarray, c: float = 6.0, tol: float = 1e-6) -> float:
    """
    Biweight location estimator.
    
    Parameters
    ----------
    data : array_like
        Input data
    c : float
        Tuning constant (default=6.0)
    tol : float
        Convergence tolerance
        
    Returns
    -------
    float
        Biweight location estimate
    """
    data = np.asarray(data)
    
    location = np.median(data)
    scale = mad(data)
    
    for _ in range(100):
        u = (data - location) / (c * scale)
        weights = np.where(np.abs(u) < 1, (1 - u**2)**2, 0)
        
        location_new = np.sum(weights * data) / np.sum(weights)
        
        if np.abs(location_new - location) < tol:
            break
        location = location_new
    
    return location


def biweight_scale(data: np.ndarray, c: float = 9.0) -> float:
    """
    Biweight scale estimator.
    
    Parameters
    ----------
    data : array_like
        Input data
    c : float
        Tuning constant (default=9.0)
        
    Returns
    -------
    float
        Biweight scale estimate
    """
    data = np.asarray(data)
    
    median = np.median(data)
    mad_value = mad(data)
    
    u = (data - median) / (c * mad_value)
    
    numerator = np.sum((data - median)**2 * (1 - u**2)**4 * (np.abs(u) < 1))
    denominator = np.sum((1 - u**2) * (1 - 5*u**2) * (np.abs(u) < 1))
    
    n = len(data)
    scale = np.sqrt(n * numerator / np.abs(denominator))
    
    return scale
