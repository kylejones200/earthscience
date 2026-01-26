"""
Extreme value analysis

Statistical analysis of extremes, return periods, and risk assessment.
"""

import numpy as np
from scipy import stats
from scipy.optimize import minimize, OptimizeWarning
from typing import Dict, Tuple, Optional
import warnings


def fit_gev(data: np.ndarray,
           method: str = 'mle') -> Dict:
    """
    Fit Generalized Extreme Value (GEV) distribution.
    
    Used for modeling extreme events (floods, earthquakes, etc.).
    
    Parameters
    ----------
    data : array_like
        Extreme values (e.g., annual maxima)
    method : str, optional
        Fitting method: 'mle' (Maximum Likelihood) (default='mle')
        
    Returns
    -------
    dict
        Fitted GEV parameters and statistics
        
    Examples
    --------
    >>> # Annual maximum flood levels
    >>> flood_maxima = np.array([5.2, 6.1, 5.8, 7.5, 6.0, 8.2, 5.5, 6.8])
    >>> fit = fit_gev(flood_maxima)
    >>> print(f"Shape (ξ): {fit['shape']:.3f}")
    >>> print(f"Location (μ): {fit['location']:.3f}")
    >>> print(f"Scale (σ): {fit['scale']:.3f}")
    """
    data = np.asarray(data)
    data = data[~np.isnan(data)]
    
    # Fit GEV distribution
    # shape (c), location (loc), scale (scale)
    shape, location, scale = stats.genextreme.fit(data)
    
    # Log-likelihood
    log_likelihood = np.sum(stats.genextreme.logpdf(data, shape, location, scale))
    
    # AIC and BIC
    n = len(data)
    k = 3  # Number of parameters
    aic = 2 * k - 2 * log_likelihood
    bic = k * np.log(n) - 2 * log_likelihood
    
    return {
        'shape': shape,
        'location': location,
        'scale': scale,
        'log_likelihood': log_likelihood,
        'aic': aic,
        'bic': bic,
        'distribution': stats.genextreme(shape, location, scale),
    }


def return_level(gev_params: Dict,
                return_period: float) -> float:
    """
    Calculate return level for a given return period.
    
    Return level is the value expected to be exceeded on average
    once every T years.
    
    Parameters
    ----------
    gev_params : dict
        GEV parameters from fit_gev()
    return_period : float
        Return period in years (e.g., 100 for 100-year event)
        
    Returns
    -------
    float
        Return level
        
    Examples
    --------
    >>> flood_maxima = np.array([5.2, 6.1, 5.8, 7.5, 6.0, 8.2, 5.5, 6.8])
    >>> fit = fit_gev(flood_maxima)
    >>> level_100 = return_level(fit, 100)
    >>> print(f"100-year flood level: {level_100:.2f} m")
    """
    shape = gev_params['shape']
    location = gev_params['location']
    scale = gev_params['scale']
    
    # Calculate return level
    p = 1 - 1/return_period
    
    if np.abs(shape) > 1e-6:
        # Non-zero shape parameter
        z_p = location - (scale / shape) * (1 - (-np.log(p))**(-shape))
    else:
        # Gumbel distribution (shape ≈ 0)
        z_p = location - scale * np.log(-np.log(p))
    
    return z_p


def return_period(gev_params: Dict,
                 level: float) -> float:
    """
    Calculate return period for a given level.
    
    Parameters
    ----------
    gev_params : dict
        GEV parameters
    level : float
        Event magnitude
        
    Returns
    -------
    float
        Return period in years
        
    Examples
    --------
    >>> fit = fit_gev(flood_maxima)
    >>> period = return_period(fit, 9.0)
    >>> print(f"9m flood has return period of {period:.1f} years")
    """
    shape = gev_params['shape']
    location = gev_params['location']
    scale = gev_params['scale']
    
    # Calculate exceedance probability
    if np.abs(shape) > 1e-6:
        # Non-zero shape
        y = (level - location) / scale
        p = np.exp(-np.power(1 + shape * y, -1/shape))
    else:
        # Gumbel
        y = (level - location) / scale
        p = np.exp(-np.exp(-y))
    
    # Return period
    T = 1 / (1 - p)
    
    return T


def fit_gpd(data: np.ndarray,
           threshold: Optional[float] = None) -> Dict:
    """
    Fit Generalized Pareto Distribution (GPD) for peaks-over-threshold.
    
    Parameters
    ----------
    data : array_like
        Data (all values, not just extremes)
    threshold : float, optional
        Threshold for peak selection (if None, use 95th percentile)
        
    Returns
    -------
    dict
        GPD parameters and exceedances
        
    Examples
    --------
    >>> # Daily rainfall data
    >>> rainfall = np.random.exponential(10, 1000)
    >>> fit = fit_gpd(rainfall)
    >>> print(f"Shape: {fit['shape']:.3f}, Scale: {fit['scale']:.3f}")
    """
    data = np.asarray(data)
    data = data[~np.isnan(data)]
    
    if threshold is None:
        threshold = np.percentile(data, 95)
    
    # Extract exceedances
    exceedances = data[data > threshold] - threshold
    
    if len(exceedances) < 10:
        raise ValueError("Too few exceedances. Lower the threshold.")
    
    # Fit GPD
    shape, loc, scale = stats.genpareto.fit(exceedances, floc=0)
    
    return {
        'shape': shape,
        'scale': scale,
        'threshold': threshold,
        'n_exceedances': len(exceedances),
        'exceedance_rate': len(exceedances) / len(data),
        'exceedances': exceedances,
        'distribution': stats.genpareto(shape, loc=0, scale=scale),
    }


def mean_excess_plot(data: np.ndarray,
                    thresholds: Optional[np.ndarray] = None) -> Dict:
    """
    Mean excess plot for threshold selection in GPD.
    
    For GPD data, the mean excess plot should be approximately linear.
    
    Parameters
    ----------
    data : array_like
        Input data
    thresholds : array_like, optional
        Thresholds to test
        
    Returns
    -------
    dict
        Data for mean excess plot
        
    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> data = np.random.exponential(10, 1000)
    >>> mep = mean_excess_plot(data)
    >>> plt.plot(mep['thresholds'], mep['mean_excesses'])
    >>> plt.xlabel('Threshold')
    >>> plt.ylabel('Mean Excess')
    >>> plt.show()
    """
    data = np.asarray(data)
    data = data[~np.isnan(data)]
    
    if thresholds is None:
        # Use quantiles
        thresholds = np.percentile(data, np.linspace(50, 99, 20))
    
    mean_excesses = []
    n_exceedances = []
    
    for threshold in thresholds:
        exceedances = data[data > threshold] - threshold
        if len(exceedances) > 0:
            mean_excesses.append(np.mean(exceedances))
            n_exceedances.append(len(exceedances))
        else:
            mean_excesses.append(np.nan)
            n_exceedances.append(0)
    
    return {
        'thresholds': thresholds,
        'mean_excesses': np.array(mean_excesses),
        'n_exceedances': np.array(n_exceedances),
    }


def block_maxima(data: np.ndarray,
                block_size: int,
                method: str = 'max') -> np.ndarray:
    """
    Extract block maxima (or minima) from data.
    
    Parameters
    ----------
    data : array_like
        Time series data
    block_size : int
        Block size (e.g., 365 for annual maxima from daily data)
    method : str, optional
        'max' or 'min' (default='max')
        
    Returns
    -------
    ndarray
        Block extremes
        
    Examples
    --------
    >>> # Daily temperature data
    >>> daily_temps = np.random.randn(3650) + 20  # 10 years
    >>> annual_maxima = block_maxima(daily_temps, block_size=365)
    >>> print(f"Annual maxima: {annual_maxima}")
    """
    data = np.asarray(data)
    
    n_blocks = len(data) // block_size
    blocks = data[:n_blocks * block_size].reshape(n_blocks, block_size)
    
    if method == 'max':
        return np.max(blocks, axis=1)
    elif method == 'min':
        return np.min(blocks, axis=1)
    else:
        raise ValueError(f"Unknown method: {method}")


def l_moments(data: np.ndarray) -> Dict:
    """
    Calculate L-moments for distribution fitting.
    
    L-moments are more robust than ordinary moments for extreme value analysis.
    
    Parameters
    ----------
    data : array_like
        Input data
        
    Returns
    -------
    dict
        L-moments (λ1, λ2, τ3, τ4)
        
    Examples
    --------
    >>> data = np.random.gumbel(0, 1, 100)
    >>> lm = l_moments(data)
    >>> print(f"L-location: {lm['lambda1']:.3f}")
    >>> print(f"L-scale: {lm['lambda2']:.3f}")
    >>> print(f"L-skewness: {lm['tau3']:.3f}")
    """
    data = np.asarray(data)
    data = np.sort(data[~np.isnan(data)])
    n = len(data)
    
    # Sample L-moments
    b0 = np.mean(data)
    
    # Calculate probability weighted moments
    j = np.arange(n)
    b1 = np.sum((j / (n - 1)) * data) / n
    b2 = np.sum((j * (j - 1) / ((n - 1) * (n - 2))) * data) / n
    
    # L-moments
    lambda1 = b0
    lambda2 = 2 * b1 - b0
    lambda3 = 6 * b2 - 6 * b1 + b0
    
    # L-moment ratios
    tau2 = lambda2 / lambda1 if lambda1 != 0 else 0
    tau3 = lambda3 / lambda2 if lambda2 != 0 else 0
    
    return {
        'lambda1': lambda1,  # L-location
        'lambda2': lambda2,  # L-scale
        'lambda3': lambda3,
        'tau2': tau2,        # L-CV
        'tau3': tau3,        # L-skewness
    }


def confidence_intervals_return_level(gev_params: Dict,
                                     return_period: float,
                                     data: np.ndarray,
                                     confidence: float = 0.95,
                                     n_bootstrap: int = 1000) -> Dict:
    """
    Bootstrap confidence intervals for return levels.
    
    Parameters
    ----------
    gev_params : dict
        GEV parameters
    return_period : float
        Return period
    data : array_like
        Original data for bootstrapping
    confidence : float, optional
        Confidence level (default=0.95)
    n_bootstrap : int, optional
        Number of bootstrap samples (default=1000)
        
    Returns
    -------
    dict
        Return level with confidence intervals
        
    Examples
    --------
    >>> flood_maxima = np.array([5.2, 6.1, 5.8, 7.5, 6.0, 8.2, 5.5, 6.8])
    >>> fit = fit_gev(flood_maxima)
    >>> ci = confidence_intervals_return_level(fit, 100, flood_maxima)
    >>> print(f"100-year level: {ci['return_level']:.2f}")
    >>> print(f"95% CI: [{ci['lower']:.2f}, {ci['upper']:.2f}]")
    """
    data = np.asarray(data)
    n = len(data)
    
    # Point estimate
    rl = return_level(gev_params, return_period)
    
    # Bootstrap
    bootstrap_rls = []
    for _ in range(n_bootstrap):
        # Resample
        sample = np.random.choice(data, size=n, replace=True)
        
        try:
            # Fit GEV
            sample_fit = fit_gev(sample)
            # Calculate return level
            sample_rl = return_level(sample_fit, return_period)
            bootstrap_rls.append(sample_rl)
        except (RuntimeError, ValueError, OptimizeWarning, Exception) as e:
            # Skip if fitting fails (can happen with small or poor bootstrap samples)
            warnings.warn(
                f"GEV fitting failed for bootstrap sample {i}: {e}. Skipping this sample.",
                RuntimeWarning
            )
            continue
    
    bootstrap_rls = np.array(bootstrap_rls)
    
    # Confidence intervals
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_rls, alpha/2 * 100)
    upper = np.percentile(bootstrap_rls, (1 - alpha/2) * 100)
    
    return {
        'return_level': rl,
        'lower': lower,
        'upper': upper,
        'confidence': confidence,
        'bootstrap_samples': bootstrap_rls,
    }


def gev_return_level(params: Dict, return_period: float) -> float:
    """
    Calculate return level from GEV parameters.
    
    Parameters
    ----------
    params : dict
        GEV parameters (location, scale, shape)
    return_period : float
        Return period in years
        
    Returns
    -------
    float
        Return level
    """
    location = params['location']
    scale = params['scale']
    shape = params['shape']
    
    p = 1.0 / return_period
    
    if abs(shape) < 1e-10:
        return_level = location - scale * np.log(-np.log(1 - p))
    else:
        return_level = location + (scale / shape) * ((-np.log(1 - p))**(-shape) - 1)
    
    return return_level


def select_threshold(data: np.ndarray, method: str = 'percentile', percentile: float = 90) -> float:
    """
    Select threshold for POT analysis.
    
    Parameters
    ----------
    data : array_like
        Data
    method : str
        'percentile' or 'mean_excess'
    percentile : float
        Percentile for threshold (default=90)
        
    Returns
    -------
    float
        Threshold value
    """
    data = np.asarray(data)
    
    if method == 'percentile':
        return np.percentile(data, percentile)
    elif method == 'mean_excess':
        thresholds = np.percentile(data, np.linspace(80, 95, 10))
        best_threshold = thresholds[len(thresholds)//2]
        return best_threshold
    else:
        raise ValueError(f"Unknown method: {method}")


def annual_maxima(data: np.ndarray, observations_per_year: int = 365) -> np.ndarray:
    """
    Extract annual maxima from time series.
    
    Parameters
    ----------
    data : array_like
        Time series data
    observations_per_year : int
        Number of observations per year
        
    Returns
    -------
    ndarray
        Annual maxima
    """
    data = np.asarray(data)
    n_years = len(data) // observations_per_year
    
    maxima = []
    for i in range(n_years):
        year_data = data[i*observations_per_year:(i+1)*observations_per_year]
        maxima.append(np.max(year_data))
    
    return np.array(maxima)


def empirical_return_period(data: np.ndarray) -> Dict:
    """
    Calculate empirical return periods.
    
    Parameters
    ----------
    data : array_like
        Extreme values
        
    Returns
    -------
    dict
        Dictionary with values and return_periods
    """
    data = np.asarray(data)
    sorted_data = np.sort(data)[::-1]
    n = len(data)
    
    ranks = np.arange(1, n + 1)
    return_periods = (n + 1) / ranks
    
    return {
        'values': sorted_data,
        'return_periods': return_periods
    }


def pot_analysis(data: np.ndarray, threshold: float) -> Dict:
    """
    Peaks Over Threshold analysis.
    
    Parameters
    ----------
    data : array_like
        Data
    threshold : float
        Threshold value
        
    Returns
    -------
    dict
        Dictionary with exceedances and count
    """
    data = np.asarray(data)
    exceedances = data[data > threshold] - threshold
    
    return {
        'exceedances': exceedances,
        'n_exceedances': len(exceedances),
        'count': len(exceedances),
        'threshold': threshold
    }


def hill_estimator(data: np.ndarray, k: int) -> float:
    """
    Hill estimator for tail index.
    
    Parameters
    ----------
    data : array_like
        Data
    k : int
        Number of upper order statistics
        
    Returns
    -------
    float
        Tail index estimate
    """
    data = np.asarray(data)
    sorted_data = np.sort(data)[::-1]
    
    log_ratios = np.log(sorted_data[:k]) - np.log(sorted_data[k])
    xi = np.mean(log_ratios)
    
    return xi


def return_level_plot(data: np.ndarray, method: str = 'gev') -> Dict:
    """
    Generate return level plot data.
    
    Parameters
    ----------
    data : array_like
        Extreme values
    method : str
        'gev' or 'empirical'
        
    Returns
    -------
    dict
        Dictionary with return_periods and return_levels
    """
    data = np.asarray(data)
    
    if method == 'gev':
        params = fit_gev(data)
        return_periods = np.logspace(0, 2, 50)
        return_levels = [gev_return_level(params, rp) for rp in return_periods]
    else:
        result = empirical_return_period(data)
        return_periods = result['return_periods']
        return_levels = result['values']
    
    return {
        'return_periods': return_periods,
        'return_levels': return_levels
    }


def bootstrap_return_level(data: np.ndarray,
                           return_period: float,
                           n_bootstrap: int = 1000,
                           confidence: float = 0.95) -> Dict:
    """
    Bootstrap confidence intervals for return level.
    
    Parameters
    ----------
    data : array_like
        Extreme values
    return_period : float
        Return period
    n_bootstrap : int
        Number of bootstrap samples
    confidence : float
        Confidence level
        
    Returns
    -------
    dict
        Dictionary with return_level and confidence intervals
    """
    data = np.asarray(data)
    n = len(data)
    
    params = fit_gev(data)
    rl = gev_return_level(params, return_period)
    
    bootstrap_rls = []
    for i in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        try:
            params_boot = fit_gev(sample)
            rl_boot = gev_return_level(params_boot, return_period)
            bootstrap_rls.append(rl_boot)
        except (RuntimeError, ValueError) as e:
            continue
    
    bootstrap_rls = np.array(bootstrap_rls)
    
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_rls, alpha/2 * 100)
    upper = np.percentile(bootstrap_rls, (1 - alpha/2) * 100)
    
    return {
        'return_level': rl,
        'lower': lower,
        'upper': upper,
        'confidence': confidence,
        'bootstrap_samples': bootstrap_rls,
    }
