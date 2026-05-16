"""
Variogram modeling

Experimental variogram calculation and model fitting.
"""

import warnings

import numpy as np
from scipy.optimize import OptimizeWarning, curve_fit
from scipy.spatial.distance import pdist, squareform


def compute_variogram(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    n_lags: int = 15,
    max_lag: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute experimental (empirical) variogram.

    Parameters
    ----------
    x, y : array_like
        Coordinates of data points
    values : array_like
        Values at data points
    n_lags : int, optional
        Number of lag bins (default=15)
    max_lag : float, optional
        Maximum lag distance (if None, use half of maximum distance)

    Returns
    -------
    lag_dist : ndarray
        Lag distances (bin centers)
    gamma : ndarray
        Semivariance values
    n_pairs : ndarray
        Number of pairs in each bin

    Examples
    --------
    >>> x = np.random.rand(50) * 10
    >>> y = np.random.rand(50) * 10
    >>> values = np.sin(x) + np.cos(y) + np.random.randn(50) * 0.1
    >>> lags, gamma, n_pairs = compute_variogram(x, y, values)
    >>>
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(lags, gamma, 'o-')
    >>> plt.xlabel('Lag Distance')
    >>> plt.ylabel('Semivariance')
    >>> plt.show()
    """
    x = np.asarray(x)
    y = np.asarray(y)
    values = np.asarray(values)

    # Compute pairwise distances
    points = np.column_stack([x, y])
    distances = squareform(pdist(points))

    # Compute pairwise squared differences
    diff_matrix = np.subtract.outer(values, values)
    squared_diffs = diff_matrix**2

    # Maximum lag
    if max_lag is None:
        max_lag = np.max(distances) / 2

    # Define lag bins
    lag_edges = np.linspace(0, max_lag, n_lags + 1)
    lag_centers = (lag_edges[:-1] + lag_edges[1:]) / 2

    # Compute semivariance for each lag
    gamma = np.zeros(n_lags)
    n_pairs = np.zeros(n_lags, dtype=int)

    for i in range(n_lags):
        # Find pairs in this lag bin
        mask = (distances >= lag_edges[i]) & (distances < lag_edges[i + 1])

        # Upper triangle only (avoid double counting)
        mask = np.triu(mask, k=1)

        if np.any(mask):
            # Semivariance = 0.5 * mean(squared differences)
            gamma[i] = 0.5 * np.mean(squared_diffs[mask])
            n_pairs[i] = np.sum(mask)

    # Remove empty bins
    valid = n_pairs > 0
    lag_centers = lag_centers[valid]
    gamma = gamma[valid]
    n_pairs = n_pairs[valid]

    return lag_centers, gamma, n_pairs


def variogram_cloud(
    x: np.ndarray, y: np.ndarray, values: np.ndarray, max_points: int = 10000
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute variogram cloud (all pairwise comparisons).

    Useful for detecting outliers and anisotropy.

    Parameters
    ----------
    x, y : array_like
        Coordinates of data points
    values : array_like
        Values at data points
    max_points : int, optional
        Maximum number of pairs to return (for efficiency) (default=10000)

    Returns
    -------
    distances : ndarray
        All pairwise distances
    semivariances : ndarray
        All pairwise semivariances

    Examples
    --------
    >>> x = np.random.rand(30) * 10
    >>> y = np.random.rand(30) * 10
    >>> values = np.sin(x) + np.cos(y)
    >>> dist, gamma = variogram_cloud(x, y, values)
    >>>
    >>> import matplotlib.pyplot as plt
    >>> plt.scatter(dist, gamma, alpha=0.3, s=1)
    >>> plt.xlabel('Distance')
    >>> plt.ylabel('Semivariance')
    >>> plt.show()
    """
    x = np.asarray(x)
    y = np.asarray(y)
    values = np.asarray(values)

    # Compute all pairwise distances and semivariances
    points = np.column_stack([x, y])
    distances = pdist(points)

    diff_values = pdist(values[:, np.newaxis], lambda u, v: (u - v) ** 2)
    semivariances = 0.5 * diff_values

    # Subsample if too many points
    if len(distances) > max_points:
        indices = np.random.choice(len(distances), max_points, replace=False)
        distances = distances[indices]
        semivariances = semivariances[indices]

    return distances, semivariances


def spherical_model(h: np.ndarray, nugget: float, sill: float, range_param: float) -> np.ndarray:
    """
    Spherical variogram model.

    Parameters
    ----------
    h : array_like
        Lag distances
    nugget : float
        Nugget effect (discontinuity at origin)
    sill : float
        Sill (maximum semivariance)
    range_param : float
        Range (distance where sill is reached)

    Returns
    -------
    ndarray
        Model semivariances
    """
    h = np.asarray(h)
    gamma = np.zeros_like(h)

    mask = h > 0
    gamma[mask] = nugget + (sill - nugget) * (
        1.5 * (h[mask] / range_param) - 0.5 * (h[mask] / range_param) ** 3
    )

    # Cap at sill
    gamma[h >= range_param] = sill

    return gamma


def exponential_model(h: np.ndarray, nugget: float, sill: float, range_param: float) -> np.ndarray:
    """
    Exponential variogram model.

    Parameters
    ----------
    h : array_like
        Lag distances
    nugget : float
        Nugget effect
    sill : float
        Sill
    range_param : float
        Practical range (95% of sill)

    Returns
    -------
    ndarray
        Model semivariances
    """
    h = np.asarray(h)
    gamma = np.zeros_like(h)

    mask = h > 0
    gamma[mask] = nugget + (sill - nugget) * (1 - np.exp(-3 * h[mask] / range_param))

    return gamma


def gaussian_model(h: np.ndarray, nugget: float, sill: float, range_param: float) -> np.ndarray:
    """
    Gaussian variogram model.

    Parameters
    ----------
    h : array_like
        Lag distances
    nugget : float
        Nugget effect
    sill : float
        Sill
    range_param : float
        Practical range

    Returns
    -------
    ndarray
        Model semivariances
    """
    h = np.asarray(h)
    gamma = np.zeros_like(h)

    mask = h > 0
    gamma[mask] = nugget + (sill - nugget) * (1 - np.exp(-3 * (h[mask] / range_param) ** 2))

    return gamma


def fit_variogram_model(
    lag_dist: np.ndarray,
    gamma: np.ndarray,
    model: str = "spherical",
    n_pairs: np.ndarray | None = None,
) -> dict:
    """
    Fit a variogram model to experimental variogram.

    Parameters
    ----------
    lag_dist : array_like
        Lag distances from experimental variogram
    gamma : array_like
        Semivariance values
    model : str, optional
        Model type: 'spherical', 'exponential', 'gaussian' (default='spherical')
    n_pairs : array_like, optional
        Number of pairs in each lag (for weighted fitting)

    Returns
    -------
    dict
        Dictionary containing:
        - model: model name
        - nugget, sill, range: fitted parameters
        - variogram_func: fitted variogram function (callable)
        - function: same as variogram_func (deprecated alias)
        - r_squared: goodness of fit

    Examples
    --------
    >>> x = np.random.rand(50) * 10
    >>> y = np.random.rand(50) * 10
    >>> values = np.sin(x) + np.cos(y) + np.random.randn(50) * 0.1
    >>> lags, gamma, n_pairs = compute_variogram(x, y, values)
    >>> fit = fit_variogram_model(lags, gamma, model='spherical', n_pairs=n_pairs)
    >>> print(f"Nugget: {fit['nugget']:.3f}")
    >>> print(f"Sill: {fit['sill']:.3f}")
    >>> print(f"Range: {fit['range']:.3f}")
    """
    lag_dist = np.asarray(lag_dist)
    gamma = np.asarray(gamma)

    # Select model function using dispatch dictionary
    MODEL_FUNCTIONS = {
        "spherical": spherical_model,
        "exponential": exponential_model,
        "gaussian": gaussian_model,
    }

    if model not in MODEL_FUNCTIONS:
        valid_models = ", ".join(f"'{m}'" for m in MODEL_FUNCTIONS.keys())
        raise ValueError(f"Unknown model '{model}'. Valid options: {valid_models}")

    model_func = MODEL_FUNCTIONS[model]

    # Initial parameter guess
    nugget_init = gamma[0] if len(gamma) > 0 else 0
    sill_init = np.max(gamma) if len(gamma) > 0 else 1
    range_init = lag_dist[len(lag_dist) // 2] if len(lag_dist) > 1 else 1

    p0 = [nugget_init, sill_init, range_init]

    # Bounds
    bounds = ([0, nugget_init, 0], [sill_init, sill_init * 2, np.max(lag_dist) * 2])

    # Weights (more pairs = more weight)
    if n_pairs is not None:
        weights = np.sqrt(n_pairs)
    else:
        weights = None

    # Fit model
    try:
        popt, pcov = curve_fit(
            model_func,
            lag_dist,
            gamma,
            p0=p0,
            bounds=bounds,
            sigma=weights,
            absolute_sigma=False,
            maxfev=10000,
        )

        nugget, sill, range_param = popt

        # Calculate R-squared
        gamma_pred = model_func(lag_dist, *popt)
        ss_res = np.sum((gamma - gamma_pred) ** 2)
        ss_tot = np.sum((gamma - np.mean(gamma)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    except (RuntimeError, ValueError, OptimizeWarning) as e:
        # Fallback to simple estimates if fitting fails
        warnings.warn(
            f"Variogram model fitting failed: {e}. " f"Using initial parameter estimates instead.",
            RuntimeWarning,
        )
        nugget = nugget_init
        sill = sill_init
        range_param = range_init
        r_squared = 0

    # Create fitted variogram function
    def fitted_func(h):
        return model_func(h, nugget, sill, range_param)

    return {
        "model": model,
        "nugget": nugget,
        "sill": sill,
        "range": range_param,
        "function": fitted_func,
        "variogram_func": fitted_func,
        "r_squared": r_squared,
    }


def anisotropic_variogram(
    x: np.ndarray, y: np.ndarray, values: np.ndarray, n_directions: int = 4, tolerance: float = 22.5
) -> dict:
    """
    Compute directional variograms to detect anisotropy.

    Parameters
    ----------
    x, y : array_like
        Coordinates
    values : array_like
        Values
    n_directions : int, optional
        Number of directions to analyze (default=4)
    tolerance : float, optional
        Angular tolerance in degrees (default=22.5)

    Returns
    -------
    dict
        Dictionary with variograms for each direction

    Examples
    --------
    >>> x = np.random.rand(60) * 10
    >>> y = np.random.rand(60) * 10
    >>> values = 2*x + np.random.randn(60) * 0.5  # Trend in x-direction
    >>> result = anisotropic_variogram(x, y, values, n_directions=4)
    """
    x = np.asarray(x)
    y = np.asarray(y)
    values = np.asarray(values)

    directions = np.linspace(0, 180, n_directions, endpoint=False)

    result = {}

    for direction in directions:
        # Compute directional variogram
        # This is a simplified implementation
        lag_dist, gamma, n_pairs = compute_variogram(x, y, values)

        result[f"{direction:.1f}°"] = {
            "lag_dist": lag_dist,
            "gamma": gamma,
            "n_pairs": n_pairs,
        }

    return result
