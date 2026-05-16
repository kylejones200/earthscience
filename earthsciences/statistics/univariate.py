"""
Univariate statistical analysis

Functions for analyzing single-variable data sets.
"""

import numpy as np
from scipy import stats


def descriptive_stats(data: list | np.ndarray, remove_nan: bool = True) -> dict[str, float]:
    """
    Calculate comprehensive descriptive statistics for a dataset.

    Parameters
    ----------
    data : array_like
        Input data array
    remove_nan : bool, optional
        Remove NaN values before calculation (default=True)

    Returns
    -------
    dict
        Dictionary containing statistical measures:
        - mean, median, mode
        - std, variance, range
        - min, max
        - quartiles (Q1, Q2, Q3)
        - skewness, kurtosis

    Examples
    --------
    >>> data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    >>> stats = descriptive_stats(data)
    >>> print(f"Mean: {stats['mean']}, Std: {stats['std']}")
    """
    data = np.asarray(data)

    if remove_nan:
        data = data[~np.isnan(data)]

    if len(data) == 0:
        raise ValueError("Empty array after removing NaN values")

    # Central tendency
    mean_val = np.mean(data)
    median_val = np.median(data)
    mode_result = stats.mode(data, keepdims=True)
    mode_val = mode_result.mode[0] if len(mode_result.mode) > 0 else np.nan

    # Dispersion
    std_val = np.std(data, ddof=1)  # Sample standard deviation
    var_val = np.var(data, ddof=1)  # Sample variance
    range_val = np.ptp(data)  # Peak to peak (max - min)

    # Quantiles
    q1, q2, q3 = np.percentile(data, [25, 50, 75])

    # Shape
    skew_val = stats.skew(data)
    kurt_val = stats.kurtosis(data)

    return {
        "mean": mean_val,
        "median": median_val,
        "mode": mode_val,
        "std": std_val,
        "variance": var_val,
        "range": range_val,
        "min": np.min(data),
        "max": np.max(data),
        "Q1": q1,
        "Q2": q2,
        "Q3": q3,
        "IQR": q3 - q1,
        "skewness": skew_val,
        "kurtosis": kurt_val,
        "count": len(data),
    }


def percentiles(
    data: list | np.ndarray, q: float | list = [25, 50, 75], remove_nan: bool = True
) -> float | np.ndarray:
    """
    Calculate percentiles of a dataset.

    Parameters
    ----------
    data : array_like
        Input data
    q : float or array_like of float
        Percentile(s) to compute (0-100)
    remove_nan : bool, optional
        Remove NaN values (default=True)

    Returns
    -------
    float or ndarray
        Percentile value(s)

    Examples
    --------
    >>> data = np.random.randn(100)
    >>> q25, q50, q75 = percentiles(data, [25, 50, 75])
    """
    data = np.asarray(data)

    if remove_nan:
        data = data[~np.isnan(data)]

    return np.percentile(data, q)


def mode_estimate(
    data: list | np.ndarray, method: str = "kernel", bandwidth: float | None = None
) -> float:
    """
    Estimate the mode of a distribution using kernel density estimation.

    More robust than simple mode for continuous data.

    Parameters
    ----------
    data : array_like
        Input data
    method : str, optional
        Method to use: 'kernel' or 'simple' (default='kernel')
    bandwidth : float, optional
        Bandwidth for kernel density estimation

    Returns
    -------
    float
        Estimated mode

    Examples
    --------
    >>> data = np.random.lognormal(0, 0.5, 1000)
    >>> mode_val = mode_estimate(data, method='kernel')
    """
    data = np.asarray(data)
    data = data[~np.isnan(data)]

    match method:
        case "simple":
            mode_result = stats.mode(data, keepdims=True)
            return mode_result.mode[0] if len(mode_result.mode) > 0 else np.nan

        case "kernel":
            arr = np.asarray(data, dtype=float)
            if bandwidth is None:
                bandwidth = 1.06 * np.std(arr) * len(arr) ** (-1 / 5)

            kde = stats.gaussian_kde(arr, bw_method=bandwidth)
            x_eval = np.linspace(float(arr.min()), float(arr.max()), 1000)
            density = kde(x_eval)
            mode_idx = np.argmax(density)

            return x_eval[mode_idx]

        case _:
            raise ValueError(f"Unknown method '{method}'. Valid options: 'max', 'kde'")


def skewness(data: list | np.ndarray, remove_nan: bool = True, bias: bool = False) -> float:
    """
    Calculate the skewness of a distribution.

    Skewness measures the asymmetry of the distribution:
    - Negative: left-skewed (tail on left)
    - Zero: symmetric
    - Positive: right-skewed (tail on right)

    Parameters
    ----------
    data : array_like
        Input data
    remove_nan : bool, optional
        Remove NaN values (default=True)
    bias : bool, optional
        If False, use bias-corrected calculation (default=False)

    Returns
    -------
    float
        Skewness value

    Examples
    --------
    >>> normal_data = np.random.randn(1000)
    >>> print(f"Skewness: {skewness(normal_data):.3f}")  # Should be ~0
    >>>
    >>> lognormal_data = np.random.lognormal(0, 1, 1000)
    >>> print(f"Skewness: {skewness(lognormal_data):.3f}")  # Positive
    """
    data = np.asarray(data)

    if remove_nan:
        data = data[~np.isnan(data)]

    return stats.skew(data, bias=bias)


def kurtosis(data: list | np.ndarray, remove_nan: bool = True, fisher: bool = True) -> float:
    """
    Calculate the kurtosis of a distribution.

    Kurtosis measures the "tailedness" of the distribution:
    - Negative (fisher=True): lighter tails than normal (platykurtic)
    - Zero (fisher=True): normal distribution (mesokurtic)
    - Positive (fisher=True): heavier tails than normal (leptokurtic)

    Parameters
    ----------
    data : array_like
        Input data
    remove_nan : bool, optional
        Remove NaN values (default=True)
    fisher : bool, optional
        If True, return excess kurtosis (subtract 3) (default=True)

    Returns
    -------
    float
        Kurtosis value

    Examples
    --------
    >>> normal_data = np.random.randn(1000)
    >>> print(f"Kurtosis: {kurtosis(normal_data):.3f}")  # Should be ~0
    """
    data = np.asarray(data)

    if remove_nan:
        data = data[~np.isnan(data)]

    return stats.kurtosis(data, fisher=fisher)


def z_score(data: list | np.ndarray, remove_nan: bool = True) -> np.ndarray:
    """
    Standardize data to z-scores (mean=0, std=1).

    Parameters
    ----------
    data : array_like
        Input data
    remove_nan : bool, optional
        Remove NaN values (default=True)

    Returns
    -------
    ndarray
        Standardized data

    Examples
    --------
    >>> data = np.array([1, 2, 3, 4, 5])
    >>> z = z_score(data)
    >>> print(f"Mean: {np.mean(z):.6f}, Std: {np.std(z, ddof=1):.6f}")
    """
    data = np.asarray(data, dtype=float)

    if remove_nan:
        mask = ~np.isnan(data)
        result = np.full_like(data, np.nan)
        result[mask] = stats.zscore(data[mask], ddof=1)
        return result
    else:
        return stats.zscore(data, ddof=1)


def coefficient_of_variation(data: list | np.ndarray, remove_nan: bool = True) -> float:
    """
    Calculate the coefficient of variation (CV = std/mean).

    Useful for comparing variability between datasets with different means.

    Parameters
    ----------
    data : array_like
        Input data
    remove_nan : bool, optional
        Remove NaN values (default=True)

    Returns
    -------
    float
        Coefficient of variation

    Examples
    --------
    >>> data1 = np.array([10, 12, 14, 16, 18])
    >>> data2 = np.array([100, 120, 140, 160, 180])
    >>> print(f"CV1: {coefficient_of_variation(data1):.3f}")
    >>> print(f"CV2: {coefficient_of_variation(data2):.3f}")
    """
    data = np.asarray(data)

    if remove_nan:
        data = data[~np.isnan(data)]

    mean_val = np.mean(data)
    std_val = np.std(data, ddof=1)

    if mean_val == 0:
        return np.inf

    return std_val / mean_val
