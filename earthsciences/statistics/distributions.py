"""
Probability distributions for earth sciences

Common distributions and fitting methods.
"""

import numpy as np
from scipy import stats


def fit_distribution(data: np.ndarray, dist_name: str = "norm") -> dict:
    """
    Fit a probability distribution to data.

    Parameters
    ----------
    data : array_like
        Input data
    dist_name : str, optional
        Distribution name: 'norm', 'lognorm', 'exponential',
        'weibull', 'gamma', 'uniform', etc. (default='norm')

    Returns
    -------
    dict
        Dictionary containing:
        - params: fitted parameters
        - distribution: scipy distribution object
        - aic: Akaike Information Criterion
        - bic: Bayesian Information Criterion

    Examples
    --------
    >>> data = np.random.lognormal(0, 0.5, 1000)
    >>> result = fit_distribution(data, 'lognorm')
    >>> print(f"Parameters: {result['params']}")
    """
    data = np.asarray(data)
    data = data[~np.isnan(data)]

    # Get distribution from scipy.stats
    try:
        dist = getattr(stats, dist_name)
    except AttributeError:
        raise ValueError(f"Unknown distribution: {dist_name}")

    # Fit distribution
    params = dist.fit(data)

    # Calculate log-likelihood
    log_likelihood = np.sum(dist.logpdf(data, *params))

    # Calculate AIC and BIC
    k = len(params)  # Number of parameters
    n = len(data)  # Sample size
    aic = 2 * k - 2 * log_likelihood
    bic = k * np.log(n) - 2 * log_likelihood

    return {
        "params": params,
        "distribution": dist,
        "log_likelihood": log_likelihood,
        "aic": aic,
        "bic": bic,
    }


def test_normality(data: np.ndarray, method: str = "shapiro") -> tuple[float, float]:
    """
    Test if data follows a normal distribution.

    Parameters
    ----------
    data : array_like
        Input data
    method : str, optional
        Test method: 'shapiro', 'kstest', 'anderson' (default='shapiro')

    Returns
    -------
    statistic : float
        Test statistic
    p_value : float
        P-value (except for Anderson-Darling)

    Examples
    --------
    >>> data = np.random.randn(100)
    >>> stat, p = test_normality(data, method='shapiro')
    >>> print(f"Shapiro-Wilk test: p = {p:.3f}")
    """
    data = np.asarray(data)
    data = data[~np.isnan(data)]

    match method:
        case "shapiro":
            return stats.shapiro(data)

        case "kstest":
            return stats.kstest(data, "norm", args=(np.mean(data), np.std(data)))

        case "anderson":
            result = stats.anderson(data, dist="norm")
            return result.statistic, result.critical_values[2]

        case _:
            raise ValueError(f"Unknown method: {method}")


def generate_random(
    dist_name: str, size: int = 100, params: tuple | None = None, seed: int | None = None
) -> np.ndarray:
    """
    Generate random numbers from a specified distribution.

    Parameters
    ----------
    dist_name : str
        Distribution name ('norm', 'lognorm', 'uniform', etc.)
    size : int, optional
        Number of samples (default=100)
    params : tuple, optional
        Distribution parameters (if None, use defaults)
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    ndarray
        Random samples

    Examples
    --------
    >>> data = generate_random('norm', size=1000, params=(0, 1), seed=42)
    >>> print(f"Mean: {np.mean(data):.3f}, Std: {np.std(data):.3f}")
    """
    if seed is not None:
        np.random.seed(seed)

    # Get distribution from scipy.stats
    try:
        dist = getattr(stats, dist_name)
    except AttributeError:
        raise ValueError(f"Unknown distribution: {dist_name}")

    if params is None:
        match dist_name:
            case "norm":
                params = (0, 1)
            case "lognorm":
                params = (0.5,)
            case "uniform":
                params = (0, 1)
            case _:
                params = ()

    return dist.rvs(*params, size=size)


def qq_plot_data(
    data: np.ndarray, dist_name: str = "norm", params: tuple | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate data for Q-Q (quantile-quantile) plot.

    Parameters
    ----------
    data : array_like
        Input data
    dist_name : str, optional
        Distribution to compare against (default='norm')
    params : tuple, optional
        Distribution parameters (if None, estimate from data)

    Returns
    -------
    theoretical_quantiles : ndarray
        Theoretical quantiles
    sample_quantiles : ndarray
        Sample quantiles (sorted data)

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> data = np.random.randn(100)
    >>> theoretical, sample = qq_plot_data(data, 'norm')
    >>> plt.scatter(theoretical, sample)
    >>> plt.plot([-3, 3], [-3, 3], 'r--')  # Reference line
    >>> plt.xlabel('Theoretical Quantiles')
    >>> plt.ylabel('Sample Quantiles')
    >>> plt.show()
    """
    data = np.asarray(data)
    data = data[~np.isnan(data)]
    data = np.sort(data)

    # Get distribution
    try:
        dist = getattr(stats, dist_name)
    except AttributeError:
        raise ValueError(f"Unknown distribution: {dist_name}")

    # Estimate parameters if not provided
    if params is None:
        params = dist.fit(data)

    # Calculate theoretical quantiles
    n = len(data)
    probabilities = (np.arange(1, n + 1) - 0.5) / n
    theoretical_quantiles = dist.ppf(probabilities, *params)

    return theoretical_quantiles, data


def histogram_bins(data: np.ndarray, method: str = "auto") -> int:
    """
    Calculate optimal number of histogram bins.

    Parameters
    ----------
    data : array_like
        Input data
    method : str, optional
        Method: 'auto', 'sturges', 'scott', 'fd', 'sqrt' (default='auto')

    Returns
    -------
    int
        Number of bins

    Examples
    --------
    >>> data = np.random.randn(1000)
    >>> n_bins = histogram_bins(data, method='sturges')
    >>> print(f"Optimal bins: {n_bins}")
    """
    data = np.asarray(data)
    data = data[~np.isnan(data)]

    n = len(data)

    match method:
        case "auto":
            _, bins = np.histogram(data, bins="auto")
            return len(bins) - 1

        case "sturges":
            return int(np.ceil(np.log2(n) + 1))

        case "scott":
            h = 3.5 * np.std(data) * n ** (-1 / 3)
            return int(np.ceil((data.max() - data.min()) / h))

        case "fd":
            q75, q25 = np.percentile(data, [75, 25])
            iqr = q75 - q25
            h = 2 * iqr * n ** (-1 / 3)
            return int(np.ceil((data.max() - data.min()) / h))

        case "sqrt":
            return int(np.ceil(np.sqrt(n)))

        case _:
            raise ValueError(f"Unknown method: {method}")


def lognormal_stats(data: np.ndarray) -> dict[str, float]:
    """
    Calculate statistics for lognormal data.

    For lognormal data, it's often better to work in log-space.

    Parameters
    ----------
    data : array_like
        Input data (assumed to be lognormally distributed)

    Returns
    -------
    dict
        Statistics in both original and log space

    Examples
    --------
    >>> data = np.random.lognormal(0, 0.5, 1000)
    >>> stats = lognormal_stats(data)
    >>> print(f"Geometric mean: {stats['geometric_mean']:.3f}")
    """
    data = np.asarray(data)
    data = data[~np.isnan(data)]

    if np.any(data <= 0):
        raise ValueError("Lognormal data must be positive")

    # Log-transformed data
    log_data = np.log(data)

    # Geometric mean: exp(mean(log(x)))
    geometric_mean = np.exp(np.mean(log_data))

    # Multiplicative standard deviation: exp(std(log(x)))
    multiplicative_std = np.exp(np.std(log_data, ddof=1))

    return {
        "arithmetic_mean": np.mean(data),
        "geometric_mean": geometric_mean,
        "median": np.median(data),
        "arithmetic_std": np.std(data, ddof=1),
        "multiplicative_std": multiplicative_std,
        "log_mean": np.mean(log_data),
        "log_std": np.std(log_data, ddof=1),
    }
