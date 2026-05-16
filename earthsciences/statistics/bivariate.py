"""
Bivariate statistical analysis

Functions for analyzing relationships between two variables.
"""

import numpy as np
from scipy import stats


def correlation(x: np.ndarray, y: np.ndarray, method: str = "pearson") -> tuple[float, float]:
    """
    Calculate correlation coefficient between two variables.

    Parameters
    ----------
    x, y : array_like
        Input data arrays
    method : str, optional
        Correlation method: 'pearson', 'spearman', or 'kendall' (default='pearson')

    Returns
    -------
    r : float
        Correlation coefficient
    p_value : float
        Two-tailed p-value

    Examples
    --------
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> y = np.array([2, 4, 5, 4, 5])
    >>> r, p = correlation(x, y, method='pearson')
    >>> print(f"r = {r:.3f}, p = {p:.3f}")
    """
    x = np.asarray(x)
    y = np.asarray(y)

    # Remove NaN values
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]

    if len(x) < 3:
        raise ValueError("Need at least 3 data points")

    match method:
        case "pearson":
            r, p = stats.pearsonr(x, y)
        case "spearman":
            r, p = stats.spearmanr(x, y)
        case "kendall":
            r, p = stats.kendalltau(x, y)
        case _:
            raise ValueError(f"Unknown method: {method}")

    return r, p


def linear_regression(x: np.ndarray, y: np.ndarray) -> dict:
    """
    Perform ordinary least squares (OLS) linear regression.

    Fits y = slope * x + intercept

    Parameters
    ----------
    x, y : array_like
        Input data (independent and dependent variables)

    Returns
    -------
    dict
        Dictionary containing:
        - slope, intercept: regression coefficients
        - r_value: correlation coefficient
        - p_value: two-tailed p-value
        - std_err: standard error of the estimate
        - residuals: residuals (y - y_pred)

    Examples
    --------
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> y = np.array([2.1, 3.9, 6.2, 7.8, 10.1])
    >>> result = linear_regression(x, y)
    >>> print(f"y = {result['slope']:.2f}x + {result['intercept']:.2f}")
    """
    x = np.asarray(x)
    y = np.asarray(y)

    # Remove NaN values
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        raise ValueError("Need at least 2 data points")

    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    # Calculate residuals
    y_pred = slope * x + intercept
    residuals = y - y_pred

    # Calculate additional statistics
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    return {
        "slope": slope,
        "intercept": intercept,
        "r_value": r_value,
        "r_squared": r_squared,
        "p_value": p_value,
        "std_err": std_err,
        "residuals": residuals,
        "predicted": y_pred,
    }


def rma_regression(x: np.ndarray, y: np.ndarray) -> dict:
    """
    Reduced Major Axis (RMA) regression, also known as Type II regression.

    More appropriate when both x and y have measurement errors.

    Parameters
    ----------
    x, y : array_like
        Input data

    Returns
    -------
    dict
        Dictionary containing:
        - slope, intercept: RMA regression coefficients
        - r_value: correlation coefficient

    Examples
    --------
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> y = np.array([2.1, 3.9, 6.2, 7.8, 10.1])
    >>> result = rma_regression(x, y)
    >>> print(f"RMA: y = {result['slope']:.2f}x + {result['intercept']:.2f}")
    """
    x = np.asarray(x)
    y = np.asarray(y)

    # Remove NaN values
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        raise ValueError("Need at least 2 data points")

    # Calculate means and standard deviations
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    x_std = np.std(x, ddof=1)
    y_std = np.std(y, ddof=1)

    # Calculate correlation
    r_value = np.corrcoef(x, y)[0, 1]

    # RMA slope: sign(r) * std(y) / std(x)
    slope = np.sign(r_value) * y_std / x_std
    intercept = y_mean - slope * x_mean

    # Predictions
    y_pred = slope * x + intercept
    residuals = y - y_pred

    return {
        "slope": slope,
        "intercept": intercept,
        "r_value": r_value,
        "predicted": y_pred,
        "residuals": residuals,
    }


def polynomial_fit(x: np.ndarray, y: np.ndarray, degree: int = 2) -> dict:
    """
    Fit polynomial to data.

    Parameters
    ----------
    x, y : array_like
        Input data
    degree : int, optional
        Degree of polynomial (default=2)

    Returns
    -------
    dict
        Dictionary containing:
        - coefficients: polynomial coefficients (highest degree first)
        - polynomial: numpy polynomial object
        - predicted: predicted y values
        - residuals: residuals (y - y_pred)
        - r_squared: coefficient of determination

    Examples
    --------
    >>> x = np.linspace(0, 10, 20)
    >>> y = 2*x**2 - 3*x + 1 + np.random.randn(20)
    >>> result = polynomial_fit(x, y, degree=2)
    >>> print(f"Coefficients: {result['coefficients']}")
    """
    x = np.asarray(x)
    y = np.asarray(y)

    # Remove NaN values
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]

    if len(x) <= degree:
        raise ValueError(f"Need at least {degree+1} data points for degree {degree}")

    # Fit polynomial
    coefficients = np.polyfit(x, y, degree)
    polynomial = np.poly1d(coefficients)

    # Calculate predictions and residuals
    y_pred = polynomial(x)
    residuals = y - y_pred

    # Calculate R-squared
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    return {
        "coefficients": coefficients,
        "polynomial": polynomial,
        "predicted": y_pred,
        "residuals": residuals,
        "r_squared": r_squared,
    }


def moving_average(data: np.ndarray, window: int = 3, mode: str = "same") -> np.ndarray:
    """
    Calculate moving average (running mean) of data.

    Parameters
    ----------
    data : array_like
        Input data
    window : int, optional
        Window size (default=3)
    mode : str, optional
        Boundary handling: 'same', 'valid', 'full' (default='same')

    Returns
    -------
    ndarray
        Smoothed data

    Examples
    --------
    >>> data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> smoothed = moving_average(data, window=3)
    """
    data = np.asarray(data)

    if window < 1:
        raise ValueError("Window size must be at least 1")

    # Use convolution for moving average
    weights = np.ones(window) / window

    match mode:
        case "same":
            return np.convolve(data, weights, mode="same")
        case "valid":
            return np.convolve(data, weights, mode="valid")
        case "full":
            return np.convolve(data, weights, mode="full")
        case _:
            raise ValueError(f"Unknown mode: {mode}")


def detrend(data: np.ndarray, method: str = "linear") -> np.ndarray:
    """
    Remove trend from data.

    Parameters
    ----------
    data : array_like
        Input data
    method : str, optional
        Detrending method: 'linear' or 'constant' (default='linear')

    Returns
    -------
    ndarray
        Detrended data

    Examples
    --------
    >>> x = np.linspace(0, 10, 100)
    >>> data = 2*x + 5 + np.random.randn(100)
    >>> detrended = detrend(data, method='linear')
    """
    from scipy import signal

    data = np.asarray(data)

    match method:
        case "linear":
            return signal.detrend(data, type="linear")
        case "constant":
            return signal.detrend(data, type="constant")
        case _:
            raise ValueError(f"Unknown method: {method}")


def confidence_interval(data: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    """
    Calculate confidence interval for the mean.

    Parameters
    ----------
    data : array_like
        Input data
    confidence : float, optional
        Confidence level (default=0.95)

    Returns
    -------
    lower, upper : float
        Lower and upper confidence bounds

    Examples
    --------
    >>> data = np.random.randn(100)
    >>> lower, upper = confidence_interval(data, confidence=0.95)
    >>> print(f"95% CI: [{lower:.3f}, {upper:.3f}]")
    """
    data = np.asarray(data)
    data = data[~np.isnan(data)]

    n = len(data)
    mean = np.mean(data)
    sem = stats.sem(data)  # Standard error of the mean

    # t-distribution for confidence interval
    t_value = stats.t.ppf((1 + confidence) / 2, n - 1)
    margin = t_value * sem

    return mean - margin, mean + margin
