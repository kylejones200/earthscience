"""
Change point detection and trend analysis

Detecting regime shifts and structural breaks in time series.
"""

import numpy as np
from scipy import stats


def cusum(data: np.ndarray, threshold: float | None = None) -> dict:
    """
    Cumulative Sum (CUSUM) for detecting shifts in mean.

    Parameters
    ----------
    data : array_like
        Input time series
    threshold : float, optional
        Detection threshold (if None, use 3*std)

    Returns
    -------
    dict
        CUSUM statistics and detected change points

    Examples
    --------
    >>> # Signal with mean shift
    >>> signal = np.concatenate([np.random.randn(100),
    ...                          np.random.randn(100) + 2])
    >>> result = cusum(signal)
    >>> print(f"Change points detected at: {result['change_points']}")
    """
    data = np.asarray(data)
    n = len(data)

    # Estimate mean and std from first portion
    mean_est = np.mean(data[: min(50, n // 2)])
    std_est = np.std(data[: min(50, n // 2)])

    if threshold is None:
        threshold = 3 * std_est

    # CUSUM statistic
    S_high = np.zeros(n)
    S_low = np.zeros(n)

    for i in range(1, n):
        S_high[i] = max(0, S_high[i - 1] + (data[i] - mean_est) - std_est / 2)
        S_low[i] = max(0, S_low[i - 1] - (data[i] - mean_est) - std_est / 2)

    # Detect change points
    change_points = []

    # High side (increase)
    high_exceed = np.where(S_high > threshold)[0]
    if len(high_exceed) > 0:
        change_points.extend(high_exceed.tolist())

    # Low side (decrease)
    low_exceed = np.where(S_low > threshold)[0]
    if len(low_exceed) > 0:
        change_points.extend(low_exceed.tolist())

    change_points = sorted(list(set(change_points)))

    return {
        "S_high": S_high,
        "S_low": S_low,
        "threshold": threshold,
        "change_points": change_points,
        "n_changes": len(change_points),
    }


def pettitt_test(data: np.ndarray) -> dict:
    """
    Pettitt test for detecting a single change point.

    Non-parametric test for detecting a change in the median.

    Parameters
    ----------
    data : array_like
        Input time series

    Returns
    -------
    dict
        Test results with change point location

    Examples
    --------
    >>> # Data with change point
    >>> data = np.concatenate([np.random.randn(50),
    ...                        np.random.randn(50) + 1.5])
    >>> result = pettitt_test(data)
    >>> print(f"Change point at index: {result['change_point']}")
    >>> print(f"P-value: {result['p_value']:.4f}")
    """
    data = np.asarray(data)
    n = len(data)

    # Calculate U statistic for each potential change point
    U = np.zeros(n - 1)

    for t in range(1, n):
        U[t - 1] = 2 * np.sum(np.sign(data[t:] - data[:t]))

    # Find maximum absolute value
    K = np.argmax(np.abs(U))
    K_stat = np.abs(U[K])

    # P-value approximation
    p_value = 2 * np.exp(-6 * K_stat**2 / (n**3 + n**2))

    return {
        "statistic": K_stat,
        "change_point": K + 1,
        "p_value": p_value,
        "significant": p_value < 0.05,
        "U": U,
    }


def mann_kendall_test(data: np.ndarray, alpha: float = 0.05) -> dict:
    """
    Mann-Kendall test for monotonic trend.

    Non-parametric test for detecting trend in time series.

    Parameters
    ----------
    data : array_like
        Input time series
    alpha : float, optional
        Significance level (default=0.05)

    Returns
    -------
    dict
        Test results

    Examples
    --------
    >>> # Increasing trend
    >>> t = np.arange(100)
    >>> data = 0.5*t + np.random.randn(100)*5
    >>> result = mann_kendall_test(data)
    >>> print(f"Trend: {result['trend']}")
    >>> print(f"P-value: {result['p_value']:.4f}")
    """
    data = np.asarray(data)
    n = len(data)

    # Calculate S statistic
    S = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            S += np.sign(data[j] - data[i])

    # Variance of S
    var_S = n * (n - 1) * (2 * n + 5) / 18

    # Z statistic
    if S > 0:
        Z = (S - 1) / np.sqrt(var_S)
    elif S < 0:
        Z = (S + 1) / np.sqrt(var_S)
    else:
        Z = 0

    # Two-tailed p-value
    p_value = 2 * (1 - stats.norm.cdf(abs(Z)))

    # Trend direction
    if p_value < alpha:
        if Z > 0:
            trend = "increasing"
        else:
            trend = "decreasing"
    else:
        trend = "no trend"

    # Sen's slope estimate
    slopes = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            if j != i:
                slope = (data[j] - data[i]) / (j - i)
                slopes.append(slope)

    sen_slope = np.median(slopes) if slopes else 0

    return {
        "statistic": S,
        "Z": Z,
        "p_value": p_value,
        "trend": trend,
        "sen_slope": sen_slope,
        "significant": p_value < alpha,
    }


def binary_segmentation(data: np.ndarray, min_size: int = 10, max_splits: int = 10) -> list[int]:
    """
    Binary segmentation for multiple change point detection.

    Parameters
    ----------
    data : array_like
        Input time series
    min_size : int, optional
        Minimum segment size (default=10)
    max_splits : int, optional
        Maximum number of splits (default=10)

    Returns
    -------
    list
        Indices of detected change points

    Examples
    --------
    >>> # Multiple regime changes
    >>> data = np.concatenate([
    ...     np.random.randn(50),
    ...     np.random.randn(50) + 2,
    ...     np.random.randn(50) - 1
    ... ])
    >>> change_points = binary_segmentation(data, min_size=20)
    >>> print(f"Change points: {change_points}")
    """
    data = np.asarray(data)

    def cost_function(segment):
        """Sum of squared deviations from mean"""
        return np.sum((segment - np.mean(segment)) ** 2)

    def find_best_split(start, end):
        """Find best split point in segment"""
        if end - start < 2 * min_size:
            return None, np.inf

        best_split = None
        best_cost = np.inf

        for split in range(start + min_size, end - min_size + 1):
            # Cost before split
            cost_left = cost_function(data[start:split])
            cost_right = cost_function(data[split:end])
            total_cost = cost_left + cost_right

            if total_cost < best_cost:
                best_cost = total_cost
                best_split = split

        # Compare with no split
        no_split_cost = cost_function(data[start:end])

        if best_cost < no_split_cost:
            return best_split, best_cost
        else:
            return None, no_split_cost

    # Find change points recursively
    change_points = []
    segments_to_split = [(0, len(data))]

    for _ in range(max_splits):
        if not segments_to_split:
            break

        # Find best segment to split
        best_segment = None
        best_improvement = 0
        best_split_point = None

        for seg_idx, (start, end) in enumerate(segments_to_split):
            split, cost = find_best_split(start, end)

            if split is not None:
                no_split_cost = cost_function(data[start:end])
                improvement = no_split_cost - cost

                if improvement > best_improvement:
                    best_improvement = improvement
                    best_segment = seg_idx
                    best_split_point = split

        if best_segment is None:
            break

        # Apply best split
        start, end = segments_to_split.pop(best_segment)
        change_points.append(best_split_point)

        # Add new segments
        segments_to_split.append((start, best_split_point))
        segments_to_split.append((best_split_point, end))

    return sorted(change_points)


def breakpoint_detection_bayesian(data: np.ndarray, prior_prob: float = 0.01) -> dict:
    """
    Bayesian change point detection.

    Parameters
    ----------
    data : array_like
        Input time series
    prior_prob : float, optional
        Prior probability of change point (default=0.01)

    Returns
    -------
    dict
        Posterior probabilities of change points

    Examples
    --------
    >>> data = np.concatenate([np.random.randn(100),
    ...                        np.random.randn(100) + 2])
    >>> result = breakpoint_detection_bayesian(data)
    >>> prob = result['change_probability']
    >>> likely_changes = np.where(prob > 0.5)[0]
    >>> print(f"Likely change points: {likely_changes}")
    """
    data = np.asarray(data)
    n = len(data)

    # Simple Bayesian approach
    # Calculate probability of change at each point
    change_prob = np.zeros(n)

    window = 20  # Window for local statistics

    for t in range(window, n - window):
        # Statistics before and after
        before = data[t - window : t]
        after = data[t : t + window]

        # T-test for difference in means
        t_stat, p_value = stats.ttest_ind(before, after)

        # Convert to probability (simple approach)
        # High t-stat (low p-value) = high probability of change
        change_prob[t] = 1 - p_value if p_value < 0.5 else 0

    # Apply prior
    change_prob = change_prob * (1 - prior_prob) + prior_prob

    return {
        "change_probability": change_prob,
        "likely_changes": np.where(change_prob > 0.5)[0].tolist(),
    }


def trend_decomposition_stl(data: np.ndarray, period: int, seasonal: int = 7) -> dict:
    """
    STL (Seasonal and Trend decomposition using Loess) decomposition.

    Parameters
    ----------
    data : array_like
        Input time series
    period : int
        Seasonal period
    seasonal : int, optional
        Length of seasonal smoother (default=7)

    Returns
    -------
    dict
        Decomposed components

    Examples
    --------
    >>> # Seasonal data
    >>> t = np.arange(365*3)
    >>> seasonal = 10*np.sin(2*np.pi*t/365)
    >>> trend = 0.01*t
    >>> noise = np.random.randn(len(t))*2
    >>> data = seasonal + trend + noise
    >>>
    >>> result = trend_decomposition_stl(data, period=365)
    >>> print("Decomposed into trend, seasonal, and residual")
    """
    from statsmodels.tsa.seasonal import STL

    data = np.asarray(data)

    # Perform STL decomposition
    stl = STL(data, period=period, seasonal=seasonal)
    result = stl.fit()

    return {
        "trend": result.trend,
        "seasonal": result.seasonal,
        "residual": result.resid,
        "observed": data,
    }
