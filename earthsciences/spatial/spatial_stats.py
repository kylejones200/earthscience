"""
Spatial autocorrelation and spatial statistics

Moran's I, Geary's C, LISA, and spatial regression.
"""

import numpy as np
from scipy import stats
from scipy.spatial import distance_matrix


def spatial_weights_matrix(
    coordinates: np.ndarray,
    method: str = "inverse_distance",
    k_neighbors: int | None = None,
    distance_threshold: float | None = None,
    power: float = 1.0,
) -> np.ndarray:
    """
    Create spatial weights matrix.

    Parameters
    ----------
    coordinates : array_like
        Coordinates of points (n_points, 2)
    method : str, optional
        Method: 'inverse_distance', 'knn', 'threshold' (default='inverse_distance')
    k_neighbors : int, optional
        Number of nearest neighbors for 'knn'
    distance_threshold : float, optional
        Distance threshold for 'threshold' method
    power : float, optional
        Power for inverse distance weighting (default=1.0)

    Returns
    -------
    ndarray
        Spatial weights matrix (n_points, n_points)

    Examples
    --------
    >>> coords = np.random.rand(50, 2) * 100
    >>> W = spatial_weights_matrix(coords, method='inverse_distance')
    >>> print(f"Weights matrix shape: {W.shape}")
    """
    coordinates = np.asarray(coordinates)
    n = len(coordinates)

    # Calculate distance matrix
    D = distance_matrix(coordinates, coordinates)

    if method == "inverse_distance":
        # Inverse distance weighting
        with np.errstate(divide="ignore", invalid="ignore"):
            W = 1 / (D**power)
            W[~np.isfinite(W)] = 0  # Set diagonal (self-distance) to 0
        np.fill_diagonal(W, 0)

    elif method == "knn":
        # K-nearest neighbors
        if k_neighbors is None:
            k_neighbors = int(np.sqrt(n))

        W = np.zeros((n, n))
        for i in range(n):
            # Find k nearest neighbors
            distances = D[i, :]
            nearest = np.argsort(distances)[1 : k_neighbors + 1]  # Exclude self
            W[i, nearest] = 1

    elif method == "threshold":
        # Distance threshold
        if distance_threshold is None:
            distance_threshold = np.percentile(D[D > 0], 25)

        W = (D <= distance_threshold).astype(float)
        np.fill_diagonal(W, 0)

    else:
        raise ValueError(f"Unknown method: {method}")

    # Row-standardize
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    W = W / row_sums

    return W


def morans_i(
    values: np.ndarray, coordinates: np.ndarray, weights: np.ndarray | None = None
) -> dict:
    """
    Calculate Moran's I statistic for spatial autocorrelation.

    Moran's I ranges from -1 (perfect dispersion) to +1 (perfect clustering).

    Parameters
    ----------
    values : array_like
        Variable values at each location
    coordinates : array_like
        Spatial coordinates (n_points, 2)
    weights : array_like, optional
        Spatial weights matrix (if None, compute inverse distance)

    Returns
    -------
    dict
        Moran's I statistic, expected value, variance, and p-value

    Examples
    --------
    >>> # Spatially clustered data
    >>> coords = np.random.rand(100, 2) * 10
    >>> values = np.sin(coords[:, 0]) + np.sin(coords[:, 1])
    >>> result = morans_i(values, coords)
    >>> print(f"Moran's I: {result['I']:.3f}")
    >>> print(f"P-value: {result['p_value']:.4f}")
    >>> if result['p_value'] < 0.05:
    ...     print("Significant spatial autocorrelation detected")
    """
    values = np.asarray(values)
    coordinates = np.asarray(coordinates)

    if weights is None:
        weights = spatial_weights_matrix(coordinates, method="inverse_distance")

    n = len(values)

    # Standardize values
    z = values - np.mean(values)

    # Moran's I
    numerator = n * np.sum(weights * np.outer(z, z))
    denominator = np.sum(weights) * np.sum(z**2)

    I = numerator / denominator if denominator != 0 else 0

    # Expected value under null hypothesis
    EI = -1 / (n - 1)

    # Variance (under randomization assumption)
    S0 = np.sum(weights)
    S1 = 0.5 * np.sum((weights + weights.T) ** 2)
    S2 = np.sum((weights.sum(axis=1) + weights.sum(axis=0)) ** 2)

    b2 = (n * np.sum(z**4)) / (np.sum(z**2) ** 2)

    VI_num = n * ((n**2 - 3 * n + 3) * S1 - n * S2 + 3 * (S0**2))
    VI_den = (n - 1) * (n - 2) * (n - 3) * (S0**2)
    VI = (VI_num / VI_den) - (EI**2)

    # Z-score
    Z = (I - EI) / np.sqrt(VI) if VI > 0 else 0

    # P-value (two-tailed)
    p_value = 2 * (1 - stats.norm.cdf(abs(Z)))

    return {
        "I": I,
        "EI": EI,
        "VI": VI,
        "Z": Z,
        "p_value": p_value,
        "significant": p_value < 0.05,
    }


def gearys_c(
    values: np.ndarray, coordinates: np.ndarray, weights: np.ndarray | None = None
) -> dict:
    """
    Calculate Geary's C statistic for spatial autocorrelation.

    Geary's C ranges from 0 (perfect clustering) to 2 (perfect dispersion).
    C = 1 indicates random spatial pattern.

    Parameters
    ----------
    values : array_like
        Variable values
    coordinates : array_like
        Spatial coordinates
    weights : array_like, optional
        Spatial weights matrix

    Returns
    -------
    dict
        Geary's C statistic and significance

    Examples
    --------
    >>> coords = np.random.rand(80, 2) * 10
    >>> values = np.random.randn(80).cumsum()  # Spatially correlated
    >>> result = gearys_c(values, coords)
    >>> print(f"Geary's C: {result['C']:.3f}")
    >>> if result['C'] < 1:
    ...     print("Positive spatial autocorrelation (clustering)")
    """
    values = np.asarray(values)
    coordinates = np.asarray(coordinates)

    if weights is None:
        weights = spatial_weights_matrix(coordinates, method="inverse_distance")

    n = len(values)

    # Geary's C
    numerator = 0
    for i in range(n):
        for j in range(n):
            numerator += weights[i, j] * (values[i] - values[j]) ** 2

    numerator *= n - 1
    denominator = 2 * np.sum(weights) * np.sum((values - np.mean(values)) ** 2)

    C = numerator / denominator if denominator != 0 else 1

    # Expected value
    EC = 1.0

    # Approximate Z-score (simplified)
    # For detailed variance calculation, see Cliff & Ord (1981)
    Z = (C - EC) / 0.1  # Simplified
    p_value = 2 * (1 - stats.norm.cdf(abs(Z)))

    return {
        "C": C,
        "EC": EC,
        "Z": Z,
        "p_value": p_value,
        "significant": p_value < 0.05,
    }


def local_morans_i(
    values: np.ndarray, coordinates: np.ndarray, weights: np.ndarray | None = None
) -> dict:
    """
    Calculate Local Indicators of Spatial Association (LISA).

    Identifies local clusters and spatial outliers.

    Parameters
    ----------
    values : array_like
        Variable values
    coordinates : array_like
        Spatial coordinates
    weights : array_like, optional
        Spatial weights matrix

    Returns
    -------
    dict
        Local Moran's I for each location

    Examples
    --------
    >>> coords = np.random.rand(100, 2) * 10
    >>> values = np.random.randn(100)
    >>> values[:20] += 5  # Create local cluster
    >>> result = local_morans_i(values, coords)
    >>> hotspots = result['significant'] & (result['Ii'] > 0)
    >>> print(f"Hot spots detected: {np.sum(hotspots)}")
    """
    values = np.asarray(values)
    coordinates = np.asarray(coordinates)

    if weights is None:
        weights = spatial_weights_matrix(coordinates, method="inverse_distance")

    n = len(values)

    # Standardize
    z = values - np.mean(values)
    z_std = z / np.std(values)

    # Local Moran's I for each location
    Ii = np.zeros(n)

    for i in range(n):
        # Spatial lag
        lag = np.sum(weights[i, :] * z_std)
        Ii[i] = z_std[i] * lag

    # Approximate significance (permutation test would be more accurate)
    EIi = -1 / (n - 1)
    p_values = np.ones(n)

    # Simple Z-score approximation
    for i in range(n):
        # Variance approximation
        var_Ii = 1.0  # Simplified
        Z = (Ii[i] - EIi) / np.sqrt(var_Ii)
        p_values[i] = 2 * (1 - stats.norm.cdf(abs(Z)))

    # Classify into quadrants
    classifications = np.zeros(n, dtype=int)
    for i in range(n):
        lag = np.sum(weights[i, :] * z_std)
        if z_std[i] > 0 and lag > 0:
            classifications[i] = 1  # HH: High-High
        elif z_std[i] < 0 and lag < 0:
            classifications[i] = 2  # LL: Low-Low
        elif z_std[i] > 0 and lag < 0:
            classifications[i] = 3  # HL: High-Low (outlier)
        elif z_std[i] < 0 and lag > 0:
            classifications[i] = 4  # LH: Low-High (outlier)

    return {
        "Ii": Ii,
        "p_values": p_values,
        "significant": p_values < 0.05,
        "classifications": classifications,
        "cluster_names": ["Not significant", "HH", "LL", "HL", "LH"],
    }


def getis_ord_g(
    values: np.ndarray, coordinates: np.ndarray, weights: np.ndarray | None = None
) -> dict:
    """
    Calculate Getis-Ord G* statistic for hot spot analysis.

    Parameters
    ----------
    values : array_like
        Variable values
    coordinates : array_like
        Spatial coordinates
    weights : array_like, optional
        Spatial weights matrix

    Returns
    -------
    dict
        G* statistics identifying hot and cold spots

    Examples
    --------
    >>> coords = np.random.rand(100, 2) * 10
    >>> values = np.random.randn(100)
    >>> values[coords[:, 0] < 3] += 3  # Create hot spot
    >>> result = getis_ord_g(values, coords)
    >>> hot_spots = result['Gi_star'] > 1.96  # 95% confidence
    >>> print(f"Hot spots: {np.sum(hot_spots)}")
    """
    values = np.asarray(values)
    coordinates = np.asarray(coordinates)

    if weights is None:
        weights = spatial_weights_matrix(coordinates, method="inverse_distance")

    n = len(values)
    mean_val = np.mean(values)
    std_val = np.std(values)

    # G* statistic for each location
    Gi_star = np.zeros(n)

    for i in range(n):
        # Include location i in the calculation (G*)
        W_i = weights[i, :].copy()
        W_i[i] = 1  # Include self

        # Calculate G*
        numerator = np.sum(W_i * values) - mean_val * np.sum(W_i)
        denominator = std_val * np.sqrt((n * np.sum(W_i**2) - np.sum(W_i) ** 2) / (n - 1))

        Gi_star[i] = numerator / denominator if denominator != 0 else 0

    # P-values
    p_values = 2 * (1 - stats.norm.cdf(np.abs(Gi_star)))

    return {
        "Gi_star": Gi_star,
        "p_values": p_values,
        "hot_spots": (Gi_star > 1.96) & (p_values < 0.05),
        "cold_spots": (Gi_star < -1.96) & (p_values < 0.05),
    }


def spatial_correlogram(
    values: np.ndarray,
    coordinates: np.ndarray,
    n_lags: int = 10,
    max_distance: float | None = None,
) -> dict:
    """
    Calculate spatial correlogram (Moran's I vs distance).

    Parameters
    ----------
    values : array_like
        Variable values
    coordinates : array_like
        Spatial coordinates
    n_lags : int, optional
        Number of distance lags (default=10)
    max_distance : float, optional
        Maximum distance to consider

    Returns
    -------
    dict
        Moran's I at different distance lags

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> coords = np.random.rand(100, 2) * 100
    >>> values = np.sin(coords[:, 0]/10) + np.cos(coords[:, 1]/10)
    >>> result = spatial_correlogram(values, coords, n_lags=15)
    >>>
    >>> plt.plot(result['distances'], result['morans_i'], 'o-')
    >>> plt.axhline(y=0, color='r', linestyle='--')
    >>> plt.xlabel('Distance')
    >>> plt.ylabel("Moran's I")
    >>> plt.title('Spatial Correlogram')
    >>> plt.show()
    """
    values = np.asarray(values)
    coordinates = np.asarray(coordinates)

    # Distance matrix
    D = distance_matrix(coordinates, coordinates)

    if max_distance is None:
        max_distance = np.percentile(D[D > 0], 75)

    # Distance bins
    distance_bins = np.linspace(0, max_distance, n_lags + 1)
    distance_centers = (distance_bins[:-1] + distance_bins[1:]) / 2

    morans_i_vals = []

    for i in range(n_lags):
        # Create weights for this distance lag
        mask = (D >= distance_bins[i]) & (D < distance_bins[i + 1])

        if np.sum(mask) == 0:
            morans_i_vals.append(np.nan)
            continue

        W = mask.astype(float)

        # Row-standardize
        row_sums = W.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        W = W / row_sums

        # Calculate Moran's I for this lag
        result = morans_i(values, coordinates, weights=W)
        morans_i_vals.append(result["I"])

    return {
        "distances": distance_centers,
        "morans_i": np.array(morans_i_vals),
        "distance_bins": distance_bins,
    }


def semivariogram(
    values: np.ndarray,
    coordinates: np.ndarray,
    n_lags: int = 15,
    max_distance: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate empirical semivariogram (same as variogram in variogram.py but here for completeness).

    Parameters
    ----------
    values : array_like
        Variable values
    coordinates : array_like
        Spatial coordinates
    n_lags : int, optional
        Number of distance lags
    max_distance : float, optional
        Maximum distance

    Returns
    -------
    distances : ndarray
        Lag distances
    semivariance : ndarray
        Semivariance values

    Examples
    --------
    >>> coords = np.random.rand(50, 2) * 10
    >>> values = np.sin(coords[:, 0]) + np.random.randn(50) * 0.1
    >>> dist, gamma = semivariogram(values, coords)
    >>>
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(dist, gamma, 'o-')
    >>> plt.xlabel('Distance')
    >>> plt.ylabel('Semivariance')
    >>> plt.show()
    """
    from ..spatial.variogram import compute_variogram

    # Use the existing variogram function
    x, y = coordinates[:, 0], coordinates[:, 1]
    lag_dist, gamma, _ = compute_variogram(x, y, values, n_lags=n_lags, max_lag=max_distance)

    return lag_dist, gamma
