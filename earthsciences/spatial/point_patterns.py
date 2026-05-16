"""
Point pattern analysis

Spatial statistics for point distributions.
"""

import numpy as np
from scipy.spatial import cKDTree, distance


def nearest_neighbor_distance(x: np.ndarray, y: np.ndarray, k: int = 1) -> np.ndarray:
    """
    Calculate nearest neighbor distances.

    Parameters
    ----------
    x, y : array_like
        Point coordinates
    k : int, optional
        Which nearest neighbor (1=closest, 2=second closest, etc.) (default=1)

    Returns
    -------
    ndarray
        Nearest neighbor distances for each point

    Examples
    --------
    >>> # Random points
    >>> x = np.random.rand(100) * 10
    >>> y = np.random.rand(100) * 10
    >>> distances = nearest_neighbor_distance(x, y, k=1)
    >>> print(f"Mean NN distance: {np.mean(distances):.3f}")
    """
    x = np.asarray(x)
    y = np.asarray(y)

    points = np.column_stack([x, y])
    tree = cKDTree(points)

    # Query for k+1 nearest neighbors (including self)
    distances, _ = tree.query(points, k=k + 1)

    # Return k-th nearest neighbor (skip self at index 0)
    return distances[:, k]


def ripley_k(
    x: np.ndarray, y: np.ndarray, distances: np.ndarray, area: float | None = None
) -> np.ndarray:
    """
    Calculate Ripley's K function for spatial point pattern analysis.

    Used to detect clustering or dispersion at different scales.

    Parameters
    ----------
    x, y : array_like
        Point coordinates
    distances : array_like
        Distances at which to evaluate K
    area : float, optional
        Study area (if None, use bounding box)

    Returns
    -------
    ndarray
        K values at each distance

    Examples
    --------
    >>> # Generate clustered points
    >>> n_clusters = 5
    >>> points_per_cluster = 20
    >>> x = np.concatenate([np.random.randn(points_per_cluster) + i*5
    ...                     for i in range(n_clusters)])
    >>> y = np.concatenate([np.random.randn(points_per_cluster) + i*5
    ...                     for i in range(n_clusters)])
    >>>
    >>> dist = np.linspace(0.1, 10, 50)
    >>> K = ripley_k(x, y, dist)
    >>>
    >>> # Compare to random expectation (K = pi * r^2)
    >>> K_random = np.pi * dist**2
    >>>
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(dist, K, label='Observed')
    >>> plt.plot(dist, K_random, '--', label='Random')
    >>> plt.xlabel('Distance')
    >>> plt.ylabel('K(r)')
    >>> plt.legend()
    >>> plt.show()
    """
    x = np.asarray(x)
    y = np.asarray(y)
    distances = np.asarray(distances)

    n = len(x)

    if area is None:
        # Use bounding box area
        area = (np.max(x) - np.min(x)) * (np.max(y) - np.min(y))

    # Compute pairwise distances
    points = np.column_stack([x, y])
    dist_matrix = distance.squareform(distance.pdist(points))

    K = np.zeros(len(distances))

    for i, r in enumerate(distances):
        # Count pairs within distance r
        count = np.sum(dist_matrix < r) - n  # Subtract diagonal

        # Ripley's K
        K[i] = (area * count) / (n * (n - 1))

    return K


def point_density(
    x: np.ndarray,
    y: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    bandwidth: float | None = None,
) -> np.ndarray:
    """
    Calculate kernel density estimate for point pattern.

    Parameters
    ----------
    x, y : array_like
        Point coordinates
    grid_x, grid_y : array_like
        Grid for density estimation
    bandwidth : float, optional
        Kernel bandwidth (if None, use Scott's rule)

    Returns
    -------
    ndarray
        Density values on grid

    Examples
    --------
    >>> # Random points
    >>> x = np.random.rand(100) * 10
    >>> y = np.random.rand(100) * 10
    >>>
    >>> # Create grid
    >>> grid_x, grid_y = np.meshgrid(np.linspace(0, 10, 50),
    ...                              np.linspace(0, 10, 50))
    >>>
    >>> # Calculate density
    >>> density = point_density(x, y, grid_x, grid_y)
    >>>
    >>> import matplotlib.pyplot as plt
    >>> plt.pcolormesh(grid_x, grid_y, density, shading='auto')
    >>> plt.scatter(x, y, c='red', s=10, alpha=0.5)
    >>> plt.colorbar(label='Density')
    >>> plt.show()
    """
    from scipy.stats import gaussian_kde

    x = np.asarray(x)
    y = np.asarray(y)

    if bandwidth is None:
        # Scott's rule
        bandwidth = len(x) ** (-1 / 6)

    # Prepare points
    points = np.vstack([x, y])

    # KDE
    kde = gaussian_kde(points, bw_method=bandwidth)

    # Evaluate on grid
    grid_points = np.vstack([grid_x.ravel(), grid_y.ravel()])
    density = kde(grid_points)

    return density.reshape(grid_x.shape)


def clark_evans_statistic(
    x: np.ndarray, y: np.ndarray, area: float | None = None
) -> tuple[float, float]:
    """
    Calculate Clark-Evans aggregation index.

    R = observed mean NN distance / expected mean NN distance
    R < 1: clustered
    R = 1: random
    R > 1: dispersed

    Parameters
    ----------
    x, y : array_like
        Point coordinates
    area : float, optional
        Study area (if None, use bounding box)

    Returns
    -------
    R : float
        Clark-Evans statistic
    z_score : float
        Z-score for significance test

    Examples
    --------
    >>> # Clustered points
    >>> x = np.concatenate([np.random.randn(20) + 5,
    ...                     np.random.randn(20) + 15])
    >>> y = np.concatenate([np.random.randn(20) + 5,
    ...                     np.random.randn(20) + 15])
    >>> R, z = clark_evans_statistic(x, y)
    >>> print(f"R = {R:.3f} (R<1 indicates clustering)")
    >>> print(f"Z-score = {z:.3f}")
    """
    x = np.asarray(x)
    y = np.asarray(y)

    n = len(x)

    if area is None:
        area = (np.max(x) - np.min(x)) * (np.max(y) - np.min(y))

    # Observed mean nearest neighbor distance
    nn_dist = nearest_neighbor_distance(x, y, k=1)
    r_obs = np.mean(nn_dist)

    # Expected mean NN distance for random pattern
    density = n / area
    r_exp = 0.5 / np.sqrt(density)

    # Clark-Evans statistic
    R = r_obs / r_exp

    # Standard error
    se = 0.26136 / np.sqrt(n * density)

    # Z-score
    z_score = (r_obs - r_exp) / se

    return R, z_score


def g_function(x: np.ndarray, y: np.ndarray, distances: np.ndarray) -> np.ndarray:
    """
    Calculate nearest neighbor distribution function G(r).

    G(r) = proportion of points with nearest neighbor within distance r

    Parameters
    ----------
    x, y : array_like
        Point coordinates
    distances : array_like
        Distances at which to evaluate G

    Returns
    -------
    ndarray
        G values

    Examples
    --------
    >>> x = np.random.rand(100) * 10
    >>> y = np.random.rand(100) * 10
    >>> dist = np.linspace(0, 2, 50)
    >>> G = g_function(x, y, dist)
    """
    x = np.asarray(x)
    y = np.asarray(y)
    distances = np.asarray(distances)

    # Get nearest neighbor distances
    nn_dist = nearest_neighbor_distance(x, y, k=1)

    # Calculate G function
    G = np.zeros(len(distances))
    for i, r in enumerate(distances):
        G[i] = np.mean(nn_dist <= r)

    return G


def f_function(
    x: np.ndarray, y: np.ndarray, distances: np.ndarray, n_random: int = 100
) -> np.ndarray:
    """
    Calculate empty space function F(r).

    F(r) = proportion of random points with nearest event within distance r

    Parameters
    ----------
    x, y : array_like
        Point coordinates
    distances : array_like
        Distances at which to evaluate F
    n_random : int, optional
        Number of random points to generate (default=100)

    Returns
    -------
    ndarray
        F values

    Examples
    --------
    >>> x = np.random.rand(50) * 10
    >>> y = np.random.rand(50) * 10
    >>> dist = np.linspace(0, 2, 50)
    >>> F = f_function(x, y, dist, n_random=200)
    """
    x = np.asarray(x)
    y = np.asarray(y)
    distances = np.asarray(distances)

    # Generate random points in study area
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)

    x_random = np.random.uniform(x_min, x_max, n_random)
    y_random = np.random.uniform(y_min, y_max, n_random)

    # Build tree of actual points
    points = np.column_stack([x, y])
    tree = cKDTree(points)

    # Find nearest event for each random point
    random_points = np.column_stack([x_random, y_random])
    nn_dist, _ = tree.query(random_points, k=1)

    # Calculate F function
    F = np.zeros(len(distances))
    for i, r in enumerate(distances):
        F[i] = np.mean(nn_dist <= r)

    return F
