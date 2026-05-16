"""
Spatial interpolation methods

Various interpolation techniques for scattered spatial data.
"""

import warnings

import numpy as np
from scipy import interpolate
from scipy.interpolate import RBFInterpolator
from scipy.spatial import cKDTree


def idw_interpolation(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    power: float = 2.0,
    radius: float | None = None,
    min_points: int = 1,
) -> np.ndarray:
    """
    Inverse Distance Weighting (IDW) interpolation.

    Weights values by inverse of distance raised to a power.

    Parameters
    ----------
    x, y : array_like
        Coordinates of known points
    values : array_like
        Values at known points
    grid_x, grid_y : array_like
        Grid coordinates for interpolation
    power : float, optional
        Power parameter (default=2.0, higher = more local)
    radius : float, optional
        Search radius (if None, use all points)
    min_points : int, optional
        Minimum number of points for interpolation (default=1)

    Returns
    -------
    ndarray
        Interpolated values on grid

    Examples
    --------
    >>> # Random points
    >>> np.random.seed(42)
    >>> x = np.random.rand(50) * 10
    >>> y = np.random.rand(50) * 10
    >>> values = np.sin(x) + np.cos(y)
    >>>
    >>> # Create grid
    >>> grid_x, grid_y = np.meshgrid(
    ...     np.linspace(0, 10, 50),
    ...     np.linspace(0, 10, 50)
    ... )
    >>>
    >>> # Interpolate
    >>> grid_values = idw_interpolation(x, y, values, grid_x, grid_y)
    """
    x = np.asarray(x)
    y = np.asarray(y)
    values = np.asarray(values)

    # Flatten grid coordinates
    grid_x_flat = grid_x.ravel()
    grid_y_flat = grid_y.ravel()

    # Build KD-tree for efficient nearest neighbor search
    points = np.column_stack([x, y])
    tree = cKDTree(points)

    # Query points
    query_points = np.column_stack([grid_x_flat, grid_y_flat])

    # Initialize result
    result = np.zeros(len(query_points))

    for i, query in enumerate(query_points):
        if radius is None:
            # Use all points
            distances = np.sqrt((x - query[0]) ** 2 + (y - query[1]) ** 2)
            mask = distances > 0  # Exclude exact matches
        else:
            # Find points within radius
            indices = tree.query_ball_point(query, radius)
            if len(indices) < min_points:
                # Fallback: use nearest min_points
                distances, indices = tree.query(query, k=min_points)
            else:
                distances = np.sqrt((x[indices] - query[0]) ** 2 + (y[indices] - query[1]) ** 2)
                mask = distances > 0
                indices = np.array(indices)[mask]
                distances = distances[mask]

        if len(distances) == 0:
            # No valid points, use nearest
            dist, idx = tree.query(query, k=1)
            result[i] = values[idx]
        else:
            if radius is None:
                valid_values = values[mask]
                valid_distances = distances[mask]
            else:
                valid_values = values[indices]
                valid_distances = distances

            # IDW weights
            weights = 1.0 / (valid_distances**power)
            weights /= np.sum(weights)

            # Weighted average
            result[i] = np.sum(weights * valid_values)

    # Reshape to grid
    return result.reshape(grid_x.shape)


def nearest_neighbor(
    x: np.ndarray, y: np.ndarray, values: np.ndarray, grid_x: np.ndarray, grid_y: np.ndarray
) -> np.ndarray:
    """
    Nearest neighbor interpolation.

    Assigns each grid point the value of the nearest data point.

    Parameters
    ----------
    x, y : array_like
        Coordinates of known points
    values : array_like
        Values at known points
    grid_x, grid_y : array_like
        Grid coordinates for interpolation

    Returns
    -------
    ndarray
        Interpolated values on grid

    Examples
    --------
    >>> x = np.array([0, 1, 2])
    >>> y = np.array([0, 1, 2])
    >>> values = np.array([1, 2, 3])
    >>> grid_x, grid_y = np.meshgrid(np.linspace(0, 2, 10),
    ...                              np.linspace(0, 2, 10))
    >>> result = nearest_neighbor(x, y, values, grid_x, grid_y)
    """
    x = np.asarray(x)
    y = np.asarray(y)
    values = np.asarray(values)

    # Build KD-tree
    points = np.column_stack([x, y])
    tree = cKDTree(points)

    # Query grid points
    grid_x_flat = grid_x.ravel()
    grid_y_flat = grid_y.ravel()
    query_points = np.column_stack([grid_x_flat, grid_y_flat])

    # Find nearest neighbors
    distances, indices = tree.query(query_points, k=1)

    # Get values
    result = values[indices]

    return result.reshape(grid_x.shape)


def spline_interpolation(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    smooth: float = 0.0,
) -> np.ndarray:
    """
    Spline interpolation for scattered data.

    Uses radial basis functions (thin-plate splines).

    Parameters
    ----------
    x, y : array_like
        Coordinates of known points
    values : array_like
        Values at known points
    grid_x, grid_y : array_like
        Grid coordinates for interpolation
    smooth : float, optional
        Smoothing parameter (0 = exact interpolation) (default=0.0)

    Returns
    -------
    ndarray
        Interpolated values on grid

    Examples
    --------
    >>> x = np.random.rand(20) * 10
    >>> y = np.random.rand(20) * 10
    >>> values = np.sin(x) * np.cos(y)
    >>> grid_x, grid_y = np.meshgrid(np.linspace(0, 10, 50),
    ...                              np.linspace(0, 10, 50))
    >>> result = spline_interpolation(x, y, values, grid_x, grid_y)
    """
    x = np.asarray(x)
    y = np.asarray(y)
    values = np.asarray(values)

    # Create spline interpolator
    rbf = interpolate.Rbf(x, y, values, function="thin_plate", smooth=smooth)

    # Evaluate on grid
    result = rbf(grid_x, grid_y)

    return result


def griddata_interpolation(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    method: str = "linear",
) -> np.ndarray:
    """
    General griddata interpolation with multiple methods.

    Parameters
    ----------
    x, y : array_like
        Coordinates of known points
    values : array_like
        Values at known points
    grid_x, grid_y : array_like
        Grid coordinates for interpolation
    method : str, optional
        Interpolation method: 'linear', 'cubic', 'nearest' (default='linear')

    Returns
    -------
    ndarray
        Interpolated values on grid

    Examples
    --------
    >>> x = np.random.rand(30) * 10
    >>> y = np.random.rand(30) * 10
    >>> values = x**2 + y**2
    >>> grid_x, grid_y = np.meshgrid(np.linspace(0, 10, 50),
    ...                              np.linspace(0, 10, 50))
    >>> result = griddata_interpolation(x, y, values, grid_x, grid_y,
    ...                                 method='cubic')
    """
    x = np.asarray(x)
    y = np.asarray(y)
    values = np.asarray(values)

    # Points
    points = np.column_stack([x, y])

    # Grid points
    grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    # Interpolate
    result = interpolate.griddata(points, values, grid_points, method=method)

    return result.reshape(grid_x.shape)


def natural_neighbor(
    x: np.ndarray, y: np.ndarray, values: np.ndarray, grid_x: np.ndarray, grid_y: np.ndarray
) -> np.ndarray:
    """
    Natural neighbor interpolation using Voronoi tesselation.

    Simplified implementation using linear interpolation on Delaunay triangulation.

    Parameters
    ----------
    x, y : array_like
        Coordinates of known points
    values : array_like
        Values at known points
    grid_x, grid_y : array_like
        Grid coordinates for interpolation

    Returns
    -------
    ndarray
        Interpolated values on grid

    Examples
    --------
    >>> x = np.random.rand(25) * 10
    >>> y = np.random.rand(25) * 10
    >>> values = np.sin(x) + np.cos(y)
    >>> grid_x, grid_y = np.meshgrid(np.linspace(0, 10, 40),
    ...                              np.linspace(0, 10, 40))
    >>> result = natural_neighbor(x, y, values, grid_x, grid_y)
    """
    # Use linear griddata as approximation to natural neighbor
    return griddata_interpolation(x, y, values, grid_x, grid_y, method="linear")


def bilinear_interpolation(
    x: np.ndarray, y: np.ndarray, values: np.ndarray, query_x: float, query_y: float
) -> float:
    """
    Bilinear interpolation on regular grid.

    Parameters
    ----------
    x : array_like
        X coordinates of grid (1D, regular spacing)
    y : array_like
        Y coordinates of grid (1D, regular spacing)
    values : array_like
        Values on grid (2D array)
    query_x, query_y : float
        Query point coordinates

    Returns
    -------
    float
        Interpolated value

    Examples
    --------
    >>> x = np.arange(0, 10, 1)
    >>> y = np.arange(0, 10, 1)
    >>> xx, yy = np.meshgrid(x, y)
    >>> values = xx + yy
    >>> result = bilinear_interpolation(x, y, values, 2.5, 3.7)
    """
    x = np.asarray(x)
    y = np.asarray(y)
    values = np.asarray(values)

    # Find bounding indices
    i = np.searchsorted(x, query_x) - 1
    j = np.searchsorted(y, query_y) - 1

    # Handle boundary cases
    i = np.clip(i, 0, len(x) - 2)
    j = np.clip(j, 0, len(y) - 2)

    # Get bounding coordinates and values
    x1, x2 = x[i], x[i + 1]
    y1, y2 = y[j], y[j + 1]

    q11 = values[j, i]
    q12 = values[j + 1, i]
    q21 = values[j, i + 1]
    q22 = values[j + 1, i + 1]

    # Bilinear interpolation
    wx = (query_x - x1) / (x2 - x1)
    wy = (query_y - y1) / (y2 - y1)

    result = (1 - wx) * (1 - wy) * q11 + (1 - wx) * wy * q12 + wx * (1 - wy) * q21 + wx * wy * q22

    return result


def rbf_interpolation(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    kernel: str = "multiquadric",
    epsilon: float | None = None,
    smoothing: float = 0.0,
) -> np.ndarray:
    """
    Radial Basis Function (RBF) interpolation for scattered data.

    Parameters
    ----------
    x, y : array_like
        Coordinates of known data points
    values : array_like
        Values at known points
    grid_x, grid_y : array_like
        Grid coordinates for interpolation
    kernel : str, optional
        RBF kernel: 'multiquadric', 'gaussian', 'thin_plate', 'inverse',
        'linear', 'cubic', 'quintic' (default='multiquadric')
    epsilon : float, optional
        Shape parameter for kernels that require it. If None, a heuristic
        based on the average point spacing is used.
    smoothing : float, optional
        Smoothing parameter — 0 gives exact interpolation (default=0.0)

    Returns
    -------
    ndarray
        Interpolated values on the grid, same shape as grid_x

    Examples
    --------
    >>> np.random.seed(42)
    >>> x = np.random.rand(30) * 10
    >>> y = np.random.rand(30) * 10
    >>> values = np.sin(x) * np.cos(y)
    >>> grid_x, grid_y = np.meshgrid(np.linspace(0, 10, 20),
    ...                              np.linspace(0, 10, 20))
    >>> result = rbf_interpolation(x, y, values, grid_x, grid_y)
    """
    # Kernel name mapping (scipy.interpolate.RBFInterpolator uses different names)
    KERNEL_MAP = {
        "multiquadric": "multiquadric",
        "inverse": "inverse_multiquadric",
        "gaussian": "gaussian",
        "thin_plate": "thin_plate_spline",
        "linear": "linear",
        "cubic": "cubic",
        "quintic": "quintic",
    }

    rbf_kernel = KERNEL_MAP.get(kernel, kernel)

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    values = np.asarray(values, dtype=float)

    points = np.column_stack([x, y])

    if epsilon is None:
        # Heuristic: average nearest-neighbour distance
        from scipy.spatial import cKDTree as _KDTree

        tree = _KDTree(points)
        dists, _ = tree.query(points, k=2)
        epsilon = float(np.mean(dists[:, 1]))
        epsilon = max(epsilon, 1e-6)

    rbf = RBFInterpolator(points, values, kernel=rbf_kernel, epsilon=epsilon, smoothing=smoothing)

    grid_x_flat = grid_x.ravel()
    grid_y_flat = grid_y.ravel()
    query_points = np.column_stack([grid_x_flat, grid_y_flat])

    result = rbf(query_points)
    return result.reshape(grid_x.shape)


def nearest_neighbor_interpolation(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
) -> np.ndarray:
    """Deprecated alias for :func:`nearest_neighbor`."""
    warnings.warn(
        "nearest_neighbor_interpolation is deprecated; use nearest_neighbor",
        DeprecationWarning,
        stacklevel=2,
    )
    return nearest_neighbor(x, y, values, grid_x, grid_y)
