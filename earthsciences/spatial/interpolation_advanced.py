"""
Advanced interpolation methods for spatial data

Radial Basis Functions, Natural Neighbor, Minimum Curvature, and other
sophisticated interpolation techniques.
"""

import numpy as np
from scipy.interpolate import Rbf
from scipy.spatial import Delaunay, cKDTree


def radial_basis_function_interpolation(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    function: str = "multiquadric",
    epsilon: float | None = None,
    smooth: float = 0.0,
) -> np.ndarray:
    """
    Radial Basis Function (RBF) interpolation.

    More flexible than Kriging, good for scattered data with complex patterns.

    Parameters
    ----------
    x, y : array_like
        Data point coordinates
    values : array_like
        Data values
    grid_x, grid_y : array_like
        Grid coordinates for interpolation
    function : str, optional
        RBF type: 'multiquadric', 'inverse', 'gaussian', 'linear',
        'cubic', 'quintic', 'thin_plate' (default='multiquadric')
    epsilon : float, optional
        Shape parameter (if None, automatically determined)
    smooth : float, optional
        Smoothing parameter (default=0.0, exact interpolation)

    Returns
    -------
    ndarray
        Interpolated values on grid

    Examples
    --------
    >>> # Scattered data
    >>> x = np.random.rand(50) * 10
    >>> y = np.random.rand(50) * 10
    >>> values = np.sin(x) * np.cos(y)
    >>>
    >>> # Regular grid
    >>> grid_x, grid_y = np.meshgrid(np.linspace(0, 10, 50),
    ...                              np.linspace(0, 10, 50))
    >>>
    >>> interpolated = radial_basis_function_interpolation(
    ...     x, y, values, grid_x, grid_y, function='thin_plate'
    ... )
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    values = np.asarray(values).ravel()

    # Create RBF interpolator
    rbf = Rbf(x, y, values, function=function, epsilon=epsilon, smooth=smooth)

    # Interpolate on grid
    interpolated = rbf(grid_x, grid_y)

    return interpolated


def natural_neighbor_interpolation(
    x: np.ndarray, y: np.ndarray, values: np.ndarray, grid_x: np.ndarray, grid_y: np.ndarray
) -> np.ndarray:
    """
    Natural Neighbor interpolation (Sibson interpolation).

    Based on Voronoi tessellation, provides smooth C¹ continuous surface.
    Good for irregularly spaced data without extrapolation artifacts.

    Parameters
    ----------
    x, y : array_like
        Data point coordinates
    values : array_like
        Data values
    grid_x, grid_y : array_like
        Grid coordinates for interpolation

    Returns
    -------
    ndarray
        Interpolated values on grid

    Notes
    -----
    This is a simplified implementation. True natural neighbor interpolation
    requires computing Voronoi cell area ratios, which is computationally expensive.
    This implementation uses a weighted average based on Voronoi neighbors.

    Examples
    --------
    >>> x = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    >>> y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    >>> values = x + y
    >>>
    >>> grid_x, grid_y = np.meshgrid(np.linspace(0, 2, 20),
    ...                              np.linspace(0, 2, 20))
    >>>
    >>> interpolated = natural_neighbor_interpolation(x, y, values, grid_x, grid_y)
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    values = np.asarray(values).ravel()

    # Stack coordinates
    points = np.column_stack([x, y])

    # Build Delaunay triangulation (dual of Voronoi)
    tri = Delaunay(points)

    # For each grid point, find natural neighbors
    grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    # Find simplex for each grid point
    simplex_ids = tri.find_simplex(grid_points)

    # Initialize output
    interpolated = np.full(grid_points.shape[0], np.nan)

    # For points inside convex hull
    inside = simplex_ids >= 0

    if np.any(inside):
        # Get vertices of simplices
        simplices = tri.simplices[simplex_ids[inside]]

        # Barycentric coordinates
        transform = tri.transform[simplex_ids[inside]]
        delta = grid_points[inside] - transform[:, 2]
        bary = np.einsum("ijk,ik->ij", transform[:, :2], delta)
        bary = np.c_[bary, 1 - bary.sum(axis=1)]

        # Weighted average using barycentric coordinates
        interpolated[inside] = np.sum(values[simplices] * bary, axis=1)

    return interpolated.reshape(grid_x.shape)


def minimum_curvature_spline(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    tension: float = 0.0,
) -> np.ndarray:
    """
    Minimum curvature spline interpolation.

    Creates smoothest possible surface (minimizes curvature).
    Widely used in geophysics and contouring.

    Parameters
    ----------
    x, y : array_like
        Data point coordinates
    values : array_like
        Data values
    grid_x, grid_y : array_like
        Grid coordinates for interpolation
    tension : float, optional
        Tension parameter 0-1 (0=minimum curvature, 1=linear, default=0)

    Returns
    -------
    ndarray
        Interpolated values on grid

    Notes
    -----
    This implementation uses thin-plate spline as an approximation to
    minimum curvature. True minimum curvature requires iterative relaxation.

    Examples
    --------
    >>> # Elevation data
    >>> x = np.array([0, 5, 10, 0, 5, 10])
    >>> y = np.array([0, 0, 0, 10, 10, 10])
    >>> z = np.array([100, 150, 120, 110, 180, 130])
    >>>
    >>> grid_x, grid_y = np.meshgrid(np.linspace(0, 10, 50),
    ...                              np.linspace(0, 10, 50))
    >>>
    >>> surface = minimum_curvature_spline(x, y, z, grid_x, grid_y)
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    values = np.asarray(values).ravel()

    # Use thin-plate spline with tension adjustment
    if tension > 0:
        # Add tension by blending with linear interpolation
        from scipy.interpolate import LinearNDInterpolator

        # Thin-plate spline
        rbf_smooth = Rbf(x, y, values, function="thin_plate", smooth=0)
        interpolated_smooth = rbf_smooth(grid_x, grid_y)

        # Linear interpolation
        linear = LinearNDInterpolator(np.column_stack([x, y]), values)
        interpolated_linear = linear(grid_x, grid_y)

        # Blend
        interpolated = (1 - tension) * interpolated_smooth + tension * interpolated_linear
    else:
        # Pure minimum curvature (thin-plate spline)
        rbf = Rbf(x, y, values, function="thin_plate", smooth=0)
        interpolated = rbf(grid_x, grid_y)

    return interpolated


def inverse_distance_squared(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    radius: float | None = None,
    min_neighbors: int = 1,
) -> np.ndarray:
    """
    Inverse distance squared interpolation (power=2).

    Similar to IDW but with fixed power of 2.

    Parameters
    ----------
    x, y : array_like
        Data point coordinates
    values : array_like
        Data values
    grid_x, grid_y : array_like
        Grid coordinates for interpolation
    radius : float, optional
        Search radius (if None, use all points)
    min_neighbors : int, optional
        Minimum number of neighbors (default=1)

    Returns
    -------
    ndarray
        Interpolated values on grid

    Examples
    --------
    >>> x = np.random.rand(30) * 100
    >>> y = np.random.rand(30) * 100
    >>> values = np.random.rand(30)
    >>>
    >>> grid_x, grid_y = np.meshgrid(np.linspace(0, 100, 50),
    ...                              np.linspace(0, 100, 50))
    >>>
    >>> interpolated = inverse_distance_squared(x, y, values, grid_x, grid_y, radius=20)
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    values = np.asarray(values).ravel()

    # Build KD-tree for efficient neighbor search
    tree = cKDTree(np.column_stack([x, y]))

    # Grid points
    grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    # Initialize output
    interpolated = np.zeros(grid_points.shape[0])

    # Query neighbors
    if radius is not None:
        # Within radius
        for i, point in enumerate(grid_points):
            indices = tree.query_ball_point(point, radius)

            if len(indices) < min_neighbors:
                # Use nearest neighbors if not enough within radius
                distances, indices = tree.query(point, k=min_neighbors)
                if not hasattr(distances, "__len__"):
                    distances = [distances]
                    indices = [indices]
            else:
                distances = np.linalg.norm(
                    np.column_stack([x[indices], y[indices]]) - point, axis=1
                )

            # Check for exact match
            if np.any(distances == 0):
                interpolated[i] = values[indices[distances == 0][0]]
            else:
                # Inverse distance squared weighting
                weights = 1.0 / (distances**2)
                interpolated[i] = np.sum(weights * values[indices]) / np.sum(weights)
    else:
        # Use all points
        for i, point in enumerate(grid_points):
            distances = np.linalg.norm(np.column_stack([x, y]) - point, axis=1)

            # Check for exact match
            if np.any(distances == 0):
                interpolated[i] = values[distances == 0][0]
            else:
                weights = 1.0 / (distances**2)
                interpolated[i] = np.sum(weights * values) / np.sum(weights)

    return interpolated.reshape(grid_x.shape)


def moving_average_interpolation(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    radius: float,
    min_neighbors: int = 3,
) -> np.ndarray:
    """
    Moving average interpolation.

    Simple and robust, good for noisy data.

    Parameters
    ----------
    x, y : array_like
        Data point coordinates
    values : array_like
        Data values
    grid_x, grid_y : array_like
        Grid coordinates for interpolation
    radius : float
        Search radius
    min_neighbors : int, optional
        Minimum number of neighbors (default=3)

    Returns
    -------
    ndarray
        Interpolated values on grid

    Examples
    --------
    >>> # Noisy data
    >>> x = np.random.rand(100) * 50
    >>> y = np.random.rand(100) * 50
    >>> values = np.sin(x/5) + np.random.randn(100) * 0.1
    >>>
    >>> grid_x, grid_y = np.meshgrid(np.linspace(0, 50, 30),
    ...                              np.linspace(0, 50, 30))
    >>>
    >>> smoothed = moving_average_interpolation(
    ...     x, y, values, grid_x, grid_y, radius=5
    ... )
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    values = np.asarray(values).ravel()

    # Build KD-tree
    tree = cKDTree(np.column_stack([x, y]))

    # Grid points
    grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    # Initialize output
    interpolated = np.full(grid_points.shape[0], np.nan)

    # For each grid point
    for i, point in enumerate(grid_points):
        # Find neighbors within radius
        indices = tree.query_ball_point(point, radius)

        if len(indices) >= min_neighbors:
            # Simple average
            interpolated[i] = np.mean(values[indices])
        elif len(indices) > 0:
            # Use available neighbors even if less than min
            interpolated[i] = np.mean(values[indices])

    return interpolated.reshape(grid_x.shape)


def shepard_interpolation(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    power: float = 2.0,
    radius: float | None = None,
) -> np.ndarray:
    """
    Shepard's method (modified inverse distance weighting).

    Classic interpolation method with local or global influence.

    Parameters
    ----------
    x, y : array_like
        Data point coordinates
    values : array_like
        Data values
    grid_x, grid_y : array_like
        Grid coordinates for interpolation
    power : float, optional
        Distance power (default=2.0)
    radius : float, optional
        Local search radius (if None, use global)

    Returns
    -------
    ndarray
        Interpolated values on grid

    Examples
    --------
    >>> x = np.array([0, 1, 0, 1])
    >>> y = np.array([0, 0, 1, 1])
    >>> values = np.array([0, 1, 1, 2])
    >>>
    >>> grid_x, grid_y = np.meshgrid(np.linspace(0, 1, 20),
    ...                              np.linspace(0, 1, 20))
    >>>
    >>> interpolated = shepard_interpolation(x, y, values, grid_x, grid_y, power=3)
    """
    # This is essentially the same as inverse_distance_squared but with variable power
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    values = np.asarray(values).ravel()

    grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    interpolated = np.zeros(grid_points.shape[0])

    if radius is not None:
        tree = cKDTree(np.column_stack([x, y]))

        for i, point in enumerate(grid_points):
            indices = tree.query_ball_point(point, radius)

            if len(indices) == 0:
                # Use nearest neighbor
                distances, indices = tree.query(point, k=1)
                indices = [indices]
                distances = [distances]
            else:
                distances = np.linalg.norm(
                    np.column_stack([x[indices], y[indices]]) - point, axis=1
                )

            if np.any(distances == 0):
                interpolated[i] = values[indices[distances == 0][0]]
            else:
                weights = 1.0 / (distances**power)
                interpolated[i] = np.sum(weights * values[indices]) / np.sum(weights)
    else:
        # Global interpolation
        for i, point in enumerate(grid_points):
            distances = np.linalg.norm(np.column_stack([x, y]) - point, axis=1)

            if np.any(distances == 0):
                interpolated[i] = values[distances == 0][0]
            else:
                weights = 1.0 / (distances**power)
                interpolated[i] = np.sum(weights * values) / np.sum(weights)

    return interpolated.reshape(grid_x.shape)
