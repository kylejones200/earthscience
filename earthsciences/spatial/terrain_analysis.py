"""
Digital Elevation Model (DEM) and terrain analysis

Slope, aspect, curvature, and derived terrain attributes.
"""

import numpy as np
from scipy import ndimage


def slope_aspect(
    dem: np.ndarray, cell_size: float = 1.0, degrees: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate slope and aspect from DEM.

    Parameters
    ----------
    dem : array_like
        Digital Elevation Model (2D array)
    cell_size : float, optional
        Grid cell size (default=1.0)
    degrees : bool, optional
        Return values in degrees (default=True)

    Returns
    -------
    slope : ndarray
        Slope (gradient magnitude)
    aspect : ndarray
        Aspect (direction of steepest descent)

    Examples
    --------
    >>> # Create synthetic DEM
    >>> x = np.linspace(-5, 5, 100)
    >>> y = np.linspace(-5, 5, 100)
    >>> X, Y = np.meshgrid(x, y)
    >>> dem = 100 - 0.5*(X**2 + Y**2)  # Paraboloid
    >>>
    >>> slope, aspect = slope_aspect(dem, cell_size=0.1)
    >>> print(f"Max slope: {np.max(slope):.2f}°")
    """
    dem = np.asarray(dem)

    # Calculate gradients using central differences
    dz_dx = np.gradient(dem, cell_size, axis=1)
    dz_dy = np.gradient(dem, cell_size, axis=0)

    # Slope (gradient magnitude)
    slope = np.sqrt(dz_dx**2 + dz_dy**2)

    if degrees:
        slope = np.rad2deg(np.arctan(slope))
    else:
        slope = np.arctan(slope)

    # Aspect (direction)
    aspect = np.arctan2(-dz_dy, dz_dx)

    # Convert aspect to compass bearing (0 = North, clockwise)
    aspect = np.pi / 2 - aspect
    aspect = np.where(aspect < 0, aspect + 2 * np.pi, aspect)

    if degrees:
        aspect = np.rad2deg(aspect)

    return slope, aspect


def curvature(dem: np.ndarray, cell_size: float = 1.0) -> dict:
    """
    Calculate surface curvatures from DEM.

    Parameters
    ----------
    dem : array_like
        Digital Elevation Model
    cell_size : float, optional
        Grid cell size (default=1.0)

    Returns
    -------
    dict
        Dictionary with profile, planform, and mean curvature

    Examples
    --------
    >>> dem = np.random.randn(50, 50).cumsum(axis=0).cumsum(axis=1)
    >>> curv = curvature(dem, cell_size=1.0)
    >>> print(f"Mean curvature range: [{np.min(curv['mean']):.3f}, "
    ...       f"{np.max(curv['mean']):.3f}]")
    """
    dem = np.asarray(dem)

    # First derivatives
    dz_dx = np.gradient(dem, cell_size, axis=1)
    dz_dy = np.gradient(dem, cell_size, axis=0)

    # Second derivatives
    d2z_dx2 = np.gradient(dz_dx, cell_size, axis=1)
    d2z_dy2 = np.gradient(dz_dy, cell_size, axis=0)
    d2z_dxdy = np.gradient(dz_dx, cell_size, axis=0)

    # Gradient magnitude squared
    p = dz_dx
    q = dz_dy
    p2_q2 = p**2 + q**2

    # Profile curvature (curvature in direction of max slope)
    with np.errstate(divide="ignore", invalid="ignore"):
        profile_curv = -(p**2 * d2z_dx2 + 2 * p * q * d2z_dxdy + q**2 * d2z_dy2) / (
            p2_q2 * np.sqrt(1 + p2_q2) ** 3
        )
        profile_curv = np.where(np.isfinite(profile_curv), profile_curv, 0)

    # Planform curvature (curvature perpendicular to max slope)
    with np.errstate(divide="ignore", invalid="ignore"):
        plan_curv = -(q**2 * d2z_dx2 - 2 * p * q * d2z_dxdy + p**2 * d2z_dy2) / (
            p2_q2 * np.sqrt(1 + p2_q2)
        )
        plan_curv = np.where(np.isfinite(plan_curv), plan_curv, 0)

    # Mean curvature
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_curv = -((1 + q**2) * d2z_dx2 - 2 * p * q * d2z_dxdy + (1 + p**2) * d2z_dy2) / (
            2 * (1 + p2_q2) ** 1.5
        )
        mean_curv = np.where(np.isfinite(mean_curv), mean_curv, 0)

    return {
        "profile": profile_curv,
        "planform": plan_curv,
        "mean": mean_curv,
    }


def hillshade(
    dem: np.ndarray, azimuth: float = 315.0, altitude: float = 45.0, cell_size: float = 1.0
) -> np.ndarray:
    """
    Calculate hillshade for visualization.

    Parameters
    ----------
    dem : array_like
        Digital Elevation Model
    azimuth : float, optional
        Sun azimuth angle in degrees (default=315, NW)
    altitude : float, optional
        Sun altitude angle in degrees (default=45)
    cell_size : float, optional
        Grid cell size (default=1.0)

    Returns
    -------
    ndarray
        Hillshade values (0-255)

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> dem = np.random.randn(100, 100).cumsum(axis=0).cumsum(axis=1)
    >>> hs = hillshade(dem, azimuth=315, altitude=45)
    >>> plt.imshow(hs, cmap='gray')
    >>> plt.title('Hillshade')
    >>> plt.show()
    """
    dem = np.asarray(dem)

    # Calculate slope and aspect
    slope, aspect = slope_aspect(dem, cell_size, degrees=False)

    # Convert angles to radians
    azimuth_rad = np.deg2rad(azimuth)
    altitude_rad = np.deg2rad(altitude)

    # Calculate hillshade
    hillshade_val = np.cos(altitude_rad) * np.cos(slope) + np.sin(altitude_rad) * np.sin(
        slope
    ) * np.cos(azimuth_rad - aspect)

    # Scale to 0-255
    hillshade_val = np.clip(hillshade_val, 0, 1)
    hillshade_8bit = (hillshade_val * 255).astype(np.uint8)

    return hillshade_8bit


def topographic_wetness_index(dem: np.ndarray, cell_size: float = 1.0) -> np.ndarray:
    """
    Calculate Topographic Wetness Index (TWI).

    TWI = ln(a / tan(β))
    where a is upslope contributing area and β is slope

    Parameters
    ----------
    dem : array_like
        Digital Elevation Model
    cell_size : float, optional
        Grid cell size (default=1.0)

    Returns
    -------
    ndarray
        Topographic Wetness Index

    Examples
    --------
    >>> dem = np.random.randn(50, 50).cumsum(axis=0).cumsum(axis=1)
    >>> twi = topographic_wetness_index(dem)
    >>> print(f"TWI range: [{np.min(twi):.2f}, {np.max(twi):.2f}]")
    """
    dem = np.asarray(dem)

    # Calculate slope
    slope, _ = slope_aspect(dem, cell_size, degrees=False)

    # Calculate flow accumulation (simplified - assumes D8 flow)
    flow_accum = calculate_flow_accumulation(dem, cell_size)

    # Contributing area
    contributing_area = flow_accum * cell_size**2

    # TWI calculation
    with np.errstate(divide="ignore", invalid="ignore"):
        tan_slope = np.tan(slope)
        tan_slope = np.where(tan_slope < 0.001, 0.001, tan_slope)  # Avoid division by zero
        twi = np.log(contributing_area / tan_slope)
        twi = np.where(np.isfinite(twi), twi, 0)

    return twi


def calculate_flow_accumulation(dem: np.ndarray, cell_size: float = 1.0) -> np.ndarray:
    """
    Calculate flow accumulation using D8 algorithm (simplified).

    Parameters
    ----------
    dem : array_like
        Digital Elevation Model
    cell_size : float, optional
        Grid cell size

    Returns
    -------
    ndarray
        Flow accumulation (number of cells draining to each cell)

    Examples
    --------
    >>> dem = np.random.randn(30, 30).cumsum(axis=0).cumsum(axis=1)
    >>> flow_accum = calculate_flow_accumulation(dem)
    >>> print(f"Max accumulation: {np.max(flow_accum)}")
    """
    dem = np.asarray(dem)
    rows, cols = dem.shape

    # Initialize flow accumulation
    flow_accum = np.ones_like(dem)

    # Get elevation order (process from highest to lowest)
    order = np.unravel_index(np.argsort(dem.ravel())[::-1], dem.shape)

    # D8 flow directions (8 neighbors)
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    # Process each cell from highest to lowest
    for i, j in zip(order[0], order[1]):
        if flow_accum[i, j] == 0:
            continue

        # Find steepest descent
        max_slope = 0
        flow_i, flow_j = None, None

        for di, dj in directions:
            ni, nj = i + di, j + dj

            # Check bounds
            if 0 <= ni < rows and 0 <= nj < cols:
                slope = (dem[i, j] - dem[ni, nj]) / cell_size
                if slope > max_slope:
                    max_slope = slope
                    flow_i, flow_j = ni, nj

        # Route flow to steepest neighbor
        if flow_i is not None:
            flow_accum[flow_i, flow_j] += flow_accum[i, j]

    return flow_accum


def stream_power_index(dem: np.ndarray, cell_size: float = 1.0) -> np.ndarray:
    """
    Calculate Stream Power Index (SPI).

    SPI = As × tan(β)
    where As is specific catchment area and β is slope

    Parameters
    ----------
    dem : array_like
        Digital Elevation Model
    cell_size : float, optional
        Grid cell size

    Returns
    -------
    ndarray
        Stream Power Index

    Examples
    --------
    >>> dem = np.random.randn(50, 50).cumsum(axis=0).cumsum(axis=1)
    >>> spi = stream_power_index(dem)
    >>> print(f"High erosion potential: {np.sum(spi > np.percentile(spi, 90))}")
    """
    dem = np.asarray(dem)

    # Calculate slope
    slope, _ = slope_aspect(dem, cell_size, degrees=False)

    # Calculate flow accumulation
    flow_accum = calculate_flow_accumulation(dem, cell_size)

    # Specific catchment area
    specific_area = flow_accum * cell_size

    # SPI
    tan_slope = np.tan(slope)
    spi = specific_area * tan_slope

    return spi


def terrain_ruggedness_index(dem: np.ndarray, window_size: int = 3) -> np.ndarray:
    """
    Calculate Terrain Ruggedness Index (TRI).

    TRI = mean absolute difference between center cell and neighbors

    Parameters
    ----------
    dem : array_like
        Digital Elevation Model
    window_size : int, optional
        Window size (default=3)

    Returns
    -------
    ndarray
        Terrain Ruggedness Index

    Examples
    --------
    >>> dem = np.random.randn(50, 50).cumsum(axis=0).cumsum(axis=1)
    >>> tri = terrain_ruggedness_index(dem)
    >>> print(f"Mean ruggedness: {np.mean(tri):.3f}")
    """
    dem = np.asarray(dem)

    # Calculate mean elevation in window
    mean_elevation = ndimage.uniform_filter(dem, size=window_size, mode="reflect")

    # Absolute difference from mean
    tri = np.abs(dem - mean_elevation)

    return tri


def topographic_position_index(
    dem: np.ndarray, inner_radius: int = 1, outer_radius: int = 5
) -> np.ndarray:
    """
    Calculate Topographic Position Index (TPI).

    TPI = elevation - mean elevation in annulus
    Positive = ridges, Negative = valleys

    Parameters
    ----------
    dem : array_like
        Digital Elevation Model
    inner_radius : int, optional
        Inner radius of annulus (default=1)
    outer_radius : int, optional
        Outer radius of annulus (default=5)

    Returns
    -------
    ndarray
        Topographic Position Index

    Examples
    --------
    >>> dem = np.random.randn(50, 50).cumsum(axis=0).cumsum(axis=1)
    >>> tpi = topographic_position_index(dem)
    >>> ridges = tpi > np.percentile(tpi, 90)
    >>> valleys = tpi < np.percentile(tpi, 10)
    >>> print(f"Ridges: {np.sum(ridges)}, Valleys: {np.sum(valleys)}")
    """
    dem = np.asarray(dem)

    # Create annulus kernel
    y, x = np.ogrid[-outer_radius : outer_radius + 1, -outer_radius : outer_radius + 1]
    dist = np.sqrt(x**2 + y**2)
    kernel = ((dist >= inner_radius) & (dist <= outer_radius)).astype(float)
    kernel /= np.sum(kernel)  # Normalize

    # Calculate mean elevation in annulus
    mean_elevation = ndimage.convolve(dem, kernel, mode="reflect")

    # TPI
    tpi = dem - mean_elevation

    return tpi
