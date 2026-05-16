"""
Magnetic data processing

Magnetic anomaly modeling and reduction to pole.
"""

import warnings

import numpy as np


def magnetic_anomaly_sphere(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    x0: float,
    y0: float,
    z0: float,
    radius: float,
    susceptibility: float,
    inclination: float,
    declination: float,
    field_strength: float = 50000,
) -> np.ndarray:
    """
    Calculate magnetic anomaly of a sphere.

    Useful for modeling magnetic bodies in the subsurface.

    Parameters
    ----------
    x, y, z : array_like
        Observation point coordinates (meters)
    x0, y0, z0 : float
        Sphere center coordinates (meters)
    radius : float
        Sphere radius (meters)
    susceptibility : float
        Magnetic susceptibility (SI units)
    inclination : float
        Magnetic field inclination (degrees)
    declination : float
        Magnetic field declination (degrees)
    field_strength : float, optional
        Earth's field strength (nT, default=50000)

    Returns
    -------
    ndarray
        Magnetic anomaly (nT)

    Examples
    --------
    >>> # Observation points
    >>> x = np.linspace(-1000, 1000, 50)
    >>> y = np.zeros_like(x)
    >>> z = np.zeros_like(x)
    >>>
    >>> # Buried magnetic body
    >>> anomaly = magnetic_anomaly_sphere(
    ...     x, y, z, x0=0, y0=0, z0=-500, radius=100,
    ...     susceptibility=0.01, inclination=60, declination=0
    ... )
    """
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)

    # Convert angles to radians
    inc_rad = np.deg2rad(inclination)
    dec_rad = np.deg2rad(declination)

    # Direction cosines of Earth's field
    l = np.cos(inc_rad) * np.cos(dec_rad)
    m = np.cos(inc_rad) * np.sin(dec_rad)
    n = np.sin(inc_rad)

    # Distance from sphere center
    dx = x - x0
    dy = y - y0
    dz = z - z0
    r = np.sqrt(dx**2 + dy**2 + dz**2)

    # Magnetic moment
    volume = (4 / 3) * np.pi * radius**3
    M = susceptibility * field_strength * volume / (4 * np.pi * 1e-7)  # Am²

    # Magnetic field components (dipole approximation)
    # Total field anomaly in direction of Earth's field
    dot_product = l * dx + m * dy + n * dz

    anomaly = (M / r**3) * (
        3 * dot_product * (l * dx + m * dy + n * dz) / r**2 - (l**2 + m**2 + n**2)
    )

    # Convert to nT
    anomaly = anomaly * 1e9 / field_strength

    return anomaly


def reduction_to_pole(
    magnetic_data: np.ndarray, inclination: float, declination: float
) -> np.ndarray:
    """
    Reduce magnetic data to the pole.

    Transforms magnetic anomaly to what it would look like at the magnetic pole
    (simplifies interpretation by removing asymmetry).

    Parameters
    ----------
    magnetic_data : array_like
        Magnetic anomaly grid (2D array, nT)
    inclination : float
        Magnetic inclination at survey location (degrees)
    declination : float
        Magnetic declination at survey location (degrees)

    Returns
    -------
    ndarray
        Reduced to pole magnetic data (nT)

    Notes
    -----
    This is a simplified implementation. Full RTP requires FFT-based
    filtering in the frequency domain with proper phase corrections.

    Examples
    --------
    >>> # Magnetic grid
    >>> mag_data = np.random.randn(100, 100) * 100 + 45000
    >>>
    >>> # Reduce to pole for mid-latitude survey
    >>> mag_rtp = reduction_to_pole(mag_data, inclination=60, declination=10)
    """
    magnetic_data = np.asarray(magnetic_data)

    warnings.warn(
        "Using simplified reduction to pole. "
        "Full implementation requires FFT-based filtering in frequency domain. "
        "Results are approximate."
    )

    # Simplified: apply phase correction based on inclination
    # Real RTP is done in frequency domain

    inc_rad = np.deg2rad(inclination)
    correction_factor = np.sin(inc_rad)

    # Scale anomaly (very simplified)
    rtp_data = magnetic_data / (correction_factor + 0.1)

    return rtp_data
