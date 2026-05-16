"""
Seismic data processing

NMO correction, velocity analysis, and traveltime calculation.
"""

import numpy as np
from scipy.ndimage import distance_transform_edt


def nmo_correction(traveltime: np.ndarray, offset: np.ndarray, velocity: float) -> np.ndarray:
    """
    Normal moveout (NMO) correction for seismic data.

    Corrects for traveltime increase with source-receiver offset.

    Parameters
    ----------
    traveltime : array_like
        Two-way traveltime (seconds)
    offset : array_like
        Source-receiver offset (meters)
    velocity : float
        RMS velocity (m/s)

    Returns
    -------
    ndarray
        Zero-offset traveltime (seconds)

    Examples
    --------
    >>> # Seismic reflection
    >>> t = np.array([1.0, 1.1, 1.3, 1.6])  # seconds
    >>> x = np.array([0, 500, 1000, 1500])  # meters offset
    >>> v = 2000  # m/s
    >>>
    >>> t0 = nmo_correction(t, x, v)
    >>> print(f"Zero-offset time: {t0}")
    """
    traveltime = np.asarray(traveltime)
    offset = np.asarray(offset)

    # Hyperbolic moveout equation
    # t² = t0² + x²/v²
    # Solve for t0
    t0_squared = traveltime**2 - (offset**2 / velocity**2)

    # Handle negative values (due to noise or incorrect velocity)
    t0_squared = np.maximum(t0_squared, 0)

    t0 = np.sqrt(t0_squared)

    return t0


def stacking_velocity_analysis(
    gathers: np.ndarray,
    offsets: np.ndarray,
    times: np.ndarray,
    velocity_range: tuple[float, float] = (1500, 4000),
    n_velocities: int = 50,
) -> dict:
    """
    Velocity analysis using semblance or coherency.

    Finds optimal stacking velocity for seismic data.

    Parameters
    ----------
    gathers : array_like
        Common midpoint gather (2D: time x offset)
    offsets : array_like
        Source-receiver offsets (meters)
    times : array_like
        Two-way traveltimes (seconds)
    velocity_range : tuple, optional
        (vmin, vmax) for velocity search (m/s, default=(1500, 4000))
    n_velocities : int, optional
        Number of test velocities (default=50)

    Returns
    -------
    dict
        Optimal velocities and semblance panel

    Examples
    --------
    >>> # CMP gather (simplified)
    >>> n_times, n_offsets = 100, 20
    >>> gathers = np.random.randn(n_times, n_offsets)
    >>> offsets = np.linspace(0, 2000, n_offsets)
    >>> times = np.linspace(0, 2, n_times)
    >>>
    >>> result = stacking_velocity_analysis(gathers, offsets, times)
    >>> print(f"Optimal velocity: {result['best_velocity']:.0f} m/s")
    """
    gathers = np.asarray(gathers)
    offsets = np.asarray(offsets)
    times = np.asarray(times)

    # Test velocities
    velocities = np.linspace(velocity_range[0], velocity_range[1], n_velocities)

    # Semblance panel
    semblance = np.zeros((len(times), len(velocities)))

    # For each time and velocity, calculate semblance
    for i, t0 in enumerate(times):
        for j, v in enumerate(velocities):
            # Calculate moveout-corrected times
            t_nmo = np.sqrt(t0**2 + (offsets**2 / v**2))

            # Extract amplitudes at these times (simplified interpolation)
            # In real implementation, would use proper interpolation
            time_indices = np.clip(
                (t_nmo / times[-1] * (len(times) - 1)).astype(int), 0, len(times) - 1
            )

            amplitudes = gathers[time_indices, np.arange(len(offsets))]

            # Semblance = (sum of amplitudes)² / (N * sum of amplitude²)
            numerator = np.sum(amplitudes) ** 2
            denominator = len(amplitudes) * np.sum(amplitudes**2)

            if denominator > 0:
                semblance[i, j] = numerator / denominator

    # Find maximum semblance at each time
    best_velocity_indices = np.argmax(semblance, axis=1)
    best_velocities = velocities[best_velocity_indices]

    return {
        "semblance_panel": semblance,
        "velocities": velocities,
        "times": times,
        "best_velocities": best_velocities,
        "best_velocity": np.median(best_velocities),
    }


def eikonal_traveltime(
    velocity_model: np.ndarray, source_x: int, source_y: int, dx: float = 1.0, dy: float = 1.0
) -> np.ndarray:
    """
    Calculate first-arrival traveltimes using Eikonal equation.

    Fast marching method for seismic traveltime calculation.

    Parameters
    ----------
    velocity_model : array_like
        2D velocity model (m/s)
    source_x, source_y : int
        Source location indices
    dx, dy : float, optional
        Grid spacing (meters, default=1.0)

    Returns
    -------
    ndarray
        Traveltime field (seconds)

    Notes
    -----
    This is a simplified implementation. Full fast marching requires
    heap-based priority queue and careful updating.

    Examples
    --------
    >>> # Velocity model
    >>> vel = np.ones((50, 50)) * 2000  # m/s
    >>> vel[25:, :] = 3000  # Layer interface
    >>>
    >>> # Calculate traveltimes from center
    >>> times = eikonal_traveltime(vel, 25, 25, dx=10, dy=10)
    """
    velocity_model = np.asarray(velocity_model)

    # Simplified: use distance transform with slowness weighting
    # Real fast marching is more sophisticated

    slowness = 1.0 / velocity_model

    # Create source mask
    source_mask = np.zeros_like(velocity_model, dtype=bool)
    source_mask[source_x, source_y] = True

    # Distance transform
    distances = distance_transform_edt(~source_mask, sampling=(dx, dy))

    # Approximate traveltime (simplified)
    traveltimes = distances * np.mean(slowness)

    return traveltimes
