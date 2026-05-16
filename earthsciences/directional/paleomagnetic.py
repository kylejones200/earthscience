"""
Paleomagnetic data analysis

Declination, inclination, VGP analysis, and magnetic reversal detection.
"""

import numpy as np


def geographic_to_magnetic(
    declination: float, inclination: float, latitude: float, longitude: float
) -> tuple[float, float]:
    """
    Convert geographic coordinates to magnetic pole position.

    Parameters
    ----------
    declination : float
        Magnetic declination in degrees
    inclination : float
        Magnetic inclination in degrees
    latitude : float
        Site latitude in degrees
    longitude : float
        Site longitude in degrees

    Returns
    -------
    pole_lat : float
        Pole latitude
    pole_lon : float
        Pole longitude

    Examples
    --------
    >>> dec, inc = 10.0, 60.0
    >>> lat, lon = 45.0, -75.0
    >>> pole_lat, pole_lon = geographic_to_magnetic(dec, inc, lat, lon)
    >>> print(f"Pole: ({pole_lat:.2f}°, {pole_lon:.2f}°)")
    """
    # Convert to radians
    dec_rad = np.deg2rad(declination)
    inc_rad = np.deg2rad(inclination)
    lat_rad = np.deg2rad(latitude)
    lon_rad = np.deg2rad(longitude)

    # Calculate pole latitude
    p = np.arctan(2 * np.tan(inc_rad))  # Colatitude

    # Calculate pole position
    pole_lat_rad = np.arcsin(
        np.sin(lat_rad) * np.cos(p) + np.cos(lat_rad) * np.sin(p) * np.cos(dec_rad)
    )

    # Calculate pole longitude
    beta = np.arcsin(np.sin(p) * np.sin(dec_rad) / np.cos(pole_lat_rad))

    if np.cos(p) < np.sin(lat_rad) * np.sin(pole_lat_rad):
        pole_lon_rad = lon_rad + np.pi - beta
    else:
        pole_lon_rad = lon_rad + beta

    # Convert back to degrees
    pole_lat = np.rad2deg(pole_lat_rad)
    pole_lon = np.rad2deg(pole_lon_rad)

    # Normalize longitude to [-180, 180]
    pole_lon = ((pole_lon + 180) % 360) - 180

    return pole_lat, pole_lon


def virtual_geomagnetic_pole(
    declination: np.ndarray, inclination: np.ndarray, latitude: float, longitude: float
) -> dict:
    """
    Calculate Virtual Geomagnetic Pole (VGP) from paleomagnetic data.

    Parameters
    ----------
    declination : array_like
        Declination values in degrees
    inclination : array_like
        Inclination values in degrees
    latitude : float
        Site latitude
    longitude : float
        Site longitude

    Returns
    -------
    dict
        VGP positions and statistics

    Examples
    --------
    >>> # Sample paleomagnetic data
    >>> dec = np.array([350, 355, 5, 10, 15])
    >>> inc = np.array([58, 62, 60, 65, 63])
    >>> result = virtual_geomagnetic_pole(dec, inc, 45.0, -75.0)
    >>> print(f"Mean VGP: ({result['mean_vgp_lat']:.2f}°, "
    ...       f"{result['mean_vgp_lon']:.2f}°)")
    """
    declination = np.asarray(declination)
    inclination = np.asarray(inclination)

    # Calculate VGP for each measurement
    vgp_lats = np.empty(len(declination), dtype=float)
    vgp_lons = np.empty(len(declination), dtype=float)

    for i, (dec, inc) in enumerate(zip(declination, inclination)):
        lat, lon = geographic_to_magnetic(dec, inc, latitude, longitude)
        vgp_lats[i] = lat
        vgp_lons[i] = lon

    # Calculate mean VGP (Fisher statistics)
    from ..directional.spherical import fisher_distribution, spherical_mean

    # Convert to colatitude/azimuth for spherical mean
    theta = 90 - vgp_lats  # Colatitude
    phi = vgp_lons

    mean_theta, mean_phi = spherical_mean(theta, phi, degrees=True)
    mean_vgp_lat = 90 - mean_theta
    mean_vgp_lon = mean_phi

    # Fisher statistics
    fisher_result = fisher_distribution(theta, phi, degrees=True)

    return {
        "vgp_latitudes": vgp_lats,
        "vgp_longitudes": vgp_lons,
        "mean_vgp_lat": mean_vgp_lat,
        "mean_vgp_lon": mean_vgp_lon,
        "kappa": fisher_result["kappa"],
        "alpha_95": fisher_result["alpha_95"],
    }


def paleomagnetic_pole_position(declination: float, inclination: float) -> tuple[float, float]:
    """
    Calculate pole position from declination and inclination.

    Parameters
    ----------
    declination : float
        Declination in degrees
    inclination : float
        Inclination in degrees

    Returns
    -------
    pole_latitude : float
        Pole latitude
    pole_colatitude : float
        Pole colatitude (distance from site)

    Examples
    --------
    >>> dec, inc = 0.0, 60.0
    >>> pole_lat, colat = paleomagnetic_pole_position(dec, inc)
    >>> print(f"Pole latitude: {pole_lat:.2f}°")
    """
    inc_rad = np.deg2rad(inclination)

    # Dipole equation: tan(I) = 2 * tan(λ)
    # where I is inclination and λ is magnetic latitude
    pole_latitude = np.rad2deg(np.arctan(np.tan(inc_rad) / 2))

    # Colatitude (angular distance from site to pole)
    pole_colatitude = 90 - pole_latitude

    return pole_latitude, pole_colatitude


def inclination_shallowing_correction(observed_inclination: float, f: float = 0.6) -> float:
    """
    Correct for inclination shallowing in sediments.

    Uses the elongation-inclination method.

    Parameters
    ----------
    observed_inclination : float
        Observed inclination in degrees
    f : float, optional
        Flattening factor (default=0.6 for typical sediments)

    Returns
    -------
    float
        Corrected inclination

    Examples
    --------
    >>> obs_inc = 45.0
    >>> corrected_inc = inclination_shallowing_correction(obs_inc, f=0.6)
    >>> print(f"Observed: {obs_inc}°, Corrected: {corrected_inc:.2f}°")
    """
    obs_rad = np.deg2rad(observed_inclination)

    # Correction formula: tan(I_true) = tan(I_obs) / f
    corrected_rad = np.arctan(np.tan(obs_rad) / f)
    corrected_inclination = np.rad2deg(corrected_rad)

    return corrected_inclination


def magnetic_reversal_detection(
    declination: np.ndarray, inclination: np.ndarray, threshold: float = 120.0
) -> dict:
    """
    Detect magnetic reversals in paleomagnetic data.

    Parameters
    ----------
    declination : array_like
        Declination values
    inclination : array_like
        Inclination values
    threshold : float, optional
        Angular threshold for reversal (default=120°)

    Returns
    -------
    dict
        Reversal locations and statistics

    Examples
    --------
    >>> # Simulate data with reversal
    >>> n = 100
    >>> dec = np.concatenate([np.random.normal(0, 5, 50),
    ...                       np.random.normal(180, 5, 50)])
    >>> inc = np.concatenate([np.random.normal(60, 3, 50),
    ...                       np.random.normal(-60, 3, 50)])
    >>> result = magnetic_reversal_detection(dec, inc)
    >>> print(f"Reversals detected: {len(result['reversal_indices'])}")
    """
    declination = np.asarray(declination)
    inclination = np.asarray(inclination)

    # Convert to Cartesian coordinates
    dec_rad = np.deg2rad(declination)
    inc_rad = np.deg2rad(inclination)

    x = np.cos(inc_rad) * np.cos(dec_rad)
    y = np.cos(inc_rad) * np.sin(dec_rad)
    z = np.sin(inc_rad)

    directions = np.column_stack([x, y, z])

    # Calculate angular distance between consecutive measurements
    angular_changes = []
    for i in range(1, len(directions)):
        dot_product = np.dot(directions[i - 1], directions[i])
        dot_product = np.clip(dot_product, -1, 1)  # Numerical stability
        angle = np.rad2deg(np.arccos(dot_product))
        angular_changes.append(angle)

    angular_changes = np.array(angular_changes)

    # Detect reversals (large angular changes)
    reversal_mask = angular_changes > threshold
    reversal_indices = np.where(reversal_mask)[0] + 1

    # Classify polarity
    # Normal: inclination > 0 in northern hemisphere
    # Reversed: inclination < 0
    polarity = np.where(inclination > 0, "Normal", "Reversed")

    return {
        "angular_changes": angular_changes,
        "reversal_indices": reversal_indices,
        "n_reversals": len(reversal_indices),
        "polarity": polarity,
    }


def paleomagnetic_secular_variation(
    declination: np.ndarray, inclination: np.ndarray, ages: np.ndarray
) -> dict:
    """
    Analyze paleomagnetic secular variation.

    Parameters
    ----------
    declination : array_like
        Declination values
    inclination : array_like
        Inclination values
    ages : array_like
        Ages of samples (e.g., in years BP)

    Returns
    -------
    dict
        Secular variation statistics

    Examples
    --------
    >>> ages = np.linspace(0, 10000, 100)
    >>> dec = 10 * np.sin(2*np.pi*ages/5000) + np.random.randn(100)*2
    >>> inc = 60 + 5*np.cos(2*np.pi*ages/5000) + np.random.randn(100)*2
    >>> result = paleomagnetic_secular_variation(dec, inc, ages)
    >>> print(f"PSV: {result['psv']:.2f}°")
    """
    declination = np.asarray(declination)
    inclination = np.asarray(inclination)
    ages = np.asarray(ages)

    # Sort by age
    sort_idx = np.argsort(ages)
    dec_sorted = declination[sort_idx]
    inc_sorted = inclination[sort_idx]

    # Calculate angular standard deviation (PSV measure)
    dec_rad = np.deg2rad(dec_sorted)
    inc_rad = np.deg2rad(inc_sorted)

    # Convert to unit vectors
    x = np.cos(inc_rad) * np.cos(dec_rad)
    y = np.cos(inc_rad) * np.sin(dec_rad)
    z = np.sin(inc_rad)

    # Mean direction
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    z_mean = np.mean(z)

    R = np.sqrt(x_mean**2 + y_mean**2 + z_mean**2)

    # Angular standard deviation
    from ..directional.spherical import spherical_variance

    # Convert back to spherical
    theta = 90 - np.rad2deg(np.arcsin(z))
    phi = np.rad2deg(np.arctan2(y, x))

    var = spherical_variance(theta, phi, degrees=True)
    psv = np.rad2deg(np.sqrt(-2 * np.log(1 - var)))

    return {
        "psv": psv,  # Paleosecular variation
        "R": R,  # Resultant length
        "mean_declination": np.rad2deg(np.arctan2(y_mean, x_mean)),
        "mean_inclination": np.rad2deg(np.arcsin(z_mean)),
    }


def paleolatitude_from_inclination(inclination: float) -> float:
    """
    Calculate paleolatitude from inclination (assuming dipole field).

    Parameters
    ----------
    inclination : float
        Magnetic inclination in degrees

    Returns
    -------
    float
        Paleolatitude in degrees

    Examples
    --------
    >>> inc = 60.0
    >>> palat = paleolatitude_from_inclination(inc)
    >>> print(f"Paleolatitude: {palat:.2f}°")
    """
    inc_rad = np.deg2rad(inclination)

    # Dipole equation: tan(I) = 2 * tan(λ)
    paleolatitude = np.rad2deg(np.arctan(np.tan(inc_rad) / 2))

    return paleolatitude


def demagnetization_analysis(temperatures: np.ndarray, magnetization: np.ndarray) -> dict:
    """
    Analyze thermal demagnetization data.

    Parameters
    ----------
    temperatures : array_like
        Demagnetization temperatures (°C)
    magnetization : array_like
        Remaining magnetization (normalized or absolute)

    Returns
    -------
    dict
        Blocking temperature spectrum

    Examples
    --------
    >>> temps = np.array([0, 100, 200, 300, 400, 500, 580, 600])
    >>> mag = np.array([1.0, 0.95, 0.85, 0.70, 0.45, 0.20, 0.05, 0.01])
    >>> result = demagnetization_analysis(temps, mag)
    >>> print(f"Curie temperature estimate: {result['curie_temp']:.0f}°C")
    """
    temperatures = np.asarray(temperatures)
    magnetization = np.asarray(magnetization)

    # Sort by temperature
    sort_idx = np.argsort(temperatures)
    temps_sorted = temperatures[sort_idx]
    mag_sorted = magnetization[sort_idx]

    # Normalize if not already
    mag_normalized = mag_sorted / mag_sorted[0]

    # Calculate derivative (blocking temperature spectrum)
    if len(temps_sorted) > 2:
        derivative = -np.gradient(mag_normalized, temps_sorted)
    else:
        derivative = np.zeros_like(mag_normalized)

    # Estimate Curie temperature (where magnetization drops to near zero)
    curie_indices = np.where(mag_normalized < 0.1)[0]
    if len(curie_indices) > 0:
        curie_temp = temps_sorted[curie_indices[0]]
    else:
        curie_temp = temps_sorted[-1]

    return {
        "temperatures": temps_sorted,
        "normalized_magnetization": mag_normalized,
        "blocking_spectrum": derivative,
        "curie_temp": curie_temp,
    }
