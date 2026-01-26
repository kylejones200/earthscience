"""
Gravity data processing and corrections

Bouguer, free-air, and terrain corrections for gravity surveys.
"""

import numpy as np
from typing import Dict
import warnings


# Physical constants
G = 6.67430e-11  # Gravitational constant (m³/kg/s²)
EARTH_RADIUS = 6371000  # meters
GAMMA = 9.780327  # Normal gravity at equator (m/s²)


def bouguer_correction(observed_gravity: np.ndarray, elevation: np.ndarray,
                      latitude: np.ndarray, density: float = 2670) -> Dict:
    """
    Apply Bouguer correction to gravity data.
    
    Removes effects of elevation and topography.
    
    Parameters
    ----------
    observed_gravity : array_like
        Observed gravity values (mGal)
    elevation : array_like
        Station elevations (meters)
    latitude : array_like
        Station latitudes (degrees)
    density : float, optional
        Rock density (kg/m³, default=2670 for crustal rocks)
        
    Returns
    -------
    dict
        Corrected gravity and individual corrections
        
    Examples
    --------
    >>> # Gravity survey
    >>> g_obs = np.array([980000, 980100, 980050])  # mGal
    >>> elev = np.array([100, 200, 150])  # meters
    >>> lat = np.array([45, 45, 45])  # degrees
    >>> 
    >>> result = bouguer_correction(g_obs, elev, lat)
    >>> print(f"Bouguer anomaly: {result['bouguer_anomaly']}")
    """
    observed_gravity = np.asarray(observed_gravity)
    elevation = np.asarray(elevation)
    latitude = np.asarray(latitude)
    
    # Free-air correction: -0.3086 mGal/m
    free_air_corr = -0.3086 * elevation
    
    # Bouguer plate correction: 0.04193 * density * height (mGal)
    bouguer_plate_corr = 0.04193 * (density / 1000) * elevation
    
    # Latitude correction (normal gravity variation)
    lat_rad = np.deg2rad(latitude)
    normal_gravity = 978032.7 * (1 + 0.0053024 * np.sin(lat_rad)**2 
                                  - 0.0000058 * np.sin(2*lat_rad)**2)
    
    # Total Bouguer correction
    corrected_gravity = observed_gravity + free_air_corr - bouguer_plate_corr
    
    # Bouguer anomaly = observed - normal - corrections
    bouguer_anomaly = corrected_gravity - normal_gravity
    
    return {
        'bouguer_anomaly': bouguer_anomaly,
        'corrected_gravity': corrected_gravity,
        'free_air_correction': free_air_corr,
        'bouguer_correction': bouguer_plate_corr,
        'normal_gravity': normal_gravity,
    }


def free_air_correction(observed_gravity: np.ndarray, elevation: np.ndarray) -> np.ndarray:
    """
    Apply free-air correction to gravity data.
    
    Corrects for elevation without considering topographic mass.
    
    Parameters
    ----------
    observed_gravity : array_like
        Observed gravity values (mGal)
    elevation : array_like
        Station elevations (meters)
        
    Returns
    -------
    ndarray
        Free-air corrected gravity (mGal)
        
    Examples
    --------
    >>> g_obs = np.array([980000, 980100])
    >>> elev = np.array([0, 100])
    >>> g_corrected = free_air_correction(g_obs, elev)
    """
    observed_gravity = np.asarray(observed_gravity)
    elevation = np.asarray(elevation)
    
    # Free-air gradient: -0.3086 mGal/m
    correction = -0.3086 * elevation
    
    return observed_gravity + correction


def terrain_correction(gravity: np.ndarray, dem: np.ndarray,
                      station_elevations: np.ndarray, grid_spacing: float,
                      density: float = 2670) -> np.ndarray:
    """
    Apply terrain correction for irregular topography.
    
    Accounts for gravitational effect of nearby topography.
    
    Parameters
    ----------
    gravity : array_like
        Gravity values at stations (mGal)
    dem : array_like
        Digital elevation model (2D array, meters)
    station_elevations : array_like
        Elevations of gravity stations (meters)
    grid_spacing : float
        DEM grid spacing (meters)
    density : float, optional
        Rock density (kg/m³, default=2670)
        
    Returns
    -------
    ndarray
        Terrain-corrected gravity values (mGal)
        
    Notes
    -----
    This is a simplified implementation using cylindrical zones.
    Full terrain correction requires 3D integration of topography.
    
    Examples
    --------
    >>> # Simplified example
    >>> g = np.array([980000, 980050])
    >>> dem = np.random.rand(10, 10) * 100 + 1000  # meters
    >>> station_elev = np.array([1050, 1060])
    >>> 
    >>> g_corrected = terrain_correction(g, dem, station_elev, grid_spacing=30)
    """
    gravity = np.asarray(gravity)
    dem = np.asarray(dem)
    station_elevations = np.asarray(station_elevations)
    
    # Simplified terrain correction using average topography
    # Real implementation would require 3D integration
    
    warnings.warn(
        "Using simplified terrain correction. "
        "Full implementation requires 3D integration with external DEM. "
        "Results may differ significantly from rigorous methods."
    )
    
    # Estimate average elevation difference around stations
    avg_terrain_elevation = np.mean(dem)
    terrain_diff = station_elevations - avg_terrain_elevation
    
    # Approximate terrain effect (mGal)
    # Uses simplified cylinder model
    terrain_effect = 0.04193 * (density / 1000) * np.abs(terrain_diff) * 0.1  # Factor for terrain irregularity
    
    return gravity + terrain_effect
