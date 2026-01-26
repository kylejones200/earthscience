"""
Earthquake analysis

Magnitude calculation, epicenter location, and focal mechanisms.
"""

import numpy as np
from scipy.optimize import minimize
from typing import Dict, List, Optional
import warnings


def magnitude_calculation(amplitude: np.ndarray, distance: np.ndarray,
                         magnitude_type: str = 'local',
                         station_correction: Optional[np.ndarray] = None) -> Dict:
    """
    Calculate earthquake magnitude from amplitude measurements.
    """
    amplitude = np.asarray(amplitude)
    distance = np.asarray(distance)
    
    if station_correction is None:
        station_correction = np.zeros_like(amplitude)
    else:
        station_correction = np.asarray(station_correction)
    
    if magnitude_type in ('local', 'ML'):
        # Richter local magnitude
        M = (np.log10(amplitude) + 1.11 * np.log10(distance) 
             + 0.00189 * distance - 2.09 + station_correction)
        
    elif magnitude_type in ('body', 'mb'):
        # Body wave magnitude
        Q = 0.01 * distance
        M = np.log10(amplitude) + Q + station_correction
        
    elif magnitude_type in ('surface', 'Ms'):
        # Surface wave magnitude
        M = np.log10(amplitude) + 1.66 * np.log10(distance) + 2.0 + station_correction
        
    elif magnitude_type in ('moment', 'Mw'):
        warnings.warn(
            "Moment magnitude calculation requires seismic moment from source parameters. "
            "Using simplified formula which may be inaccurate."
        )
        M = np.log10(amplitude) + 1.5 * np.log10(distance) + station_correction
        
    else:
        raise ValueError(f"Unknown magnitude type: {magnitude_type}")
    
    # Calculate mean and uncertainty
    magnitude = np.mean(M)
    uncertainty = np.std(M) / np.sqrt(len(M))
    
    return {
        'magnitude': magnitude,
        'uncertainty': uncertainty,
        'individual_magnitudes': M,
        'magnitude_type': magnitude_type,
        'n_stations': len(M),
    }


def epicenter_location(arrival_times: np.ndarray, station_coords: np.ndarray,
                      velocity_model: Dict[str, float],
                      phase_types: Optional[List[str]] = None) -> Dict:
    """
    Locate earthquake epicenter from arrival times.
    
    Uses least-squares inversion of P and S wave arrivals.
    """
    arrival_times = np.asarray(arrival_times)
    station_coords = np.asarray(station_coords)
    
    if phase_types is None:
        phase_types = ['P'] * len(arrival_times)
    
    Vp = velocity_model.get('Vp', 6.0)
    Vs = velocity_model.get('Vs', Vp / 1.73)
    
    # Define residual function
    def residual(params):
        lat, lon, depth, origin_time = params
        
        residuals = []
        for i, (t_obs, station, phase) in enumerate(zip(arrival_times, station_coords, phase_types)):
            # Calculate distance (simplified flat Earth)
            dlat = (lat - station[0]) * 111.0  # km
            dlon = (lon - station[1]) * 111.0 * np.cos(np.deg2rad(lat))
            dz = depth - station[2]
            
            distance = np.sqrt(dlat**2 + dlon**2 + dz**2)
            
            # Travel time
            if phase == 'P':
                t_pred = origin_time + distance / Vp
            else:
                t_pred = origin_time + distance / Vs
            
            residuals.append(t_obs - t_pred)
        
        return np.sum(np.array(residuals)**2)
    
    # Initial guess
    initial = [
        np.mean(station_coords[:, 0]),
        np.mean(station_coords[:, 1]),
        10.0,
        np.min(arrival_times) - 5.0
    ]
    
    # Optimize
    try:
        result = minimize(residual, initial, method='Nelder-Mead')
        lat, lon, depth, origin_time = result.x
        rms_residual = np.sqrt(result.fun / len(arrival_times))
    except:
        warnings.warn("Optimization failed; using initial guess")
        lat, lon, depth, origin_time = initial
        rms_residual = np.nan
    
    return {
        'latitude': lat,
        'longitude': lon,
        'depth': depth,
        'origin_time': origin_time,
        'rms_residual': rms_residual,
        'n_stations': len(arrival_times),
    }


def focal_mechanism(first_motions: np.ndarray, azimuths: np.ndarray,
                   takeoff_angles: np.ndarray) -> Dict:
    """
    Determine focal mechanism from first motion polarities.
    
    Simplified grid search for fault plane orientation.
    """
    first_motions = np.asarray(first_motions)
    azimuths = np.asarray(azimuths)
    takeoff_angles = np.asarray(takeoff_angles)
    
    # Grid search over fault orientations
    strikes = np.arange(0, 360, 10)
    dips = np.arange(0, 91, 10)
    rakes = np.arange(-180, 181, 10)
    
    best_misfit = np.inf
    best_params = (0, 0, 0)
    
    for strike in strikes:
        for dip in dips:
            if dip == 0:
                continue
            for rake in rakes:
                # Simplified radiation pattern
                azimuth_diff = np.abs(azimuths - strike)
                predicted = np.where(azimuth_diff < 90, 1, -1)
                
                # Count misfits
                misfit = np.sum(predicted != first_motions)
                
                if misfit < best_misfit:
                    best_misfit = misfit
                    best_params = (strike, dip, rake)
    
    strike, dip, rake = best_params
    
    # Calculate auxiliary plane
    strike_aux = (strike + 180) % 360
    dip_aux = 90 - dip
    rake_aux = -rake
    
    return {
        'strike': strike,
        'dip': dip,
        'rake': rake,
        'strike_aux': strike_aux,
        'dip_aux': dip_aux,
        'rake_aux': rake_aux,
        'misfit': best_misfit,
        'quality': (len(first_motions) - best_misfit) / len(first_motions),
    }


def seismic_moment(magnitude: float) -> float:
    """
    Calculate seismic moment from moment magnitude.
    
    M0 = 10^(1.5 * (Mw + 10.7)) dyne-cm
    """
    # Hanks & Kanamori (1979)
    M0_dyne_cm = 10 ** (1.5 * (magnitude + 10.7))
    
    # Convert to N·m
    M0 = M0_dyne_cm * 1e-7
    
    return M0


def rupture_dimensions(magnitude: float) -> Dict:
    """
    Estimate rupture dimensions from magnitude.
    
    Uses empirical scaling relationships.
    """
    # Wells & Coppersmith (1994) scaling relations
    
    # Rupture length (km)
    log_L = -2.57 + 0.62 * magnitude
    length = 10 ** log_L
    
    # Rupture width (km)
    log_W = -0.76 + 0.27 * magnitude
    width = 10 ** log_W
    
    # Rupture area (km²)
    log_A = -3.49 + 0.91 * magnitude
    area = 10 ** log_A
    
    # Average slip (m)
    log_D = -4.80 + 0.69 * magnitude
    slip = 10 ** log_D
    
    return {
        'length': length,
        'width': width,
        'area': area,
        'slip': slip,
        'magnitude': magnitude,
    }
