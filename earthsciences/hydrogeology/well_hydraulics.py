"""
Slug tests and well hydraulics

Slug test analysis and Darcy velocity calculations.
"""

import numpy as np
from scipy.optimize import curve_fit
from typing import Dict, Tuple, Optional
import warnings


def slug_test_analysis(drawdown: np.ndarray, time: np.ndarray,
                      method: str = 'hvorslev', r_well: float = 0.05,
                      r_casing: float = 0.10, L_screen: float = 1.0) -> Dict:
    """
    Analyze slug test data to estimate hydraulic conductivity.
    """
    drawdown = np.asarray(drawdown)
    time = np.asarray(time)
    
    if method == 'hvorslev':
        # Find time for 37% recovery
        target = 0.37
        idx = np.argmin(np.abs(drawdown - target))
        T37 = time[idx]
        
        if T37 == 0:
            T37 = 1
        
        # Hvorslev shape factor
        R = r_well
        L = L_screen
        shape_factor = np.log(L / R) if L / R > 1 else 1
        
        # Hydraulic conductivity
        K = (r_casing**2 * shape_factor) / (2 * L * T37)
        
    elif method == 'bouwer_rice':
        # Bouwer-Rice method
        log_H = np.log(drawdown[drawdown > 0])
        t_valid = time[drawdown > 0]
        
        if len(log_H) > 2:
            slope, _ = np.polyfit(t_valid, log_H, 1)
            A = np.log(L_screen / r_well)
            K = -(r_casing**2 * slope) / (2 * L_screen * A)
        else:
            K = np.nan
            
    elif method == 'cooper':
        warnings.warn("Cooper method requires type curve matching; using simplified approach")
        
        def exp_model(t, K_eff):
            alpha = (K_eff * L_screen) / (r_casing**2)
            return np.exp(-alpha * t)
        
        try:
            popt, _ = curve_fit(exp_model, time, drawdown, p0=[1e-5])
            K = popt[0]
        except:
            K = np.nan
            
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return {
        'K': K,
        'method': method,
        'r_well': r_well,
        'L_screen': L_screen,
    }


def darcy_velocity(K: float, hydraulic_gradient: float,
                  porosity: Optional[float] = None) -> Tuple[float, Optional[float]]:
    """
    Calculate Darcy velocity and seepage velocity.
    
    Darcy's Law: q = -K * (dh/dl)
    Seepage velocity: v = q / n
    """
    # Darcy velocity (specific discharge)
    q = K * hydraulic_gradient
    
    # Seepage velocity (actual groundwater velocity)
    if porosity is not None:
        v = q / porosity
    else:
        v = None
    
    return q, v


def well_capture_zone(Q: float, K: float, b: float, hydraulic_gradient: float) -> Dict:
    """
    Calculate well capture zone dimensions.
    
    For steady-state uniform flow conditions.
    """
    # Stagnation point (upgradient distance)
    x_s = -Q / (2 * np.pi * K * b * hydraulic_gradient)
    
    # Maximum width of capture zone
    y_max = Q / (2 * K * b * hydraulic_gradient)
    
    # Downgradient extent
    x_downgradient = -10 * x_s
    
    return {
        'x_stagnation': abs(x_s),
        'y_max': y_max,
        'x_downgradient': x_downgradient,
        'Q': Q,
        'K': K,
        'b': b,
    }


def aquifer_safe_yield(recharge: float, area: float, current_pumping: float,
                      allowable_drawdown: float, T: float, S: float) -> Dict:
    """
    Estimate aquifer safe yield.
    
    Balance between recharge and sustainable pumping.
    """
    # Total recharge
    total_recharge = recharge * area
    
    # Sustainable yield
    sustainable_yield = 0.8 * total_recharge
    
    # Current balance
    annual_deficit = current_pumping - total_recharge
    
    # Time to exceed allowable drawdown
    if annual_deficit > 0:
        volume_deficit = annual_deficit
        volume_per_meter_drawdown = S * area
        years_to_limit = (allowable_drawdown * volume_per_meter_drawdown) / volume_deficit
    else:
        years_to_limit = np.inf
    
    return {
        'sustainable_yield': sustainable_yield,
        'total_recharge': total_recharge,
        'current_pumping': current_pumping,
        'annual_deficit': annual_deficit,
        'years_to_limit': years_to_limit,
        'is_sustainable': annual_deficit <= 0,
    }


def groundwater_age_dating(tracer_concentration: float, initial_concentration: float,
                          decay_constant: float) -> float:
    """
    Estimate groundwater age from radioactive tracer decay.
    
    Common tracers: ³H (tritium), ¹⁴C, ³⁶Cl, ⁸⁵Kr
    """
    if tracer_concentration <= 0 or initial_concentration <= 0:
        raise ValueError(
            f"Concentrations must be positive. "
            f"Got tracer={tracer_concentration}, initial={initial_concentration}"
        )
    
    if tracer_concentration > initial_concentration:
        warnings.warn("Measured concentration exceeds initial; may indicate contamination or mixing")
    
    # Age from radioactive decay: C = C0 * exp(-λt)
    age = -np.log(tracer_concentration / initial_concentration) / decay_constant
    
    return age
