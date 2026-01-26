"""
Petrophysics and formation evaluation

Archie's equation, permeability, and reservoir property calculations.
"""

import numpy as np
from typing import Dict
import warnings


def formation_factor(porosity: np.ndarray, cementation_exponent: float = 2.0,
                    tortuosity_factor: float = 1.0) -> np.ndarray:
    """
    Calculate formation resistivity factor (F).
    
    F = a / φ^m (Archie's formation factor)
    
    Parameters
    ----------
    porosity : array_like
        Porosity (fraction, 0-1)
    cementation_exponent : float, optional
        Cementation exponent m (default=2.0)
    tortuosity_factor : float, optional
        Tortuosity factor a (default=1.0)
        
    Returns
    -------
    ndarray
        Formation factor (dimensionless)
    """
    porosity = np.asarray(porosity)
    
    if np.any(porosity <= 0) or np.any(porosity > 1):
        raise ValueError("Porosity must be between 0 and 1")
    
    F = tortuosity_factor / (porosity ** cementation_exponent)
    
    return F


def archie_equation(Rt: np.ndarray, Rw: float, porosity: np.ndarray,
                   m: float = 2.0, n: float = 2.0, a: float = 1.0) -> np.ndarray:
    """
    Calculate water saturation using Archie's equation.
    
    Sw = [(a * Rw) / (φ^m * Rt)]^(1/n)
    """
    Rt = np.asarray(Rt)
    porosity = np.asarray(porosity)
    
    if np.any(porosity <= 0) or np.any(porosity > 1):
        raise ValueError("Porosity must be between 0 and 1")
    
    if np.any(Rt <= 0) or Rw <= 0:
        raise ValueError(
            f"Resistivities must be positive. Got Rw={Rw}, Rt range: [{Rt.min()}, {Rt.max()}]"
        )
    
    # Archie equation
    Sw = ((a * Rw) / (porosity**m * Rt)) ** (1/n)
    
    # Clip to valid range
    Sw = np.clip(Sw, 0, 1)
    
    return Sw


def water_saturation(Rt: np.ndarray, Rw: float, porosity: np.ndarray,
                    method: str = 'archie', **kwargs) -> Dict:
    """
    Calculate water saturation using various methods.
    """
    Rt = np.asarray(Rt)
    porosity = np.asarray(porosity)
    
    if method == 'archie':
        m = kwargs.get('m', 2.0)
        n = kwargs.get('n', 2.0)
        a = kwargs.get('a', 1.0)
        Sw = archie_equation(Rt, Rw, porosity, m, n, a)
        
    elif method == 'simandoux':
        # Simandoux equation for shaly sands
        Vsh = kwargs.get('Vsh', 0.0)
        Rsh = kwargs.get('Rsh', 1.0)
        m = kwargs.get('m', 2.0)
        
        F = 1 / (porosity**m)
        Sw = np.sqrt((F * Rw / Rt) * (1 + Rsh * Vsh / Rw)**2)
        Sw = np.clip(Sw, 0, 1)
        
    elif method == 'indonesia':
        # Indonesian equation for shaly sands
        Vsh = kwargs.get('Vsh', 0.0)
        Rsh = kwargs.get('Rsh', 1.0)
        m = kwargs.get('m', 2.0)
        
        Sw = (1 / (porosity**m * Rt))**0.5 * ((Vsh**2 / Rsh**2) + (1/Rw))**0.5
        Sw = np.clip(Sw, 0, 1)
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return {
        'Sw': Sw,
        'Shc': 1 - Sw,
        'method': method,
    }


def permeability_kozeny_carman(porosity: np.ndarray, grain_size: np.ndarray,
                               tortuosity: float = 1.0) -> np.ndarray:
    """
    Estimate permeability using Kozeny-Carman equation.
    
    k = (d² * φ³) / (180 * (1-φ)² * τ)
    """
    porosity = np.asarray(porosity)
    grain_size = np.asarray(grain_size)
    
    # Convert grain size from microns to cm
    d_cm = grain_size * 1e-4
    
    # Kozeny-Carman equation
    k_cm2 = (d_cm**2 * porosity**3) / (180 * (1 - porosity)**2 * tortuosity)
    
    # Convert cm² to millidarcies
    k_mD = k_cm2 * 1.013e9
    
    return k_mD


def permeability_timur(porosity: np.ndarray, Swi: np.ndarray) -> np.ndarray:
    """
    Estimate permeability using Timur equation.
    
    k = 0.136 * φ^4.4 / Swi²
    """
    porosity = np.asarray(porosity)
    Swi = np.asarray(Swi)
    
    k = 0.136 * (porosity**4.4) / (Swi**2)
    
    return k
