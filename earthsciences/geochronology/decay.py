"""
Radioactive decay calculations

Basic decay equations and age calculations.
"""

import numpy as np
from typing import Optional


# Common decay constants (per year)
DECAY_CONSTANTS = {
    'U238': 1.55125e-10,    # 238U → 206Pb, half-life = 4.468 Ga
    'U235': 9.8485e-10,     # 235U → 207Pb, half-life = 0.704 Ga
    'Th232': 4.9475e-11,    # 232Th → 208Pb, half-life = 14.01 Ga
    'Rb87': 1.42e-11,       # 87Rb → 87Sr, half-life = 48.8 Ga
    'Sm147': 6.54e-12,      # 147Sm → 143Nd, half-life = 106 Ga
    'K40': 5.463e-10,       # 40K → 40Ar, half-life = 1.25 Ga
    'C14': 1.21e-4,         # 14C, half-life = 5730 years
}


def radioactive_decay(N0: float, t: float, half_life: float = None,
                     decay_constant: float = None) -> float:
    """
    Calculate amount remaining after radioactive decay.
    
    N(t) = N0 * exp(-λt)
    
    Parameters
    ----------
    N0 : float
        Initial amount
    t : float
        Time elapsed
    half_life : float, optional
        Half-life (provide either half_life or decay_constant)
    decay_constant : float, optional
        Decay constant λ (provide either half_life or decay_constant)
        
    Returns
    -------
    float
        Amount remaining after time t
        
    Examples
    --------
    >>> # Carbon-14 dating
    >>> N0 = 100  # Initial 14C atoms
    >>> t = 5730  # years (one half-life)
    >>> N = radioactive_decay(N0, t, half_life=5730)
    >>> print(f"After 1 half-life: {N:.1f} atoms")
    After 1 half-life: 50.0 atoms
    """
    if decay_constant is None:
        if half_life is None:
            raise ValueError("Must provide either half_life or decay_constant")
        decay_constant = np.log(2) / half_life
    
    return N0 * np.exp(-decay_constant * t)


def calculate_age(N_parent: float, N_daughter: float, decay_constant: float) -> float:
    """
    Calculate age from parent/daughter ratio.
    
    t = (1/λ) * ln(1 + N_daughter/N_parent)
    
    Parameters
    ----------
    N_parent : float
        Number of parent atoms
    N_daughter : float
        Number of daughter atoms
    decay_constant : float
        Decay constant λ
        
    Returns
    -------
    float
        Age in same units as 1/decay_constant
        
    Examples
    --------
    >>> # U-238 dating
    >>> N_U238 = 100
    >>> N_Pb206 = 50
    >>> lambda_U238 = DECAY_CONSTANTS['U238']
    >>> age = calculate_age(N_U238, N_Pb206, lambda_U238)
    >>> print(f"Age: {age / 1e9:.2f} Ga")
    """
    if N_parent <= 0:
        raise ValueError("N_parent must be positive")
    
    age = (1 / decay_constant) * np.log(1 + N_daughter / N_parent)
    return age
