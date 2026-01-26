"""
Radiocarbon dating

14C age calculations and calibration.
"""

import numpy as np
from typing import Tuple, Dict
import warnings


def radiocarbon_age(N_measured: float, N_modern: float = 1.0,
                   half_life: float = 5730.0) -> Tuple[float, float]:
    """
    Calculate radiocarbon age and uncertainty.
    
    Parameters
    ----------
    N_measured : float
        Measured 14C activity
    N_modern : float, optional
        Modern (pre-1950) 14C activity (default=1.0)
    half_life : float, optional
        14C half-life in years (default=5730)
        
    Returns
    -------
    age : float
        Radiocarbon age in years BP (before present = 1950)
    uncertainty : float
        Age uncertainty (simplified Poisson statistics)
        
    Examples
    --------
    >>> # Sample with 50% modern carbon
    >>> age, uncertainty = radiocarbon_age(0.5, N_modern=1.0)
    >>> print(f"Age: {age:.0f} ± {uncertainty:.0f} years BP")
    Age: 5730 ± 114 years BP
    """
    if N_measured <= 0 or N_measured > N_modern:
        raise ValueError(
            f"N_measured must be between 0 and N_modern. "
            f"Got N_measured={N_measured}, N_modern={N_modern}"
        )
    
    # Calculate age
    decay_constant = np.log(2) / half_life
    age = -np.log(N_measured / N_modern) / decay_constant
    
    # Uncertainty (simplified - assumes 1% counting uncertainty)
    # More realistic would require actual count statistics
    counting_uncertainty = 0.01 * N_measured
    uncertainty = (half_life / np.log(2)) * (counting_uncertainty / N_measured)
    
    return age, uncertainty


def radiocarbon_calibration(c14_age: float, uncertainty: float,
                           calibration_curve: str = 'intcal20') -> Dict:
    """
    Calibrate radiocarbon age to calendar years.
    
    Accounts for atmospheric 14C variations over time.
    
    Parameters
    ----------
    c14_age : float
        Radiocarbon age in years BP
    uncertainty : float
        1-sigma uncertainty
    calibration_curve : str, optional
        Calibration curve: 'intcal20', 'marine20' (default='intcal20')
        
    Returns
    -------
    dict
        Calibrated age ranges and probabilities
        
    Notes
    -----
    This is a simplified implementation. Real calibration requires
    the actual calibration curve data (IntCal20, Marine20, etc.)
    and sophisticated probability calculations.
    
    Examples
    --------
    >>> # Radiocarbon date
    >>> age_bp = 3500  # years BP
    >>> uncertainty = 40
    >>> 
    >>> result = radiocarbon_calibration(age_bp, uncertainty)
    >>> print(f"Calibrated age: {result['median_cal_age']:.0f} cal BP")
    >>> print(f"95.4% range: {result['range_2sigma']}")
    """
    # Simplified calibration (would need actual curve data)
    # For demonstration, use approximate relationship
    
    # Rough approximation: cal BP ≈ 14C BP for recent ages
    # Real curves show deviations due to atmospheric variations
    
    warnings.warn(
        "Using simplified radiocarbon calibration. "
        "For accurate dates, use calibration curves (IntCal, Marine, SHCal) "
        "with tools like OxCal or calib."
    )
    
    # Simplified: assume linear relationship with small offset
    # Real calibration is non-linear and requires interpolation from curves
    cal_age_median = c14_age * 1.02  # Approximate 2% correction
    
    # Uncertainty increases slightly during calibration
    cal_uncertainty = uncertainty * 1.1
    
    # 68.2% (1σ) range
    range_1sigma = (cal_age_median - cal_uncertainty, cal_age_median + cal_uncertainty)
    
    # 95.4% (2σ) range
    range_2sigma = (cal_age_median - 2*cal_uncertainty, cal_age_median + 2*cal_uncertainty)
    
    return {
        'median_cal_age': cal_age_median,
        'cal_uncertainty': cal_uncertainty,
        'range_1sigma': range_1sigma,
        'range_2sigma': range_2sigma,
        'calibration_curve': calibration_curve,
        'note': 'Simplified calibration - use OxCal/CALIB for research',
    }
