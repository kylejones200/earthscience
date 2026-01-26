"""
Radiometric dating methods

Isochron dating, U-Pb concordia, fission track, and cosmogenic exposure dating.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from .decay import DECAY_CONSTANTS


def isochron_dating(parent_ratios: np.ndarray, daughter_ratios: np.ndarray,
                   reference_ratios: np.ndarray, decay_constant: float) -> Dict:
    """
    Isochron dating method.
    
    More robust than simple parent-daughter dating as it accounts for
    initial daughter isotopes.
    
    Parameters
    ----------
    parent_ratios : array_like
        Parent/reference isotope ratios (e.g., 87Rb/86Sr)
    daughter_ratios : array_like
        Daughter/reference isotope ratios (e.g., 87Sr/86Sr)
    reference_ratios : array_like
        Reference isotope ratios
    decay_constant : float
        Decay constant for parent isotope
        
    Returns
    -------
    dict
        Age, initial ratio, and fit statistics
        
    Notes
    -----
    The isochron equation:
    (Daughter/Ref) = (Daughter/Ref)_initial + (Parent/Ref) * (exp(λt) - 1)
    
    Linear form: D/R = D/R_0 + P/R * (exp(λt) - 1)
    Slope m = exp(λt) - 1, thus t = ln(m + 1) / λ
    
    Examples
    --------
    >>> # Rb-Sr isochron
    >>> parent = np.array([0.5, 1.0, 1.5, 2.0])  # 87Rb/86Sr
    >>> daughter = np.array([0.710, 0.715, 0.720, 0.725])  # 87Sr/86Sr
    >>> reference = np.ones(4)  # Normalized
    >>> lambda_Rb87 = DECAY_CONSTANTS['Rb87']
    >>> 
    >>> result = isochron_dating(parent, daughter, reference, lambda_Rb87)
    >>> print(f"Age: {result['age'] / 1e9:.2f} Ga")
    >>> print(f"Initial ratio: {result['initial_ratio']:.5f}")
    >>> print(f"MSWD: {result['mswd']:.2f}")
    """
    parent_ratios = np.asarray(parent_ratios)
    daughter_ratios = np.asarray(daughter_ratios)
    
    # Linear regression
    slope, intercept = np.polyfit(parent_ratios, daughter_ratios, 1)
    
    # Calculate age from slope
    # slope = exp(λt) - 1
    age = np.log(slope + 1) / decay_constant
    
    # Initial daughter ratio is y-intercept
    initial_ratio = intercept
    
    # Calculate fit quality (MSWD - Mean Square of Weighted Deviates)
    y_pred = slope * parent_ratios + intercept
    residuals = daughter_ratios - y_pred
    n = len(parent_ratios)
    
    # Assuming equal uncertainties (simplified)
    chi_squared = np.sum(residuals**2)
    mswd = chi_squared / (n - 2) if n > 2 else np.inf
    
    return {
        'age': age,
        'initial_ratio': initial_ratio,
        'slope': slope,
        'intercept': intercept,
        'mswd': mswd,
        'n_samples': n,
    }


def concordia_diagram_age(ratio_206_238: np.ndarray, ratio_207_235: np.ndarray,
                         uncertainty_206_238: Optional[np.ndarray] = None,
                         uncertainty_207_235: Optional[np.ndarray] = None) -> Dict:
    """
    U-Pb concordia diagram analysis.
    
    Uses two independent decay systems to check concordance.
    
    Parameters
    ----------
    ratio_206_238 : array_like
        206Pb/238U ratios
    ratio_207_235 : array_like
        207Pb/235U ratios
    uncertainty_206_238 : array_like, optional
        Uncertainties in 206Pb/238U
    uncertainty_207_235 : array_like, optional
        Uncertainties in 207Pb/235U
        
    Returns
    -------
    dict
        Concordia ages and concordance statistics
        
    Examples
    --------
    >>> # Zircon U-Pb data
    >>> ratio_206_238 = np.array([0.4, 0.41, 0.39])
    >>> ratio_207_235 = np.array([5.2, 5.3, 5.1])
    >>> 
    >>> result = concordia_diagram_age(ratio_206_238, ratio_207_235)
    >>> print(f"206Pb/238U age: {result['age_206_238'] / 1e9:.2f} Ga")
    >>> print(f"207Pb/235U age: {result['age_207_235'] / 1e9:.2f} Ga")
    >>> print(f"Concordance: {result['concordance']:.1%}")
    """
    ratio_206_238 = np.asarray(ratio_206_238)
    ratio_207_235 = np.asarray(ratio_207_235)
    
    lambda_U238 = DECAY_CONSTANTS['U238']
    lambda_U235 = DECAY_CONSTANTS['U235']
    
    # Calculate ages from each system
    age_206_238 = np.log(1 + ratio_206_238) / lambda_U238
    age_207_235 = np.log(1 + ratio_207_235) / lambda_U235
    
    # Concordance metric (percent difference)
    concordance = 1 - np.abs(age_206_238 - age_207_235) / np.maximum(age_206_238, age_207_235)
    
    return {
        'age_206_238': np.mean(age_206_238),
        'age_207_235': np.mean(age_207_235),
        'ages_206_238': age_206_238,
        'ages_207_235': age_207_235,
        'concordance': np.mean(concordance),
        'concordance_all': concordance,
        'discordant': np.any(concordance < 0.9),  # >10% discordance
    }


def fission_track_age(track_density_sample: float, track_density_standard: float,
                     age_standard: float, zeta: float) -> Tuple[float, float]:
    """
    Calculate fission track age.
    
    Used for dating minerals based on uranium fission tracks.
    
    Parameters
    ----------
    track_density_sample : float
        Spontaneous track density (tracks/cm²)
    track_density_standard : float
        Induced track density in standard (tracks/cm²)
    age_standard : float
        Known age of age standard (years)
    zeta : float
        Zeta calibration factor (personal to analyst/laboratory)
        
    Returns
    -------
    age : float
        Fission track age (years)
    uncertainty : float
        Age uncertainty
        
    Examples
    --------
    >>> # Apatite fission track dating
    >>> rho_s = 1.5e6  # spontaneous tracks/cm²
    >>> rho_i = 3.0e6  # induced tracks/cm²
    >>> age_std = 28.0e6  # 28 Ma standard
    >>> zeta = 350  # typical zeta value
    >>> 
    >>> age, unc = fission_track_age(rho_s, rho_i, age_std, zeta)
    >>> print(f"FT age: {age / 1e6:.1f} ± {unc / 1e6:.1f} Ma")
    """
    # Simplified fission track age equation
    # age = (1/λ_d) * ln(1 + λ_d * ζ * ρ_s / ρ_i)
    # where λ_d is 238U decay constant
    
    lambda_d = DECAY_CONSTANTS['U238']
    
    # Age calculation
    age = (1 / lambda_d) * np.log(1 + lambda_d * zeta * track_density_sample / track_density_standard)
    
    # Uncertainty (simplified Poisson counting statistics)
    # Assumes sqrt(N) uncertainty in track counts
    # Would need actual track counts for precise calculation
    rel_uncertainty = np.sqrt(1/track_density_sample + 1/track_density_standard)
    uncertainty = age * rel_uncertainty * 0.1  # Simplified 10% typical uncertainty
    
    return age, uncertainty


def cosmogenic_exposure_age(nuclide_concentration: float, production_rate: float,
                           decay_constant: float = 0.0) -> float:
    """
    Calculate cosmogenic exposure age.
    
    Used for dating surfaces exposed to cosmic rays (e.g., glacial boulders).
    
    Parameters
    ----------
    nuclide_concentration : float
        Measured nuclide concentration (atoms/g)
    production_rate : float
        Production rate (atoms/g/year)
    decay_constant : float, optional
        Decay constant if unstable isotope (default=0 for stable)
        
    Returns
    -------
    float
        Exposure age (years)
        
    Examples
    --------
    >>> # 10Be dating of glacial boulder
    >>> concentration = 5e5  # atoms/g
    >>> production = 5.0  # atoms/g/year at sea level
    >>> lambda_Be10 = 4.6e-7  # per year
    >>> 
    >>> age = cosmogenic_exposure_age(concentration, production, lambda_Be10)
    >>> print(f"Exposure age: {age / 1000:.1f} ka")
    """
    if decay_constant == 0:
        # Stable nuclide (simple accumulation)
        age = nuclide_concentration / production_rate
    else:
        # Unstable nuclide (production and decay)
        # N = (P/λ) * (1 - exp(-λt))
        # Solve for t: t = -(1/λ) * ln(1 - λN/P)
        age = -(1 / decay_constant) * np.log(1 - decay_constant * nuclide_concentration / production_rate)
    
    return age


def th_u_dating(th230_u234: float, u234_u238: float = 1.0) -> float:
    """
    230Th/234U dating for corals and speleothems.
    
    Parameters
    ----------
    th230_u234 : float
        230Th/234U activity ratio
    u234_u238 : float, optional
        234U/238U activity ratio (default=1.0 for secular equilibrium)
        
    Returns
    -------
    float
        Age in years
        
    Examples
    --------
    >>> # Coral sample
    >>> ratio_Th_U = 0.5
    >>> age = th_u_dating(ratio_Th_U)
    >>> print(f"Age: {age / 1000:.1f} ka")
    """
    import warnings
    
    # 230Th decay constant
    lambda_230 = np.log(2) / 75690  # half-life = 75,690 years
    
    # 234U decay constant  
    lambda_234 = np.log(2) / 245500  # half-life = 245,500 years
    
    # Closed-system Th-U age equation (simplified)
    # This assumes no initial 230Th and secular equilibrium
    if th230_u234 >= 1:
        warnings.warn("230Th/234U ratio >= 1 suggests secular equilibrium or contamination")
    
    age = -(1 / lambda_230) * np.log(1 - th230_u234)
    
    return age
