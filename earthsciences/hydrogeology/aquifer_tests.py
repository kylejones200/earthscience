"""
Aquifer hydraulics

Well hydraulics, aquifer testing, and pumping test analysis.
"""

import numpy as np
from scipy.special import exp1


def theis_solution(r: np.ndarray, t: np.ndarray, T: float, S: float, Q: float) -> np.ndarray:
    """
    Theis solution for transient flow in confined aquifer.

    s = (Q / 4πT) * W(u), where u = r²S / 4Tt
    """
    r = np.asarray(r)
    t = np.asarray(t)

    # Calculate u parameter
    u = (r**2 * S) / (4 * T * t)

    # Well function W(u) = exponential integral
    W_u = exp1(u)

    # Theis solution
    s = (Q / (4 * np.pi * T)) * W_u

    return s


def hantush_jacob(
    r: np.ndarray, t: np.ndarray, T: float, S: float, Q: float, B: float
) -> np.ndarray:
    """
    Hantush-Jacob solution for leaky confined aquifer.

    Accounts for vertical leakage through aquitard.
    """
    r = np.asarray(r)
    t = np.asarray(t)

    # Calculate u
    u = (r**2 * S) / (4 * T * t)

    # Leakage parameter
    beta = r / B

    # Simplified Hantush-Jacob (approximation for small leakage)
    W_u_beta = exp1(u) * np.exp(-beta)

    s = (Q / (4 * np.pi * T)) * W_u_beta

    return s


def neuman_solution(
    r: np.ndarray, t: np.ndarray, T: float, S: float, Sy: float, Q: float, b: float
) -> np.ndarray:
    """
    Neuman solution for unconfined aquifer with delayed yield.

    Accounts for gravity drainage (delayed yield from storage).
    """
    r = np.asarray(r)
    t = np.asarray(t)

    # Early time (elastic response)
    u_early = (r**2 * S) / (4 * T * t)
    s_early = (Q / (4 * np.pi * T)) * exp1(u_early)

    # Late time (gravity drainage)
    u_late = (r**2 * Sy) / (4 * T * t)
    s_late = (Q / (4 * np.pi * T)) * exp1(u_late)

    # Transition (simplified - blend early and late)
    transition_time = (r**2 * S) / (T * 10)
    weight = 1 / (1 + np.exp(-(t - transition_time) / (transition_time * 0.5)))

    s = (1 - weight) * s_early + weight * s_late

    return s


def pumping_test_analysis(
    drawdown: np.ndarray,
    distance: np.ndarray,
    time: float,
    Q: float,
    aquifer_type: str = "confined",
) -> dict:
    """
    Analyze pumping test data using Cooper-Jacob method.

    Straight-line method for distance-drawdown or time-drawdown analysis.
    """
    drawdown = np.asarray(drawdown)
    distance = np.asarray(distance)

    # Cooper-Jacob straight-line method
    # Determine if distance-drawdown or time-drawdown
    if np.max(distance) < 10000:  # Likely spatial distance
        log_var = np.log(distance)
        var_type = "distance"
    else:
        log_var = np.log(distance)  # distance is actually time
        var_type = "time"

    # Linear regression
    slope, intercept = np.polyfit(log_var, drawdown, 1)

    # Calculate aquifer properties
    if var_type == "distance":
        T = -2.303 * Q / (4 * np.pi * slope)
        r0 = np.exp(-intercept / slope)
        S = 2.25 * T * time / r0**2
    else:
        T = 2.303 * Q / (4 * np.pi * slope)
        t0 = np.exp(-intercept / slope)
        r_obs = distance[0]
        S = 2.25 * T * t0 / r_obs**2

    return {
        "T": T,
        "S": S if aquifer_type == "confined" else None,
        "Sy": S if aquifer_type == "unconfined" else None,
        "slope": slope,
        "intercept": intercept,
        "aquifer_type": aquifer_type,
    }
