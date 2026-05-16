"""
Production decline analysis and PVT properties

Decline curve analysis, fluid properties, and reservoir pressure analysis.
"""

import warnings

import numpy as np
from scipy.optimize import curve_fit


def decline_curve_analysis(
    time: np.ndarray,
    production: np.ndarray,
    method: str = "arps",
    forecast_time: np.ndarray | None = None,
) -> dict:
    """
    Production decline curve analysis.

    Fits exponential, harmonic, or hyperbolic decline models to production data.
    """
    time = np.asarray(time)
    production = np.asarray(production)

    if len(time) != len(production):
        raise ValueError(
            f"time and production must have same length. "
            f"Got time={len(time)}, production={len(production)}"
        )

    # Define decline models
    def exponential(t, qi, Di):
        return qi * np.exp(-Di * t)

    def harmonic(t, qi, Di):
        return qi / (1 + Di * t)

    def hyperbolic(t, qi, Di, b):
        return qi / (1 + b * Di * t) ** (1 / b)

    # Fit model
    b: float
    try:
        if method == "exponential":
            popt, _ = curve_fit(
                exponential,
                time,
                production,
                p0=[production[0], 0.01],
                bounds=([0, 0], [np.inf, 1]),
            )
            qi, Di = popt
            b = 0
            model = lambda t: exponential(t, qi, Di)

        elif method == "harmonic":
            popt, _ = curve_fit(
                harmonic, time, production, p0=[production[0], 0.01], bounds=([0, 0], [np.inf, 1])
            )
            qi, Di = popt
            b = 1
            model = lambda t: harmonic(t, qi, Di)

        elif method in ("hyperbolic", "arps"):
            popt, _ = curve_fit(
                hyperbolic,
                time,
                production,
                p0=[production[0], 0.01, 0.5],
                bounds=([0, 0, 0], [np.inf, 1, 1]),
            )
            qi, Di, b = popt
            model = lambda t: hyperbolic(t, qi, Di, b)

        else:
            raise ValueError(f"Unknown method: {method}")

    except RuntimeError:
        warnings.warn(
            "Curve fitting failed to converge. Using initial estimates. "
            "Results may be inaccurate. Check data quality."
        )
        qi = float(production[0])
        Di = 0.01
        b = 0.5
        model = lambda t: hyperbolic(t, qi, Di, b)

    # Forecast
    if forecast_time is not None:
        forecast = model(forecast_time)
    else:
        forecast = None

    # Calculate EUR
    economic_limit = 10
    if b < 1:
        t_limit = ((qi / economic_limit) ** b - 1) / (b * Di)
        EUR = (qi**b) / (Di * (1 - b)) * (qi ** (1 - b) - economic_limit ** (1 - b))
    else:
        t_grid = np.linspace(0, time[-1] * 5, 1000)
        EUR = float(np.trapezoid(model(t_grid), t_grid))

    return {
        "qi": qi,
        "Di": Di,
        "b": b,
        "method": method,
        "forecast": forecast,
        "EUR": EUR,
        "fitted_curve": model(time),
    }


def oil_formation_volume_factor(
    Rs: np.ndarray, gas_gravity: float, oil_gravity: float, temperature: float
) -> np.ndarray:
    """
    Calculate oil formation volume factor (Bo).

    Standing correlation for black oil systems.
    """
    Rs = np.asarray(Rs)

    # Standing correlation
    F = Rs * (gas_gravity / oil_gravity) ** 0.5 + 1.25 * temperature
    Bo = 0.9759 + 0.00012 * F**1.2

    return Bo


def gas_deviation_factor(
    pressure: np.ndarray, temperature: float, gas_gravity: float = 0.65
) -> np.ndarray:
    """
    Calculate gas compressibility factor (Z-factor).

    Uses Standing-Katz correlation.
    """
    pressure = np.asarray(pressure)

    # Pseudo-critical properties (Sutton correlation)
    Ppc = 756.8 - 131.0 * gas_gravity - 3.6 * gas_gravity**2
    Tpc = 169.2 + 349.5 * gas_gravity - 74.0 * gas_gravity**2

    # Pseudo-reduced properties
    Ppr = pressure / Ppc
    Tpr = (temperature + 460) / Tpc  # Convert to Rankine

    # Simplified Standing-Katz (Dranchuk-Abou-Kassem)
    A = 0.06125 * Ppr * np.exp(-1.2 * (1 - 1 / Tpr) ** 2)
    Z = 1 - A

    # Ensure physical bounds
    Z = np.clip(Z, 0.3, 1.2)

    return Z


def reservoir_pressure_from_rft(depth: np.ndarray, pressure: np.ndarray) -> dict:
    """
    Analyze RFT/MDT pressure data for fluid contacts and gradients.
    """
    depth = np.asarray(depth)
    pressure = np.asarray(pressure)

    # Linear regression for gradient
    slope, intercept = np.polyfit(depth, pressure, 1)

    gradient = slope

    # Classify fluid type
    if gradient < 0.15:
        fluid_type = "gas"
    elif gradient < 0.40:
        fluid_type = "oil"
    else:
        fluid_type = "water"

    return {
        "gradient": gradient,
        "intercept": intercept,
        "fluid_type": fluid_type,
        "fitted_pressure": slope * depth + intercept,
    }
