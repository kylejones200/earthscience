"""
Spatial statistics and geostatistics.

Variograms, kriging, interpolation, and spatial pattern analysis.
"""

import warnings

from .interpolation import (
    bilinear_interpolation,
    griddata_interpolation,
    idw_interpolation,
    natural_neighbor,
    nearest_neighbor,
    nearest_neighbor_interpolation,
    rbf_interpolation,
    spline_interpolation,
)
from .interpolation_advanced import (
    inverse_distance_squared,
    minimum_curvature_spline,
    moving_average_interpolation,
    natural_neighbor_interpolation,
    radial_basis_function_interpolation,
    shepard_interpolation,
)
from .kriging import (
    cross_validate,
    kriging_variance,
    ordinary_kriging,
    simple_kriging,
    universal_kriging,
)
from .point_patterns import (
    clark_evans_statistic,
    f_function,
    g_function,
    nearest_neighbor_distance,
    point_density,
    ripley_k,
)
from .spatial_stats import (
    gearys_c,
    getis_ord_g,
    local_morans_i,
    morans_i,
    semivariogram,
    spatial_correlogram,
    spatial_weights_matrix,
)
from .terrain_analysis import (
    calculate_flow_accumulation,
    curvature,
    hillshade,
    slope_aspect,
    stream_power_index,
    terrain_ruggedness_index,
    topographic_position_index,
    topographic_wetness_index,
)
from .variogram import (
    anisotropic_variogram,
    compute_variogram,
    exponential_model,
    fit_variogram_model,
    gaussian_model,
    spherical_model,
    variogram_cloud,
)

__all__ = [
    # Variogram
    "compute_variogram",
    "fit_variogram_model",
    "variogram_cloud",
    "anisotropic_variogram",
    "spherical_model",
    "exponential_model",
    "gaussian_model",
    # Kriging
    "ordinary_kriging",
    "simple_kriging",
    "universal_kriging",
    "cross_validate",
    "kriging_variance",
    # Interpolation
    "idw_interpolation",
    "nearest_neighbor",
    "nearest_neighbor_interpolation",
    "spline_interpolation",
    "griddata_interpolation",
    "natural_neighbor",
    "natural_neighbor_interpolation",
    "bilinear_interpolation",
    "rbf_interpolation",
    "radial_basis_function_interpolation",
    "minimum_curvature_spline",
    "inverse_distance_squared",
    "moving_average_interpolation",
    "shepard_interpolation",
    # Spatial statistics
    "morans_i",
    "gearys_c",
    "local_morans_i",
    "getis_ord_g",
    "spatial_weights_matrix",
    "spatial_correlogram",
    "semivariogram",
    # Point patterns
    "nearest_neighbor_distance",
    "ripley_k",
    "point_density",
    "clark_evans_statistic",
    "g_function",
    "f_function",
    # Terrain
    "slope_aspect",
    "curvature",
    "hillshade",
    "topographic_wetness_index",
    "calculate_flow_accumulation",
    "stream_power_index",
    "terrain_ruggedness_index",
    "topographic_position_index",
]

# Backward-compatible aliases (deprecated names)
lisa = local_morans_i
spatial_autocorrelation = morans_i


def __getattr__(name: str):
    if name == "pair_correlation":
        warnings.warn(
            "spatial.pair_correlation was removed; use point pattern functions such as "
            "ripley_k or g_function",
            DeprecationWarning,
            stacklevel=2,
        )
        raise AttributeError(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
