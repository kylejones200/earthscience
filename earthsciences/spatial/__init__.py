"""
Spatial statistics and geostatistics

Variograms, kriging, interpolation, and spatial pattern analysis
for geographically referenced data.
"""

from .variogram import *
from .kriging import *
from .interpolation import *
from .interpolation_advanced import *
from .spatial_stats import *
from .point_patterns import *
from .terrain_analysis import *

__all__ = [
    # Variogram
    "compute_variogram",
    "fit_variogram_model",
    "spherical_model",
    "exponential_model",
    "gaussian_model",
    "power_model",
    "nugget_model",
    
    # Kriging
    "ordinary_kriging",
    "simple_kriging",
    "universal_kriging",
    "indicator_kriging",
    "cokriging",
    
    # Interpolation
    "idw_interpolation",
    "nearest_neighbor_interpolation",
    "natural_neighbor_interpolation",
    "rbf_interpolation",
    "spline_interpolation",
    
    # Advanced interpolation
    "radial_basis_function_interpolation",
    "natural_neighbor_interpolation",
    "minimum_curvature_spline",
    "inverse_distance_squared",
    "moving_average_interpolation",
    "shepard_interpolation",
    
    # Spatial statistics
    "morans_i",
    "gearys_c",
    "spatial_autocorrelation",
    "lisa",
    
    # Point patterns
    "nearest_neighbor_distance",
    "ripley_k",
    "pair_correlation",
    
    # Terrain analysis
    "slope_aspect",
    "curvature",
    "hillshade",
]
