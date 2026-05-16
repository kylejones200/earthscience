"""
earthsciences — geoscience data analysis for Python.

Subpackages cover statistics, time series, spatial geostatistics, multivariate
methods, directional data, geochronology, geophysics, and related domains.
"""

__version__ = "0.3.0"
__author__ = "Kyle Jones"

# Import main modules
from . import (
    data,
    directional,
    geochronology,
    geophysics,
    hydrogeology,
    imaging,
    multivariate,
    petroleum,
    seismology,
    spatial,
    statistics,
    timeseries,
    utils,
)

__all__ = [
    "statistics",
    "timeseries",
    "spatial",
    "multivariate",
    "directional",
    "imaging",
    "data",
    "geochronology",
    "geophysics",
    "petroleum",
    "hydrogeology",
    "seismology",
    "utils",
]
