"""
Earth Sciences Recipes for Python

A modern Python library for earth sciences data analysis

"""

__version__ = "0.3.0"
__author__ = "j"

# Import main modules
from . import statistics
from . import timeseries
from . import spatial
from . import multivariate
from . import directional
from . import imaging
from . import data
from . import geochronology
from . import geophysics
from . import petroleum
from . import hydrogeology
from . import seismology
from . import utils

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
