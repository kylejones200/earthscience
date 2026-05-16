"""
Geospatial data loaders

Functions for loading standard earth science datasets (ETOPO, SRTM, etc.)
and geochemical data (AGDB4).
"""

from .geochem_loaders import (
    get_element_stats,
    load_agdb_chemistry,
    load_agdb_geology,
    load_rock_samples,
    load_stream_sediments,
    prepare_spatial_data,
)
from .loaders import *

__all__ = [
    "load_etopo2022",
    "load_srtm",
    "load_gtopo30",
    "download_tile",
    "load_agdb_geology",
    "load_agdb_chemistry",
    "prepare_spatial_data",
    "load_stream_sediments",
    "load_rock_samples",
    "get_element_stats",
]
