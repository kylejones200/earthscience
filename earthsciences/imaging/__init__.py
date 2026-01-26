"""
Image processing for earth sciences

Functions for processing satellite imagery, microscopic images, and
digital elevation models.
"""

from .ndvi import *
from .enhancement import *
from .analysis import *

__all__ = [
    # NDVI and vegetation indices
    "ndvi",
    "evi",
    "savi",
    "ndwi",
    "nbr",
    "gndvi",
    "msavi",
    
    # Image enhancement
    "histogram_equalization",
    "contrast_stretch",
    "adaptive_histogram_equalization",
    "gamma_correction",
    "unsharp_mask",
    "denoise",
    "remove_periodic_noise",
    "atmospheric_correction_dos",
    "calculate_toa_reflectance",
    "pan_sharpen",
    "cloud_mask",
    
    # Advanced analysis
    "grain_size_distribution",
    "shape_analysis",
    "fabric_analysis",
    "watershed_segmentation",
    "quantify_charcoal",
    "detect_circular_objects",
]
