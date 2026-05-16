"""
Vegetation indices and spectral indices for remote sensing

Functions for calculating NDVI, EVI, SAVI, and other indices from
multispectral satellite imagery.
"""

import numpy as np


def ndvi(nir: np.ndarray, red: np.ndarray, mask_invalid: bool = True) -> np.ndarray:
    """
    Normalized Difference Vegetation Index (NDVI).

    NDVI = (NIR - Red) / (NIR + Red)

    Range: -1 to +1
    - Negative values: water
    - 0 to 0.2: bare soil, rock
    - 0.2 to 0.5: sparse vegetation
    - 0.5 to 1.0: dense vegetation

    Parameters
    ----------
    nir : array_like
        Near-infrared band reflectance
    red : array_like
        Red band reflectance
    mask_invalid : bool, optional
        Mask invalid values (default=True)

    Returns
    -------
    ndarray
        NDVI values

    Examples
    --------
    >>> # Landsat 8 bands: Band 4 (red), Band 5 (NIR)
    >>> red = np.random.rand(100, 100) * 0.3  # Typical red reflectance
    >>> nir = np.random.rand(100, 100) * 0.5  # Typical NIR reflectance
    >>> ndvi_img = ndvi(nir, red)
    >>> print(f"NDVI range: {ndvi_img.min():.2f} to {ndvi_img.max():.2f}")
    """
    nir = np.asarray(nir, dtype=float)
    red = np.asarray(red, dtype=float)

    # Calculate NDVI
    numerator = nir - red
    denominator = nir + red

    # Avoid division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        ndvi_values = numerator / denominator

    if mask_invalid:
        # Mask invalid values
        ndvi_values = np.where(np.isfinite(ndvi_values), ndvi_values, np.nan)
        # Mask values outside valid range
        ndvi_values = np.where((ndvi_values >= -1) & (ndvi_values <= 1), ndvi_values, np.nan)

    return ndvi_values


def evi(
    nir: np.ndarray,
    red: np.ndarray,
    blue: np.ndarray,
    G: float = 2.5,
    C1: float = 6.0,
    C2: float = 7.5,
    L: float = 1.0,
    mask_invalid: bool = True,
) -> np.ndarray:
    """
    Enhanced Vegetation Index (EVI).

    EVI = G * (NIR - Red) / (NIR + C1*Red - C2*Blue + L)

    More sensitive to high biomass and less affected by atmospheric conditions.

    Parameters
    ----------
    nir : array_like
        Near-infrared band reflectance
    red : array_like
        Red band reflectance
    blue : array_like
        Blue band reflectance
    G : float, optional
        Gain factor (default=2.5)
    C1, C2 : float, optional
        Aerosol resistance coefficients (default=6.0, 7.5)
    L : float, optional
        Canopy background adjustment (default=1.0)
    mask_invalid : bool, optional
        Mask invalid values (default=True)

    Returns
    -------
    ndarray
        EVI values

    Examples
    --------
    >>> # Landsat 8: Band 2 (blue), Band 4 (red), Band 5 (NIR)
    >>> blue = np.random.rand(100, 100) * 0.2
    >>> red = np.random.rand(100, 100) * 0.3
    >>> nir = np.random.rand(100, 100) * 0.5
    >>> evi_img = evi(nir, red, blue)
    """
    nir = np.asarray(nir, dtype=float)
    red = np.asarray(red, dtype=float)
    blue = np.asarray(blue, dtype=float)

    # Calculate EVI
    numerator = nir - red
    denominator = nir + C1 * red - C2 * blue + L

    with np.errstate(divide="ignore", invalid="ignore"):
        evi_values = G * (numerator / denominator)

    if mask_invalid:
        evi_values = np.where(np.isfinite(evi_values), evi_values, np.nan)

    return evi_values


def savi(nir: np.ndarray, red: np.ndarray, L: float = 0.5, mask_invalid: bool = True) -> np.ndarray:
    """
    Soil Adjusted Vegetation Index (SAVI).

    SAVI = ((NIR - Red) / (NIR + Red + L)) * (1 + L)

    Minimizes soil brightness influences using a soil-brightness correction factor.

    Parameters
    ----------
    nir : array_like
        Near-infrared band reflectance
    red : array_like
        Red band reflectance
    L : float, optional
        Soil brightness correction factor (default=0.5)
        - L=0: very high vegetation cover
        - L=0.5: intermediate vegetation cover
        - L=1: low vegetation cover
    mask_invalid : bool, optional
        Mask invalid values (default=True)

    Returns
    -------
    ndarray
        SAVI values

    Examples
    --------
    >>> red = np.random.rand(100, 100) * 0.3
    >>> nir = np.random.rand(100, 100) * 0.5
    >>> savi_img = savi(nir, red, L=0.5)
    """
    nir = np.asarray(nir, dtype=float)
    red = np.asarray(red, dtype=float)

    # Calculate SAVI
    numerator = nir - red
    denominator = nir + red + L

    with np.errstate(divide="ignore", invalid="ignore"):
        savi_values = (numerator / denominator) * (1 + L)

    if mask_invalid:
        savi_values = np.where(np.isfinite(savi_values), savi_values, np.nan)

    return savi_values


def ndwi(nir: np.ndarray, swir: np.ndarray, mask_invalid: bool = True) -> np.ndarray:
    """
    Normalized Difference Water Index (NDWI).

    NDWI = (NIR - SWIR) / (NIR + SWIR)

    Used for water body detection and vegetation water content.

    Parameters
    ----------
    nir : array_like
        Near-infrared band reflectance
    swir : array_like
        Short-wave infrared band reflectance
    mask_invalid : bool, optional
        Mask invalid values (default=True)

    Returns
    -------
    ndarray
        NDWI values

    Notes
    -----
    Positive values indicate water bodies.
    Negative values indicate vegetation and soil.

    Examples
    --------
    >>> # Landsat 8: Band 5 (NIR), Band 6 (SWIR1)
    >>> nir = np.random.rand(100, 100) * 0.5
    >>> swir = np.random.rand(100, 100) * 0.3
    >>> ndwi_img = ndwi(nir, swir)
    """
    nir = np.asarray(nir, dtype=float)
    swir = np.asarray(swir, dtype=float)

    # Calculate NDWI
    numerator = nir - swir
    denominator = nir + swir

    with np.errstate(divide="ignore", invalid="ignore"):
        ndwi_values = numerator / denominator

    if mask_invalid:
        ndwi_values = np.where(np.isfinite(ndwi_values), ndwi_values, np.nan)
        ndwi_values = np.where((ndwi_values >= -1) & (ndwi_values <= 1), ndwi_values, np.nan)

    return ndwi_values


def ndbi(swir: np.ndarray, nir: np.ndarray, mask_invalid: bool = True) -> np.ndarray:
    """
    Normalized Difference Built-up Index (NDBI).

    NDBI = (SWIR - NIR) / (SWIR + NIR)

    Used for urban area detection.

    Parameters
    ----------
    swir : array_like
        Short-wave infrared band reflectance
    nir : array_like
        Near-infrared band reflectance
    mask_invalid : bool, optional
        Mask invalid values (default=True)

    Returns
    -------
    ndarray
        NDBI values

    Notes
    -----
    Positive values indicate built-up areas.

    Examples
    --------
    >>> swir = np.random.rand(100, 100) * 0.4
    >>> nir = np.random.rand(100, 100) * 0.3
    >>> ndbi_img = ndbi(swir, nir)
    """
    swir = np.asarray(swir, dtype=float)
    nir = np.asarray(nir, dtype=float)

    numerator = swir - nir
    denominator = swir + nir

    with np.errstate(divide="ignore", invalid="ignore"):
        ndbi_values = numerator / denominator

    if mask_invalid:
        ndbi_values = np.where(np.isfinite(ndbi_values), ndbi_values, np.nan)

    return ndbi_values


def nbr(nir: np.ndarray, swir: np.ndarray, mask_invalid: bool = True) -> np.ndarray:
    """
    Normalized Burn Ratio (NBR).

    NBR = (NIR - SWIR) / (NIR + SWIR)

    Used for burn severity assessment.

    Parameters
    ----------
    nir : array_like
        Near-infrared band reflectance
    swir : array_like
        Short-wave infrared band reflectance (typically SWIR2)
    mask_invalid : bool, optional
        Mask invalid values (default=True)

    Returns
    -------
    ndarray
        NBR values

    Notes
    -----
    High values indicate healthy vegetation.
    Low values indicate burned areas.

    Examples
    --------
    >>> # Landsat 8: Band 5 (NIR), Band 7 (SWIR2)
    >>> nir = np.random.rand(100, 100) * 0.5
    >>> swir = np.random.rand(100, 100) * 0.2
    >>> nbr_img = nbr(nir, swir)
    """
    return ndwi(nir, swir, mask_invalid=mask_invalid)


def gndvi(nir: np.ndarray, green: np.ndarray, mask_invalid: bool = True) -> np.ndarray:
    """
    Green Normalized Difference Vegetation Index (GNDVI).

    GNDVI = (NIR - Green) / (NIR + Green)

    More sensitive to chlorophyll concentration than NDVI.

    Parameters
    ----------
    nir : array_like
        Near-infrared band reflectance
    green : array_like
        Green band reflectance
    mask_invalid : bool, optional
        Mask invalid values (default=True)

    Returns
    -------
    ndarray
        GNDVI values

    Examples
    --------
    >>> green = np.random.rand(100, 100) * 0.2
    >>> nir = np.random.rand(100, 100) * 0.5
    >>> gndvi_img = gndvi(nir, green)
    """
    nir = np.asarray(nir, dtype=float)
    green = np.asarray(green, dtype=float)

    numerator = nir - green
    denominator = nir + green

    with np.errstate(divide="ignore", invalid="ignore"):
        gndvi_values = numerator / denominator

    if mask_invalid:
        gndvi_values = np.where(np.isfinite(gndvi_values), gndvi_values, np.nan)

    return gndvi_values


def msavi(nir: np.ndarray, red: np.ndarray, mask_invalid: bool = True) -> np.ndarray:
    """
    Modified Soil Adjusted Vegetation Index (MSAVI).

    MSAVI = (2*NIR + 1 - sqrt((2*NIR + 1)^2 - 8*(NIR - Red))) / 2

    Self-adjusting soil brightness correction.

    Parameters
    ----------
    nir : array_like
        Near-infrared band reflectance
    red : array_like
        Red band reflectance
    mask_invalid : bool, optional
        Mask invalid values (default=True)

    Returns
    -------
    ndarray
        MSAVI values

    Examples
    --------
    >>> red = np.random.rand(100, 100) * 0.3
    >>> nir = np.random.rand(100, 100) * 0.5
    >>> msavi_img = msavi(nir, red)
    """
    nir = np.asarray(nir, dtype=float)
    red = np.asarray(red, dtype=float)

    # MSAVI formula
    term1 = 2 * nir + 1
    term2 = np.sqrt(term1**2 - 8 * (nir - red))

    with np.errstate(divide="ignore", invalid="ignore"):
        msavi_values = (term1 - term2) / 2

    if mask_invalid:
        msavi_values = np.where(np.isfinite(msavi_values), msavi_values, np.nan)

    return msavi_values


def calculate_toa_reflectance(
    dn: np.ndarray,
    radiance_mult: float,
    radiance_add: float,
    sun_elevation: float,
    esun: float,
    d: float = 1.0,
) -> np.ndarray:
    """
    Convert digital numbers (DN) to Top-of-Atmosphere (TOA) reflectance.

    Used for Landsat and similar sensors.

    Parameters
    ----------
    dn : array_like
        Digital numbers from satellite image
    radiance_mult : float
        Radiance multiplicative scaling factor
    radiance_add : float
        Radiance additive scaling factor
    sun_elevation : float
        Sun elevation angle (degrees)
    esun : float
        Mean solar exoatmospheric irradiance
    d : float, optional
        Earth-Sun distance in astronomical units (default=1.0)

    Returns
    -------
    ndarray
        TOA reflectance values

    Examples
    --------
    >>> # Landsat 8 Band 4 (Red)
    >>> dn = np.random.randint(10000, 20000, size=(100, 100))
    >>> toa = calculate_toa_reflectance(
    ...     dn, radiance_mult=0.00002, radiance_add=-0.1,
    ...     sun_elevation=45.0, esun=1580.0, d=1.0
    ... )
    """
    dn = np.asarray(dn, dtype=float)

    # Convert DN to radiance
    radiance = radiance_mult * dn + radiance_add

    # Convert sun elevation to radians
    sun_elevation_rad = np.deg2rad(sun_elevation)

    # Calculate TOA reflectance
    toa_reflectance = (np.pi * radiance * d**2) / (esun * np.sin(sun_elevation_rad))

    return toa_reflectance
