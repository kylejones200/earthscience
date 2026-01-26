"""
Image enhancement and correction

Functions for improving image quality, contrast, and removing noise.
"""

import numpy as np
from scipy import ndimage
from typing import Optional, Tuple
import warnings


def histogram_equalization(image: np.ndarray,
                           n_bins: int = 256) -> np.ndarray:
    """
    Histogram equalization for contrast enhancement.
    
    Redistributes intensity values to use the full dynamic range.
    
    Parameters
    ----------
    image : array_like
        Input image (2D or 3D)
    n_bins : int, optional
        Number of histogram bins (default=256)
        
    Returns
    -------
    ndarray
        Equalized image
        
    Examples
    --------
    >>> # Low contrast image
    >>> img = np.random.rand(100, 100) * 0.3 + 0.3  # Range [0.3, 0.6]
    >>> eq_img = histogram_equalization(img)
    >>> print(f"Original range: [{img.min():.2f}, {img.max():.2f}]")
    >>> print(f"Equalized range: [{eq_img.min():.2f}, {eq_img.max():.2f}]")
    """
    image = np.asarray(image, dtype=float)
    
    # Flatten image
    flat = image.flatten()
    
    # Remove NaN values
    valid = flat[~np.isnan(flat)]
    
    if len(valid) == 0:
        return image
    
    # Compute histogram
    hist, bins = np.histogram(valid, bins=n_bins)
    
    # Cumulative distribution function
    cdf = hist.cumsum()
    cdf = cdf / cdf[-1]  # Normalize
    
    # Interpolate to get equalized values
    equalized = np.interp(flat, bins[:-1], cdf)
    
    # Reshape back
    equalized = equalized.reshape(image.shape)
    
    return equalized


def contrast_stretch(image: np.ndarray,
                     lower_percentile: float = 2.0,
                     upper_percentile: float = 98.0) -> np.ndarray:
    """
    Linear contrast stretch.
    
    Stretches intensity values between specified percentiles to full range [0, 1].
    
    Parameters
    ----------
    image : array_like
        Input image
    lower_percentile : float, optional
        Lower percentile for stretch (default=2.0)
    upper_percentile : float, optional
        Upper percentile for stretch (default=98.0)
        
    Returns
    -------
    ndarray
        Contrast-stretched image
        
    Examples
    --------
    >>> img = np.random.rand(100, 100) * 0.5 + 0.25
    >>> stretched = contrast_stretch(img, lower_percentile=5, upper_percentile=95)
    """
    image = np.asarray(image, dtype=float)
    
    # Remove NaN values for percentile calculation
    valid = image[~np.isnan(image)]
    
    if len(valid) == 0:
        return image
    
    # Calculate percentiles
    p_low = np.percentile(valid, lower_percentile)
    p_high = np.percentile(valid, upper_percentile)
    
    # Stretch
    stretched = (image - p_low) / (p_high - p_low)
    
    # Clip to [0, 1]
    stretched = np.clip(stretched, 0, 1)
    
    return stretched


def adaptive_histogram_equalization(image: np.ndarray,
                                    clip_limit: float = 0.03,
                                    tile_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Contrast Limited Adaptive Histogram Equalization (CLAHE).
    
    Applies histogram equalization to small regions (tiles) to enhance
    local contrast while limiting noise amplification.
    
    Parameters
    ----------
    image : array_like
        Input image (2D)
    clip_limit : float, optional
        Contrast limiting threshold (default=0.03)
    tile_size : tuple, optional
        Size of tiles for local equalization (default=(8, 8))
        
    Returns
    -------
    ndarray
        Enhanced image
        
    Examples
    --------
    >>> img = np.random.rand(100, 100)
    >>> enhanced = adaptive_histogram_equalization(img, clip_limit=0.03)
    """
    try:
        from skimage import exposure
        
        image = np.asarray(image, dtype=float)
        
        # Normalize to [0, 1]
        img_min = np.nanmin(image)
        img_max = np.nanmax(image)
        
        if img_max > img_min:
            normalized = (image - img_min) / (img_max - img_min)
        else:
            return image
        
        # Apply CLAHE
        enhanced = exposure.equalize_adapthist(
            normalized,
            clip_limit=clip_limit,
            kernel_size=tile_size
        )
        
        return enhanced
        
    except ImportError:
        warnings.warn(
            "scikit-image not available. Install with: pip install scikit-image. "
            "Using global histogram equalization as fallback."
        )
        return histogram_equalization(image)


def gamma_correction(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """
    Gamma correction for brightness adjustment.
    
    Output = Input^gamma
    
    Parameters
    ----------
    image : array_like
        Input image (values should be in [0, 1])
    gamma : float, optional
        Gamma value (default=1.0)
        - gamma < 1: brighten image
        - gamma = 1: no change
        - gamma > 1: darken image
        
    Returns
    -------
    ndarray
        Gamma-corrected image
        
    Examples
    --------
    >>> img = np.random.rand(100, 100)
    >>> brightened = gamma_correction(img, gamma=0.5)  # Brighten
    >>> darkened = gamma_correction(img, gamma=2.0)    # Darken
    """
    image = np.asarray(image, dtype=float)
    
    # Ensure values are in [0, 1]
    img_min = np.nanmin(image)
    img_max = np.nanmax(image)
    
    if img_max > img_min:
        normalized = (image - img_min) / (img_max - img_min)
    else:
        return image
    
    # Apply gamma correction
    corrected = np.power(normalized, gamma)
    
    # Scale back to original range
    corrected = corrected * (img_max - img_min) + img_min
    
    return corrected


def unsharp_mask(image: np.ndarray,
                radius: float = 1.0,
                amount: float = 1.0) -> np.ndarray:
    """
    Unsharp masking for edge enhancement.
    
    Sharpened = Original + amount * (Original - Blurred)
    
    Parameters
    ----------
    image : array_like
        Input image
    radius : float, optional
        Gaussian blur radius (default=1.0)
    amount : float, optional
        Sharpening strength (default=1.0)
        
    Returns
    -------
    ndarray
        Sharpened image
        
    Examples
    --------
    >>> img = np.random.rand(100, 100)
    >>> sharpened = unsharp_mask(img, radius=2.0, amount=1.5)
    """
    image = np.asarray(image, dtype=float)
    
    # Gaussian blur
    blurred = ndimage.gaussian_filter(image, sigma=radius)
    
    # Unsharp mask
    sharpened = image + amount * (image - blurred)
    
    return sharpened


def denoise(image: np.ndarray,
           method: str = 'gaussian',
           **kwargs) -> np.ndarray:
    """
    Image denoising.
    
    Parameters
    ----------
    image : array_like
        Input image
    method : str, optional
        Denoising method: 'gaussian', 'median', 'bilateral' (default='gaussian')
    **kwargs : dict
        Method-specific parameters
        
    Returns
    -------
    ndarray
        Denoised image
        
    Examples
    --------
    >>> # Add noise
    >>> img = np.random.rand(100, 100)
    >>> noisy = img + np.random.randn(100, 100) * 0.1
    >>> 
    >>> # Denoise
    >>> denoised = denoise(noisy, method='gaussian', sigma=1.0)
    """
    image = np.asarray(image, dtype=float)
    
    match method:
        case 'gaussian':
            sigma = kwargs.get('sigma', 1.0)
            return ndimage.gaussian_filter(image, sigma=sigma)
        
        case 'median':
            size = kwargs.get('size', 3)
            return ndimage.median_filter(image, size=size)
        
        case 'bilateral':
            try:
                from skimage.restoration import denoise_bilateral
                sigma_spatial = kwargs.get('sigma_spatial', 1.0)
                sigma_color = kwargs.get('sigma_color', 0.1)
                return denoise_bilateral(image,
                                        sigma_spatial=sigma_spatial,
                                        sigma_color=sigma_color)
            except ImportError:
                warnings.warn("scikit-image not available, using Gaussian filter")
                return ndimage.gaussian_filter(image, sigma=1.0)
        
        case _:
            raise ValueError(f"Unknown denoising method: {method}")


def remove_periodic_noise(image: np.ndarray,
                         frequency_threshold: Optional[float] = None) -> np.ndarray:
    """
    Remove periodic noise using Fourier filtering.
    
    Useful for removing scan lines, banding, and other periodic artifacts.
    
    Parameters
    ----------
    image : array_like
        Input image
    frequency_threshold : float, optional
        Frequency threshold for filtering
        If None, attempts automatic detection
        
    Returns
    -------
    ndarray
        Filtered image
        
    Examples
    --------
    >>> # Create image with periodic noise
    >>> x, y = np.meshgrid(np.arange(100), np.arange(100))
    >>> img = np.random.rand(100, 100)
    >>> img += 0.2 * np.sin(2 * np.pi * x / 10)  # Periodic noise
    >>> 
    >>> # Remove noise
    >>> filtered = remove_periodic_noise(img)
    """
    image = np.asarray(image, dtype=float)
    
    # FFT
    fft = np.fft.fft2(image)
    fft_shift = np.fft.fftshift(fft)
    
    # Power spectrum
    power = np.abs(fft_shift)**2
    
    if frequency_threshold is None:
        # Automatic threshold: remove frequencies with power > 3*median
        median_power = np.median(power)
        frequency_threshold = 3 * median_power
    
    # Create mask (keep low frequencies, remove high-power periodic components)
    mask = power < frequency_threshold
    
    # Apply mask
    fft_filtered = fft_shift * mask
    
    # Inverse FFT
    fft_ishift = np.fft.ifftshift(fft_filtered)
    filtered = np.fft.ifft2(fft_ishift)
    filtered = np.real(filtered)
    
    return filtered


def atmospheric_correction_dos(image: np.ndarray,
                               dark_object_value: Optional[float] = None) -> np.ndarray:
    """
    Dark Object Subtraction (DOS) for atmospheric correction.
    
    Simple atmospheric correction by subtracting the minimum value
    (assumed to be atmospheric haze).
    
    Parameters
    ----------
    image : array_like
        Input image
    dark_object_value : float, optional
        Value to subtract (if None, uses minimum)
        
    Returns
    -------
    ndarray
        Corrected image
        
    Examples
    --------
    >>> img = np.random.rand(100, 100) * 0.8 + 0.1  # Haze offset
    >>> corrected = atmospheric_correction_dos(img)
    """
    image = np.asarray(image, dtype=float)
    
    if dark_object_value is None:
        # Use 1st percentile to avoid outliers
        dark_object_value = np.nanpercentile(image, 1)
    
    # Subtract dark object
    corrected = image - dark_object_value
    
    # Clip negative values
    corrected = np.maximum(corrected, 0)
    
    return corrected


def pan_sharpen(pan: np.ndarray,
               multispectral: np.ndarray,
               method: str = 'brovey') -> np.ndarray:
    """
    Pan-sharpening to merge high-resolution panchromatic with multispectral.
    
    Parameters
    ----------
    pan : array_like
        Panchromatic image (high resolution)
    multispectral : array_like
        Multispectral image (lower resolution, multiple bands)
        Shape: (height, width, n_bands)
    method : str, optional
        Pan-sharpening method: 'brovey', 'simple' (default='brovey')
        
    Returns
    -------
    ndarray
        Pan-sharpened multispectral image
        
    Examples
    --------
    >>> # Panchromatic: 200x200, Multispectral: 100x100x3
    >>> pan = np.random.rand(200, 200)
    >>> ms = np.random.rand(100, 100, 3)
    >>> 
    >>> # Upsample multispectral to match pan
    >>> from scipy.ndimage import zoom
    >>> ms_upsampled = zoom(ms, (2, 2, 1), order=1)
    >>> 
    >>> # Pan-sharpen
    >>> sharpened = pan_sharpen(pan, ms_upsampled, method='brovey')
    """
    pan = np.asarray(pan, dtype=float)
    multispectral = np.asarray(multispectral, dtype=float)
    
    if pan.shape[:2] != multispectral.shape[:2]:
        raise ValueError("Pan and multispectral must have same spatial dimensions")
    
    match method:
        case 'brovey':
            # Brovey transform
            intensity = np.mean(multispectral, axis=2, keepdims=True)
            sharpened = multispectral * (pan[:, :, np.newaxis] / (intensity + 1e-10))
            
        case 'simple':
            # Simple intensity substitution
            intensity = np.mean(multispectral, axis=2, keepdims=True)
            sharpened = multispectral - intensity + pan[:, :, np.newaxis]
            
        case _:
            raise ValueError(f"Unknown pan-sharpening method: {method}")
    
    return sharpened


def cloud_mask(red: np.ndarray,
              nir: np.ndarray,
              threshold: float = 0.3) -> np.ndarray:
    """
    Simple cloud masking based on reflectance.
    
    Clouds have high reflectance in both red and NIR bands.
    
    Parameters
    ----------
    red : array_like
        Red band reflectance
    nir : array_like
        Near-infrared band reflectance
    threshold : float, optional
        Reflectance threshold for cloud detection (default=0.3)
        
    Returns
    -------
    ndarray
        Boolean mask (True = cloud)
        
    Examples
    --------
    >>> red = np.random.rand(100, 100) * 0.5
    >>> nir = np.random.rand(100, 100) * 0.5
    >>> mask = cloud_mask(red, nir, threshold=0.4)
    >>> print(f"Cloud coverage: {np.sum(mask) / mask.size * 100:.1f}%")
    """
    red = np.asarray(red, dtype=float)
    nir = np.asarray(nir, dtype=float)
    
    # Clouds have high reflectance in both bands
    mask = (red > threshold) & (nir > threshold)
    
    return mask
