"""
Advanced image analysis for earth sciences

Grain size distribution, shape analysis, fabric analysis, and object detection
for microscopy and petrographic images.
"""

import numpy as np
from scipy import ndimage
from typing import Tuple, Optional, Dict, List
import warnings


def grain_size_distribution(image: np.ndarray,
                           threshold_method: str = 'otsu',
                           min_size: int = 10,
                           return_labels: bool = False) -> Dict:
    """
    Calculate grain size distribution from binary or grayscale image.
    
    Automatically segments grains and calculates size statistics.
    
    Parameters
    ----------
    image : array_like
        Input image (grayscale or binary)
    threshold_method : str, optional
        Thresholding method: 'otsu', 'manual', 'adaptive' (default='otsu')
    min_size : int, optional
        Minimum grain size in pixels (default=10)
    return_labels : bool, optional
        Return labeled image (default=False)
        
    Returns
    -------
    dict
        Grain size statistics and distributions
        
    Examples
    --------
    >>> # Load microscope image
    >>> image = np.random.rand(512, 512) * 255
    >>> image = image.astype(np.uint8)
    >>> 
    >>> result = grain_size_distribution(image, threshold_method='otsu')
    >>> print(f"Number of grains: {result['n_grains']}")
    >>> print(f"Mean diameter: {result['mean_diameter']:.1f} pixels")
    >>> print(f"Median diameter: {result['median_diameter']:.1f} pixels")
    """
    try:
        from skimage import filters, morphology, measure
    except ImportError:
        raise ImportError("scikit-image required. Install with: pip install scikit-image")
    
    image = np.asarray(image, dtype=float)
    
    # Thresholding
    if threshold_method == 'otsu':
        threshold = filters.threshold_otsu(image)
        binary = image > threshold
    elif threshold_method == 'adaptive':
        binary = image > filters.threshold_local(image, block_size=35)
    else:
        # Manual threshold at 50% of max
        threshold = np.max(image) / 2
        binary = image > threshold
    
    # Clean up binary image
    binary = morphology.remove_small_objects(binary, min_size=min_size)
    binary = morphology.remove_small_holes(binary, area_threshold=min_size)
    
    # Label connected components
    labeled, n_grains = ndimage.label(binary)
    
    # Measure properties
    props = measure.regionprops(labeled)
    
    # Extract grain sizes (equivalent diameter from area)
    areas = np.array([p.area for p in props])
    diameters = 2 * np.sqrt(areas / np.pi)  # Equivalent circular diameter
    
    # Calculate statistics
    result = {
        'n_grains': n_grains,
        'areas': areas,
        'diameters': diameters,
        'mean_diameter': np.mean(diameters),
        'median_diameter': np.median(diameters),
        'std_diameter': np.std(diameters),
        'min_diameter': np.min(diameters),
        'max_diameter': np.max(diameters),
        'percentiles': {
            '25th': np.percentile(diameters, 25),
            '50th': np.percentile(diameters, 50),
            '75th': np.percentile(diameters, 75),
        }
    }
    
    if return_labels:
        result['labeled_image'] = labeled
        result['binary_image'] = binary
    
    return result


def shape_analysis(binary_image: np.ndarray,
                  min_size: int = 10) -> Dict:
    """
    Analyze shape properties of objects in binary image.
    
    Calculates circularity, elongation, orientation, and other shape metrics.
    
    Parameters
    ----------
    binary_image : array_like
        Binary image with objects
    min_size : int, optional
        Minimum object size (default=10)
        
    Returns
    -------
    dict
        Shape properties for each object
        
    Notes
    -----
    Shape metrics:
    - Circularity: 4π*area/perimeter² (1 = perfect circle)
    - Elongation: major_axis/minor_axis
    - Solidity: area/convex_area
    - Eccentricity: 0 (circle) to 1 (line)
    
    Examples
    --------
    >>> # Create binary image with shapes
    >>> binary = np.zeros((200, 200), dtype=bool)
    >>> # Add circular object
    >>> y, x = np.ogrid[:200, :200]
    >>> mask = (x - 100)**2 + (y - 100)**2 <= 30**2
    >>> binary[mask] = True
    >>> 
    >>> result = shape_analysis(binary)
    >>> print(f"Circularity: {result['circularity'][0]:.3f}")
    """
    try:
        from skimage import morphology, measure
    except ImportError:
        raise ImportError("scikit-image required")
    
    binary_image = np.asarray(binary_image, dtype=bool)
    
    # Clean up
    binary_image = morphology.remove_small_objects(binary_image, min_size=min_size)
    
    # Label objects
    labeled = measure.label(binary_image)
    props = measure.regionprops(labeled)
    
    # Extract shape properties
    n_objects = len(props)
    
    areas = np.array([p.area for p in props])
    perimeters = np.array([p.perimeter for p in props])
    major_axes = np.array([p.major_axis_length for p in props])
    minor_axes = np.array([p.minor_axis_length for p in props])
    orientations = np.array([p.orientation for p in props])
    eccentricities = np.array([p.eccentricity for p in props])
    solidities = np.array([p.solidity for p in props])
    
    # Calculate circularity
    circularity = 4 * np.pi * areas / (perimeters**2)
    
    # Calculate elongation
    elongation = major_axes / (minor_axes + 1e-10)
    
    return {
        'n_objects': n_objects,
        'areas': areas,
        'perimeters': perimeters,
        'circularity': circularity,
        'elongation': elongation,
        'major_axis': major_axes,
        'minor_axis': minor_axes,
        'orientation': orientations,  # radians
        'eccentricity': eccentricities,
        'solidity': solidities,
        'mean_circularity': np.mean(circularity),
        'mean_elongation': np.mean(elongation),
    }


def fabric_analysis(orientations: np.ndarray,
                   weights: Optional[np.ndarray] = None) -> Dict:
    """
    Analyze fabric (preferred orientation) in oriented grains.
    
    Calculates fabric strength and orientation using eigenvalue analysis.
    
    Parameters
    ----------
    orientations : array_like
        Grain orientations in radians
    weights : array_like, optional
        Weights for each orientation (e.g., grain size)
        
    Returns
    -------
    dict
        Fabric strength, preferred orientation, and eigenvalues
        
    Notes
    -----
    Fabric strength indicators:
    - S1/S2 ratio: >3 indicates strong fabric
    - (S1-S2)/(S1+S2): 0 (isotropic) to 1 (perfect alignment)
    
    Examples
    --------
    >>> # Random orientations (no fabric)
    >>> orientations = np.random.uniform(0, np.pi, 100)
    >>> result = fabric_analysis(orientations)
    >>> print(f"Fabric strength: {result['fabric_strength']:.3f}")
    >>> 
    >>> # Preferred orientation (strong fabric)
    >>> orientations = np.random.normal(np.pi/4, 0.1, 100)
    >>> result = fabric_analysis(orientations)
    >>> print(f"Preferred orientation: {np.degrees(result['preferred_orientation']):.1f}°")
    """
    orientations = np.asarray(orientations)
    
    if weights is None:
        weights = np.ones_like(orientations)
    else:
        weights = np.asarray(weights)
    
    # Convert to double angles (for orientation data)
    double_angles = 2 * orientations
    
    # Calculate orientation tensor
    cos_2theta = np.cos(double_angles)
    sin_2theta = np.sin(double_angles)
    
    # Weighted sums
    S_xx = np.sum(weights * cos_2theta**2) / np.sum(weights)
    S_yy = np.sum(weights * sin_2theta**2) / np.sum(weights)
    S_xy = np.sum(weights * cos_2theta * sin_2theta) / np.sum(weights)
    
    # Orientation tensor
    tensor = np.array([[S_xx, S_xy],
                      [S_xy, S_yy]])
    
    # Eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(tensor)
    
    # Sort by eigenvalue
    idx = np.argsort(eigenvalues)[::-1]
    S1, S2 = eigenvalues[idx]
    
    # Preferred orientation (from largest eigenvector)
    preferred_orientation = np.arctan2(eigenvectors[1, idx[0]], eigenvectors[0, idx[0]]) / 2
    
    # Ensure positive angle
    if preferred_orientation < 0:
        preferred_orientation += np.pi
    
    # Fabric strength
    fabric_strength = (S1 - S2) / (S1 + S2) if (S1 + S2) > 0 else 0
    
    return {
        'S1': S1,
        'S2': S2,
        'eigenvalue_ratio': S1 / S2 if S2 > 0 else np.inf,
        'fabric_strength': fabric_strength,
        'preferred_orientation': preferred_orientation,
        'preferred_orientation_deg': np.degrees(preferred_orientation),
        'orientation_tensor': tensor,
    }


def watershed_segmentation(image: np.ndarray,
                          min_distance: int = 10,
                          threshold_rel: float = 0.5) -> Dict:
    """
    Watershed segmentation for separating touching objects.
    
    Useful for separating overlapping grains in microscope images.
    
    Parameters
    ----------
    image : array_like
        Grayscale or binary image
    min_distance : int, optional
        Minimum distance between peaks (default=10)
    threshold_rel : float, optional
        Relative threshold for peak detection (default=0.5)
        
    Returns
    -------
    dict
        Segmented image and object count
        
    Examples
    --------
    >>> # Image with touching objects
    >>> image = np.random.rand(200, 200)
    >>> result = watershed_segmentation(image, min_distance=15)
    >>> print(f"Segmented {result['n_objects']} objects")
    >>> segmented = result['labeled_image']
    """
    try:
        from skimage import filters, feature, morphology, segmentation
    except ImportError:
        raise ImportError("scikit-image required")
    
    image = np.asarray(image, dtype=float)
    
    # Normalize
    image = (image - image.min()) / (image.max() - image.min())
    
    # Threshold
    binary = image > filters.threshold_otsu(image)
    
    # Distance transform
    distance = ndimage.distance_transform_edt(binary)
    
    # Find peaks (local maxima)
    coords = feature.peak_local_max(
        distance,
        min_distance=min_distance,
        threshold_rel=threshold_rel,
        labels=binary
    )
    
    # Create markers
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers = morphology.label(mask)
    
    # Watershed
    labeled = segmentation.watershed(-distance, markers, mask=binary)
    
    n_objects = len(np.unique(labeled)) - 1  # Exclude background
    
    return {
        'labeled_image': labeled,
        'n_objects': n_objects,
        'distance_transform': distance,
        'markers': markers,
    }


def quantify_charcoal(image: np.ndarray,
                     threshold_method: str = 'otsu',
                     min_particle_size: int = 5,
                     pixel_size_mm: Optional[float] = None) -> Dict:
    """
    Quantify charcoal particles in microscope images.
    
    Used in paleoclimate studies to reconstruct fire history.
    
    Parameters
    ----------
    image : array_like
        Grayscale microscope image
    threshold_method : str, optional
        Thresholding method (default='otsu')
    min_particle_size : int, optional
        Minimum particle size in pixels (default=5)
    pixel_size_mm : float, optional
        Pixel size in mm for area calculations
        
    Returns
    -------
    dict
        Charcoal quantification results
        
    Examples
    --------
    >>> # Microscope image at 100x magnification
    >>> image = np.random.rand(512, 512) * 255
    >>> result = quantify_charcoal(image, pixel_size_mm=0.001)
    >>> print(f"Charcoal area fraction: {result['area_fraction']:.2%}")
    >>> print(f"Particle concentration: {result['particle_concentration']:.1f} per mm²")
    """
    try:
        from skimage import filters, morphology, measure
    except ImportError:
        raise ImportError("scikit-image required")
    
    image = np.asarray(image, dtype=float)
    
    # Threshold to identify charcoal (dark particles)
    if threshold_method == 'otsu':
        threshold = filters.threshold_otsu(image)
        binary = image < threshold  # Dark particles
    else:
        threshold = np.mean(image)
        binary = image < threshold
    
    # Clean up
    binary = morphology.remove_small_objects(binary, min_size=min_particle_size)
    binary = morphology.remove_small_holes(binary, area_threshold=min_particle_size)
    
    # Label particles
    labeled = measure.label(binary)
    props = measure.regionprops(labeled)
    
    # Calculate metrics
    n_particles = len(props)
    total_charcoal_area = np.sum(binary)
    total_image_area = image.size
    area_fraction = total_charcoal_area / total_image_area
    
    # Particle sizes
    particle_areas = np.array([p.area for p in props])
    
    result = {
        'n_particles': n_particles,
        'total_charcoal_pixels': total_charcoal_area,
        'area_fraction': area_fraction,
        'mean_particle_size': np.mean(particle_areas) if n_particles > 0 else 0,
        'particle_areas': particle_areas,
    }
    
    # Convert to physical units if pixel size provided
    if pixel_size_mm is not None:
        pixel_area_mm2 = pixel_size_mm ** 2
        image_area_mm2 = total_image_area * pixel_area_mm2
        
        result['particle_concentration'] = n_particles / image_area_mm2  # per mm²
        result['charcoal_area_mm2'] = total_charcoal_area * pixel_area_mm2
        result['image_area_mm2'] = image_area_mm2
    
    return result


def detect_circular_objects(image: np.ndarray,
                           min_radius: int = 10,
                           max_radius: int = 100,
                           sensitivity: float = 0.5) -> Dict:
    """
    Detect circular objects using Hough transform.
    
    Useful for detecting spherical grains, bubbles, or circular features.
    
    Parameters
    ----------
    image : array_like
        Grayscale image
    min_radius : int, optional
        Minimum circle radius (default=10)
    max_radius : int, optional
        Maximum circle radius (default=100)
    sensitivity : float, optional
        Detection sensitivity 0-1 (default=0.5)
        
    Returns
    -------
    dict
        Detected circles (centers and radii)
        
    Examples
    --------
    >>> image = np.zeros((200, 200))
    >>> # Draw some circles
    >>> from skimage.draw import circle_perimeter
    >>> rr, cc = circle_perimeter(100, 100, 30)
    >>> image[rr, cc] = 1
    >>> 
    >>> result = detect_circular_objects(image, min_radius=20, max_radius=40)
    >>> print(f"Detected {result['n_circles']} circles")
    """
    try:
        from skimage import feature, filters
    except ImportError:
        raise ImportError("scikit-image required")
    
    image = np.asarray(image, dtype=float)
    
    # Edge detection
    edges = filters.sobel(image)
    
    # Hough circle detection
    radii = np.arange(min_radius, max_radius + 1)
    hough_res = feature.hough_circle(edges, radii)
    
    # Find peaks
    accums, cx, cy, radii_detected = feature.hough_circle_peaks(
        hough_res, radii, threshold=sensitivity
    )
    
    return {
        'n_circles': len(cx),
        'centers_x': cx,
        'centers_y': cy,
        'radii': radii_detected,
        'accumulator_values': accums,
    }
