"""
Orientation and fabric analysis for structural geology

Strain ellipsoids, fabric tensors, and orientation statistics.
"""

import numpy as np


def fabric_tensor(orientations: np.ndarray, weights: np.ndarray | None = None) -> dict:
    """
    Calculate fabric tensor (orientation tensor) from directional data.

    Used in structural geology, sedimentology, and material science.

    Parameters
    ----------
    orientations : array_like
        Unit vectors (n_measurements, 3)
    weights : array_like, optional
        Weights for each measurement

    Returns
    -------
    dict
        Fabric tensor and derived properties

    Examples
    --------
    >>> # Random orientations on hemisphere
    >>> n = 100
    >>> theta = np.random.uniform(0, np.pi/2, n)
    >>> phi = np.random.uniform(0, 2*np.pi, n)
    >>> orientations = np.column_stack([
    ...     np.sin(theta) * np.cos(phi),
    ...     np.sin(theta) * np.sin(phi),
    ...     np.cos(theta)
    ... ])
    >>> result = fabric_tensor(orientations)
    >>> print(f"Fabric type: {result['fabric_type']}")
    """
    orientations = np.asarray(orientations)

    if weights is None:
        weights = np.ones(len(orientations))

    weights = np.asarray(weights)
    weights = weights / np.sum(weights)  # Normalize

    # Calculate fabric tensor
    T = np.zeros((3, 3))
    for i, ori in enumerate(orientations):
        T += weights[i] * np.outer(ori, ori)

    # Eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(T)

    # Sort by eigenvalue (descending)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Normalize eigenvalues
    S1, S2, S3 = eigenvalues

    # Fabric shape parameters
    # L = lineation strength
    # F = foliation strength
    # P = degree of anisotropy
    L = (S1 - S2) / S1 if S1 > 0 else 0
    F = (S2 - S3) / S1 if S1 > 0 else 0
    P = (S1 - S3) / S1 if S1 > 0 else 0

    # Classify fabric type
    if L > 2 * F:
        fabric_type = "Linear (L-fabric)"
    elif F > 2 * L:
        fabric_type = "Planar (S-fabric)"
    else:
        fabric_type = "Mixed (L-S fabric)"

    return {
        "tensor": T,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "S1": S1,
        "S2": S2,
        "S3": S3,
        "L": L,  # Lineation
        "F": F,  # Foliation
        "P": P,  # Anisotropy
        "fabric_type": fabric_type,
        "principal_direction": eigenvectors[:, 0],
    }


def strain_ellipsoid(orientations_before: np.ndarray, orientations_after: np.ndarray) -> dict:
    """
    Calculate strain ellipsoid from deformed and undeformed orientations.

    Parameters
    ----------
    orientations_before : array_like
        Original orientations (n_markers, 3)
    orientations_after : array_like
        Deformed orientations (n_markers, 3)

    Returns
    -------
    dict
        Strain ellipsoid parameters

    Examples
    --------
    >>> # Simulated simple shear
    >>> n = 50
    >>> original = np.random.randn(n, 3)
    >>> original /= np.linalg.norm(original, axis=1, keepdims=True)
    >>>
    >>> # Apply shear deformation
    >>> shear_matrix = np.array([[1, 0.5, 0],
    ...                          [0, 1, 0],
    ...                          [0, 0, 1]])
    >>> deformed = original @ shear_matrix.T
    >>> deformed /= np.linalg.norm(deformed, axis=1, keepdims=True)
    >>>
    >>> result = strain_ellipsoid(original, deformed)
    >>> print(f"Principal strains: {result['principal_strains']}")
    """
    orientations_before = np.asarray(orientations_before)
    orientations_after = np.asarray(orientations_after)

    # Estimate deformation gradient tensor using least squares
    F, _, _, _ = np.linalg.lstsq(orientations_before, orientations_after, rcond=None)

    # Right Cauchy-Green tensor
    C = F.T @ F

    # Principal stretches (eigenvalues of C)
    principal_strains_squared, principal_directions = np.linalg.eigh(C)
    principal_strains = np.sqrt(np.abs(principal_strains_squared))

    # Sort by magnitude
    idx = principal_strains.argsort()[::-1]
    principal_strains = principal_strains[idx]
    principal_directions = principal_directions[:, idx]

    # Strain ratios
    X, Y, Z = principal_strains
    Rxz = X / Z if Z > 0 else np.inf
    Ryz = Y / Z if Z > 0 else np.inf

    # Volume change
    volume_change = np.linalg.det(F)

    return {
        "deformation_gradient": F,
        "right_cauchy_green": C,
        "principal_strains": principal_strains,
        "principal_directions": principal_directions,
        "X": X,  # Maximum stretch
        "Y": Y,  # Intermediate stretch
        "Z": Z,  # Minimum stretch
        "Rxz": Rxz,  # X/Z ratio
        "Ryz": Ryz,  # Y/Z ratio
        "volume_change": volume_change,
    }


def woodcock_diagram(eigenvalues: np.ndarray) -> dict:
    """
    Calculate Woodcock parameters for fabric shape classification.

    Parameters
    ----------
    eigenvalues : array_like
        Eigenvalues of fabric tensor (sorted, S1 >= S2 >= S3)

    Returns
    -------
    dict
        Woodcock C and K parameters

    Examples
    --------
    >>> # Clustered (girdle) fabric
    >>> S = np.array([0.7, 0.2, 0.1])
    >>> result = woodcock_diagram(S)
    >>> print(f"C = {result['C']:.2f}, K = {result['K']:.2f}")
    >>> print(f"Shape: {result['shape']}")
    """
    S1, S2, S3 = eigenvalues

    # Woodcock parameters
    # C = ln(S1/S2) - shape strength
    # K = ln(S2/S3) - shape type
    with np.errstate(divide="ignore", invalid="ignore"):
        C = np.log(S1 / S2) if S2 > 0 else np.inf
        K = np.log(S2 / S3) if S3 > 0 else np.inf

    # Classify shape
    if K > 1.2 * C:
        shape = "Girdle (planar fabric)"
    elif C > 1.2 * K:
        shape = "Cluster (linear fabric)"
    else:
        shape = "Intermediate"

    return {
        "C": C,
        "K": K,
        "shape": shape,
    }


def stereonet_contours(
    theta: np.ndarray, phi: np.ndarray, grid_size: int = 50, bandwidth: float | None = None
) -> dict:
    """
    Calculate density contours for stereonet plotting.

    Parameters
    ----------
    theta : array_like
        Colatitude angles
    phi : array_like
        Azimuth angles
    grid_size : int, optional
        Grid resolution (default=50)
    bandwidth : float, optional
        Kernel bandwidth for density estimation

    Returns
    -------
    dict
        Density grid for contour plotting

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> # Random orientations
    >>> theta = np.random.uniform(0, np.pi/2, 200)
    >>> phi = np.random.uniform(0, 2*np.pi, 200)
    >>>
    >>> result = stereonet_contours(theta, phi)
    >>>
    >>> # Plot contours
    >>> fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
    >>> ax.contourf(result['phi_grid'], result['theta_grid'],
    ...            result['density'], levels=10)
    >>> plt.show()
    """
    from scipy.stats import gaussian_kde

    theta = np.asarray(theta)
    phi = np.asarray(phi)

    # Create grid
    theta_grid = np.linspace(0, np.pi / 2, grid_size)
    phi_grid = np.linspace(0, 2 * np.pi, grid_size)
    THETA, PHI = np.meshgrid(theta_grid, phi_grid)

    # Flatten for KDE
    positions = np.vstack([theta.ravel(), phi.ravel()])

    # Kernel density estimation
    if bandwidth is None:
        kde = gaussian_kde(positions)
    else:
        kde = gaussian_kde(positions, bw_method=bandwidth)

    # Evaluate on grid
    grid_positions = np.vstack([THETA.ravel(), PHI.ravel()])
    density = kde(grid_positions).reshape(THETA.shape)

    return {
        "theta_grid": THETA,
        "phi_grid": PHI,
        "density": density,
        "theta_edges": theta_grid,
        "phi_edges": phi_grid,
    }


def quartz_c_axis_fabric(orientations: np.ndarray) -> dict:
    """
    Analyze quartz c-axis orientations for strain analysis.

    Parameters
    ----------
    orientations : array_like
        C-axis orientations (n_grains, 3) as unit vectors

    Returns
    -------
    dict
        Fabric analysis results

    Examples
    --------
    >>> # Simulated c-axis data
    >>> n = 150
    >>> # Clustered around vertical with some spread
    >>> theta = np.random.normal(0, 0.2, n)
    >>> phi = np.random.uniform(0, 2*np.pi, n)
    >>> orientations = np.column_stack([
    ...     np.sin(theta) * np.cos(phi),
    ...     np.sin(theta) * np.sin(phi),
    ...     np.cos(theta)
    ... ])
    >>> result = quartz_c_axis_fabric(orientations)
    >>> print(f"Fabric strength: {result['fabric_strength']:.2f}")
    """
    # Calculate fabric tensor
    fabric_result = fabric_tensor(orientations)

    # Opening angle (measure of fabric sharpness)
    S1 = fabric_result["S1"]
    opening_angle = np.rad2deg(np.arccos(np.sqrt(S1)))

    # Fabric strength (0 = random, 1 = perfect alignment)
    P = fabric_result["P"]

    return {
        "fabric_tensor_result": fabric_result,
        "opening_angle": opening_angle,
        "fabric_strength": P,
        "principal_direction": fabric_result["principal_direction"],
        "fabric_type": fabric_result["fabric_type"],
    }


def pole_to_plane(strike: float, dip: float, degrees: bool = True) -> np.ndarray:
    """
    Convert plane orientation (strike/dip) to pole (normal vector).

    Parameters
    ----------
    strike : float
        Strike angle
    dip : float
        Dip angle
    degrees : bool, optional
        Angles in degrees (default=True)

    Returns
    -------
    ndarray
        Unit vector perpendicular to plane

    Examples
    --------
    >>> # N-S striking plane dipping 45° to east
    >>> pole = pole_to_plane(0, 45, degrees=True)
    >>> print(f"Pole direction: {pole}")
    """
    if degrees:
        strike = np.deg2rad(strike)
        dip = np.deg2rad(dip)

    # Calculate pole (normal to plane)
    # Strike is measured clockwise from north
    # Dip is downward from horizontal

    trend = strike + np.pi / 2  # Trend perpendicular to strike
    plunge = np.pi / 2 - dip  # Plunge from horizontal

    # Convert to Cartesian
    x = np.sin(plunge) * np.sin(trend)
    y = np.sin(plunge) * np.cos(trend)
    z = np.cos(plunge)

    pole = np.array([x, y, z])
    pole = pole / np.linalg.norm(pole)  # Normalize

    return pole


def plane_to_pole(x: float, y: float, z: float) -> tuple[float, float]:
    """
    Convert pole (unit vector) to plane orientation (strike/dip).

    Parameters
    ----------
    x, y, z : float
        Components of pole unit vector

    Returns
    -------
    strike : float
        Strike angle in degrees
    dip : float
        Dip angle in degrees

    Examples
    --------
    >>> # Pole pointing east and down
    >>> strike, dip = plane_to_pole(0.707, 0, 0.707)
    >>> print(f"Strike: {strike:.1f}°, Dip: {dip:.1f}°")
    """
    # Normalize
    norm = np.sqrt(x**2 + y**2 + z**2)
    x, y, z = x / norm, y / norm, z / norm

    # Calculate plunge and trend of pole
    plunge = np.arccos(z)
    trend = np.arctan2(x, y)

    # Convert to strike and dip
    dip = np.pi / 2 - plunge
    strike = trend - np.pi / 2

    # Normalize to [0, 2π)
    strike = np.mod(strike, 2 * np.pi)

    # Convert to degrees
    strike_deg = np.rad2deg(strike)
    dip_deg = np.rad2deg(dip)

    return strike_deg, dip_deg
