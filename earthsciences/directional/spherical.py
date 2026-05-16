"""
Spherical statistics

Analysis of directional data on a sphere (paleomagnetic data, orientations).
"""

import numpy as np


def spherical_mean(theta: np.ndarray, phi: np.ndarray, degrees: bool = True) -> tuple[float, float]:
    """
    Calculate mean direction on a sphere.

    Parameters
    ----------
    theta : array_like
        Colatitude angles (0 to π or 0 to 180°)
    phi : array_like
        Azimuth angles (0 to 2π or 0 to 360°)
    degrees : bool, optional
        If True, angles are in degrees (default=True)

    Returns
    -------
    mean_theta : float
        Mean colatitude
    mean_phi : float
        Mean azimuth

    Examples
    --------
    >>> # Paleomagnetic directions
    >>> theta = np.array([45, 50, 42, 48])  # Colatitude
    >>> phi = np.array([10, 15, 8, 12])     # Azimuth
    >>> mean_theta, mean_phi = spherical_mean(theta, phi, degrees=True)
    >>> print(f"Mean direction: θ={mean_theta:.1f}°, φ={mean_phi:.1f}°")
    """
    theta = np.asarray(theta)
    phi = np.asarray(phi)

    if degrees:
        theta = np.deg2rad(theta)
        phi = np.deg2rad(phi)

    # Convert to Cartesian coordinates
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    # Mean direction
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    z_mean = np.mean(z)

    # Convert back to spherical
    R = np.sqrt(x_mean**2 + y_mean**2 + z_mean**2)

    if R > 0:
        mean_theta = np.arccos(z_mean / R)
        mean_phi = np.arctan2(y_mean, x_mean)
    else:
        # Undefined mean direction
        mean_theta = 0
        mean_phi = 0

    if degrees:
        mean_theta = np.rad2deg(mean_theta)
        mean_phi = np.rad2deg(mean_phi)
        # Ensure positive azimuth
        if mean_phi < 0:
            mean_phi += 360

    return mean_theta, mean_phi


def spherical_variance(theta: np.ndarray, phi: np.ndarray, degrees: bool = True) -> float:
    """
    Calculate spherical variance (1 - R), where R is mean resultant length.

    Parameters
    ----------
    theta : array_like
        Colatitude angles
    phi : array_like
        Azimuth angles
    degrees : bool, optional
        If True, angles are in degrees (default=True)

    Returns
    -------
    float
        Spherical variance (0 to 1)

    Examples
    --------
    >>> theta = np.random.uniform(40, 50, 50)
    >>> phi = np.random.uniform(0, 20, 50)
    >>> var = spherical_variance(theta, phi, degrees=True)
    >>> print(f"Spherical variance: {var:.3f}")
    """
    theta = np.asarray(theta)
    phi = np.asarray(phi)

    if degrees:
        theta = np.deg2rad(theta)
        phi = np.deg2rad(phi)

    # Convert to Cartesian
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    # Mean resultant length
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    z_mean = np.mean(z)

    R = np.sqrt(x_mean**2 + y_mean**2 + z_mean**2)

    return 1 - R


def fisher_distribution(theta: np.ndarray, phi: np.ndarray, degrees: bool = True) -> dict:
    """
    Fit Fisher distribution (spherical analog of von Mises).

    Parameters
    ----------
    theta : array_like
        Colatitude angles
    phi : array_like
        Azimuth angles
    degrees : bool, optional
        If True, angles are in degrees (default=True)

    Returns
    -------
    dict
        Fitted parameters and statistics

    Examples
    --------
    >>> # Generate Fisher-distributed data
    >>> n = 100
    >>> theta = np.random.uniform(40, 50, n)
    >>> phi = np.random.uniform(0, 20, n)
    >>> fit = fisher_distribution(theta, phi, degrees=True)
    >>> print(f"Concentration parameter κ: {fit['kappa']:.2f}")
    """
    theta = np.asarray(theta)
    phi = np.asarray(phi)

    if degrees:
        theta = np.deg2rad(theta)
        phi = np.deg2rad(phi)

    # Convert to Cartesian
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    # Mean direction
    n = len(theta)
    x_mean = np.sum(x) / n
    y_mean = np.sum(y) / n
    z_mean = np.sum(z) / n

    R = np.sqrt(x_mean**2 + y_mean**2 + z_mean**2)

    # Mean direction
    if R > 0:
        mean_theta = np.arccos(z_mean / R)
        mean_phi = np.arctan2(y_mean, x_mean)
    else:
        mean_theta = 0
        mean_phi = 0

    # Estimate concentration parameter kappa
    if R < 1:
        kappa = (n - 1) / (n - R)
    else:
        kappa = np.inf

    # Confidence cone half-angle (95%)
    if R > 0 and kappa < np.inf:
        alpha_95 = np.arccos(1 - ((n - R) / R) * (0.05 ** (1 / (n - 1)) - 1))
    else:
        alpha_95 = 0

    if degrees:
        mean_theta = np.rad2deg(mean_theta)
        mean_phi = np.rad2deg(mean_phi)
        if mean_phi < 0:
            mean_phi += 360
        alpha_95 = np.rad2deg(alpha_95)

    return {
        "mean_theta": mean_theta,
        "mean_phi": mean_phi,
        "kappa": kappa,
        "R": R,
        "alpha_95": alpha_95,
        "n": n,
    }


def kent_distribution(theta: np.ndarray, phi: np.ndarray, degrees: bool = True) -> dict:
    """
    Estimate parameters of Kent distribution (Fisher-Bingham on sphere).

    More general than Fisher, allows for elliptical distributions.

    Parameters
    ----------
    theta : array_like
        Colatitude angles
    phi : array_like
        Azimuth angles
    degrees : bool, optional
        If True, angles are in degrees (default=True)

    Returns
    -------
    dict
        Distribution parameters (simplified estimation)

    Examples
    --------
    >>> theta = np.random.uniform(40, 50, 80)
    >>> phi = np.random.uniform(0, 30, 80)
    >>> fit = kent_distribution(theta, phi, degrees=True)
    """
    theta = np.asarray(theta)
    phi = np.asarray(phi)

    if degrees:
        theta = np.deg2rad(theta)
        phi = np.deg2rad(phi)

    # Convert to Cartesian
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    # Stack into matrix
    coords = np.column_stack([x, y, z])

    # Orientation matrix (simplified)
    T = coords.T @ coords / len(theta)

    # Eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(T)

    # Sort by eigenvalue (descending)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Mean direction (principal eigenvector)
    mean_vec = eigenvectors[:, 0]

    # Convert to spherical
    mean_theta = np.arccos(mean_vec[2])
    mean_phi = np.arctan2(mean_vec[1], mean_vec[0])

    if degrees:
        mean_theta = np.rad2deg(mean_theta)
        mean_phi = np.rad2deg(mean_phi)
        if mean_phi < 0:
            mean_phi += 360

    return {
        "mean_theta": mean_theta,
        "mean_phi": mean_phi,
        "eigenvalues": eigenvalues,
        "principal_direction": mean_vec,
    }


def stereonet_projection(
    theta: np.ndarray, phi: np.ndarray, degrees: bool = True, lower_hemisphere: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """
    Project spherical data onto stereonet (equal-area or equal-angle).

    Used in structural geology for plotting orientations.

    Parameters
    ----------
    theta : array_like
        Colatitude angles
    phi : array_like
        Azimuth angles
    degrees : bool, optional
        If True, angles are in degrees (default=True)
    lower_hemisphere : bool, optional
        Project to lower hemisphere (default=True)

    Returns
    -------
    x, y : ndarray
        Projected coordinates

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> # Generate random orientations
    >>> theta = np.random.uniform(0, 90, 100)
    >>> phi = np.random.uniform(0, 360, 100)
    >>>
    >>> # Project to stereonet
    >>> x, y = stereonet_projection(theta, phi, degrees=True)
    >>>
    >>> # Plot
    >>> fig, ax = plt.subplots(figsize=(8, 8))
    >>> circle = plt.Circle((0, 0), 1, fill=False, color='black')
    >>> ax.add_patch(circle)
    >>> ax.scatter(x, y, alpha=0.5)
    >>> ax.set_aspect('equal')
    >>> ax.set_xlim(-1.1, 1.1)
    >>> ax.set_ylim(-1.1, 1.1)
    >>> plt.show()
    """
    theta = np.asarray(theta)
    phi = np.asarray(phi)

    if degrees:
        theta = np.deg2rad(theta)
        phi = np.deg2rad(phi)

    if lower_hemisphere:
        # Reflect upper hemisphere to lower
        theta = np.where(theta > np.pi / 2, np.pi - theta, theta)

    # Equal-area (Schmidt) projection
    r = np.sqrt(2) * np.sin((np.pi / 2 - theta) / 2)

    x = r * np.sin(phi)
    y = r * np.cos(phi)

    return x, y


def great_circle_distance(
    theta1: float, phi1: float, theta2: float, phi2: float, degrees: bool = True
) -> float:
    """
    Calculate great circle distance between two points on sphere.

    Parameters
    ----------
    theta1, phi1 : float
        First point (colatitude, azimuth)
    theta2, phi2 : float
        Second point (colatitude, azimuth)
    degrees : bool, optional
        If True, angles are in degrees (default=True)

    Returns
    -------
    float
        Great circle distance (angular distance)

    Examples
    --------
    >>> # Distance between north pole and equator
    >>> dist = great_circle_distance(0, 0, 90, 0, degrees=True)
    >>> print(f"Distance: {dist:.1f}°")  # Should be 90°
    """
    if degrees:
        theta1 = np.deg2rad(theta1)
        phi1 = np.deg2rad(phi1)
        theta2 = np.deg2rad(theta2)
        phi2 = np.deg2rad(phi2)

    # Convert to Cartesian
    x1 = np.sin(theta1) * np.cos(phi1)
    y1 = np.sin(theta1) * np.sin(phi1)
    z1 = np.cos(theta1)

    x2 = np.sin(theta2) * np.cos(phi2)
    y2 = np.sin(theta2) * np.sin(phi2)
    z2 = np.cos(theta2)

    # Dot product
    dot_product = x1 * x2 + y1 * y2 + z1 * z2

    # Clamp to [-1, 1] for numerical stability
    dot_product = np.clip(dot_product, -1, 1)

    # Angular distance
    distance = np.arccos(dot_product)

    if degrees:
        distance = np.rad2deg(distance)

    return distance


def resultant_length(theta: np.ndarray, phi: np.ndarray, degrees: bool = True) -> float:
    """
    Calculate resultant length (measure of concentration).

    Parameters
    ----------
    theta : array_like
        Colatitude angles
    phi : array_like
        Azimuth angles
    degrees : bool
        If True, angles are in degrees

    Returns
    -------
    float
        Resultant length (0 to 1)
    """
    theta = np.asarray(theta)
    phi = np.asarray(phi)

    if degrees:
        theta = np.deg2rad(theta)
        phi = np.deg2rad(phi)

    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    x_mean = np.mean(x)
    y_mean = np.mean(y)
    z_mean = np.mean(z)

    R = np.sqrt(x_mean**2 + y_mean**2 + z_mean**2)

    return R


def fisher_pdf(theta: float, phi: float, mean_theta: float, mean_phi: float, kappa: float) -> float:
    """
    Fisher distribution PDF on sphere.

    Parameters
    ----------
    theta, phi : float
        Point on sphere (radians)
    mean_theta, mean_phi : float
        Mean direction (radians)
    kappa : float
        Concentration parameter

    Returns
    -------
    float
        PDF value
    """
    # Angular distance
    cos_dist = np.cos(theta) * np.cos(mean_theta) + np.sin(theta) * np.sin(mean_theta) * np.cos(
        phi - mean_phi
    )

    # Clamp to avoid numerical issues
    cos_dist = np.clip(cos_dist, -1, 1)

    # Fisher PDF
    pdf = (kappa / (4 * np.pi * np.sinh(kappa))) * np.exp(kappa * cos_dist)

    return pdf


def fisher_rvs(
    mean_theta: float, mean_phi: float, kappa: float, size: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample from Fisher distribution.

    Parameters
    ----------
    mean_theta, mean_phi : float
        Mean direction (radians)
    kappa : float
        Concentration parameter
    size : int
        Number of samples

    Returns
    -------
    theta, phi : ndarray
        Sampled angles (radians)
    """
    # Use rejection sampling
    theta_list: list[float] = []
    phi_list: list[float] = []

    while len(theta_list) < size:
        # Sample uniformly on sphere
        u = np.random.uniform(0, 1, size * 3)
        v = np.random.uniform(0, 1, size * 3)

        theta_prop = np.arccos(2 * u - 1)
        phi_prop = 2 * np.pi * v

        # Accept/reject based on Fisher density
        cos_dist = np.cos(theta_prop) * np.cos(mean_theta) + np.sin(theta_prop) * np.sin(
            mean_theta
        ) * np.cos(phi_prop - mean_phi)

        accept_prob = np.exp(kappa * (cos_dist - 1))
        accept = np.random.uniform(0, 1, size * 3) < accept_prob

        theta_list.extend(theta_prop[accept].tolist())
        phi_list.extend(phi_prop[accept].tolist())

    theta_samples = np.asarray(theta_list[:size], dtype=float)
    phi_samples = np.asarray(phi_list[:size], dtype=float)

    return theta_samples, phi_samples


def spherical_kde(theta: np.ndarray, phi: np.ndarray, bandwidth: float = 0.3) -> dict:
    """
    Kernel density estimation on sphere.

    Parameters
    ----------
    theta, phi : array_like
        Data points (radians)
    bandwidth : float
        KDE bandwidth

    Returns
    -------
    dict
        Dictionary with density function
    """
    theta = np.asarray(theta)
    phi = np.asarray(phi)

    def density(theta_eval, phi_eval):
        """Evaluate density at point."""
        total = 0
        for t, p in zip(theta, phi):
            cos_dist = np.cos(theta_eval) * np.cos(t) + np.sin(theta_eval) * np.sin(t) * np.cos(
                phi_eval - p
            )
            cos_dist = np.clip(cos_dist, -1, 1)
            dist = np.arccos(cos_dist)
            total += np.exp(-(dist**2) / (2 * bandwidth**2))

        return total / (len(theta) * 2 * np.pi * bandwidth**2)

    return {"density": density, "pdf": density}


def rotation_matrix_from_axis_angle(
    axis: np.ndarray, angle: float, degrees: bool = True
) -> np.ndarray:
    """
    Create rotation matrix from axis and angle (Rodrigues' formula).

    Useful for rotating paleomagnetic or structural data.

    Parameters
    ----------
    axis : array_like
        Rotation axis (3D unit vector)
    angle : float
        Rotation angle
    degrees : bool, optional
        If True, angle is in degrees (default=True)

    Returns
    -------
    ndarray
        3x3 rotation matrix

    Examples
    --------
    >>> # Rotate around z-axis by 45 degrees
    >>> axis = np.array([0, 0, 1])
    >>> R = rotation_matrix_from_axis_angle(axis, 45, degrees=True)
    >>>
    >>> # Apply rotation to a vector
    >>> v = np.array([1, 0, 0])
    >>> v_rotated = R @ v
    """
    axis = np.asarray(axis)
    axis = axis / np.linalg.norm(axis)  # Normalize

    if degrees:
        angle = np.deg2rad(angle)

    # Rodrigues' rotation formula
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])

    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

    return R
