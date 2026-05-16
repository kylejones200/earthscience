"""
Circular statistics

Analysis of directional data on a circle (angles, azimuths, wind directions).
"""

import numpy as np
from scipy import stats


def circular_mean(angles: np.ndarray, degrees: bool = True) -> float:
    """
    Calculate mean direction (circular mean).

    Parameters
    ----------
    angles : array_like
        Angles
    degrees : bool, optional
        If True, angles are in degrees (default=True)

    Returns
    -------
    float
        Mean direction

    Examples
    --------
    >>> # Wind directions
    >>> directions = np.array([10, 350, 5, 15, 355])  # Clustered around north
    >>> mean_dir = circular_mean(directions, degrees=True)
    >>> print(f"Mean direction: {mean_dir:.1f}°")
    """
    angles = np.asarray(angles)

    if degrees:
        angles = np.deg2rad(angles)

    # Compute mean of unit vectors
    sin_sum = np.sum(np.sin(angles))
    cos_sum = np.sum(np.cos(angles))

    mean_angle = np.arctan2(sin_sum, cos_sum)

    if degrees:
        mean_angle = np.rad2deg(mean_angle)
        # Ensure positive angle
        if mean_angle < 0:
            mean_angle += 360

    return mean_angle


def circular_variance(angles: np.ndarray, degrees: bool = True) -> float:
    """
    Calculate circular variance (1 - R), where R is mean resultant length.

    Ranges from 0 (no variance) to 1 (uniform distribution).

    Parameters
    ----------
    angles : array_like
        Angles
    degrees : bool, optional
        If True, angles are in degrees (default=True)

    Returns
    -------
    float
        Circular variance

    Examples
    --------
    >>> angles = np.array([0, 10, 350, 5, 355])
    >>> var = circular_variance(angles, degrees=True)
    >>> print(f"Circular variance: {var:.3f}")
    """
    angles = np.asarray(angles)

    if degrees:
        angles = np.deg2rad(angles)

    # Mean resultant length
    sin_sum = np.sum(np.sin(angles))
    cos_sum = np.sum(np.cos(angles))
    R = np.sqrt(sin_sum**2 + cos_sum**2) / len(angles)

    # Circular variance
    return 1 - R


def circular_std(angles: np.ndarray, degrees: bool = True) -> float:
    """
    Calculate circular standard deviation.

    Parameters
    ----------
    angles : array_like
        Angles
    degrees : bool, optional
        If True, angles are in degrees (default=True)

    Returns
    -------
    float
        Circular standard deviation

    Examples
    --------
    >>> angles = np.random.vonmises(0, 2, 100)  # Concentrated around 0
    >>> std = circular_std(angles, degrees=False)
    >>> print(f"Circular std: {std:.3f} radians")
    """
    angles = np.asarray(angles)

    if degrees:
        angles = np.deg2rad(angles)

    # Mean resultant length
    sin_sum = np.sum(np.sin(angles))
    cos_sum = np.sum(np.cos(angles))
    R = np.sqrt(sin_sum**2 + cos_sum**2) / len(angles)

    # Circular standard deviation
    std = np.sqrt(-2 * np.log(R))

    if degrees:
        std = np.rad2deg(std)

    return std


def rayleigh_test(angles: np.ndarray, degrees: bool = True) -> dict:
    """
    Rayleigh test for uniformity of circular data.

    Tests null hypothesis that data is uniformly distributed.

    Parameters
    ----------
    angles : array_like
        Angles
    degrees : bool, optional
        If True, angles are in degrees (default=True)

    Returns
    -------
    dict
        Test results with statistic, p-value, and mean direction

    Examples
    --------
    >>> # Uniform distribution (should fail to reject H0)
    >>> uniform_angles = np.random.uniform(0, 360, 100)
    >>> result_uniform = rayleigh_test(uniform_angles)
    >>> print(f"Uniform: p = {result_uniform['p_value']:.3f}")
    >>>
    >>> # Concentrated distribution (should reject H0)
    >>> concentrated = np.random.vonmises(0, 3, 100)
    >>> result_conc = rayleigh_test(concentrated, degrees=False)
    >>> print(f"Concentrated: p = {result_conc['p_value']:.3f}")
    """
    angles = np.asarray(angles)

    if degrees:
        angles = np.deg2rad(angles)

    n = len(angles)

    # Mean resultant length
    sin_sum = np.sum(np.sin(angles))
    cos_sum = np.sum(np.cos(angles))
    R = np.sqrt(sin_sum**2 + cos_sum**2) / n

    # Rayleigh test statistic
    Z = n * R**2

    # P-value (approximation for n >= 10)
    if n >= 10:
        p_value = np.exp(-Z)
    else:
        # Exact formula for small n
        p_value = np.exp(-Z) * (
            1
            + (2 * Z - Z**2) / (4 * n)
            - (24 * Z - 132 * Z**2 + 76 * Z**3 - 9 * Z**4) / (288 * n**2)
        )

    # Mean direction
    mean_dir = np.arctan2(sin_sum, cos_sum)
    if degrees:
        mean_dir = np.rad2deg(mean_dir)
        if mean_dir < 0:
            mean_dir += 360

    return {
        "statistic": Z,
        "p_value": p_value,
        "mean_resultant_length": R,
        "mean_direction": mean_dir,
        "significant": p_value < 0.05,
    }


def von_mises_fit(angles: np.ndarray, degrees: bool = True) -> dict:
    """
    Fit von Mises distribution (circular analog of normal distribution).

    Parameters
    ----------
    angles : array_like
        Angles
    degrees : bool, optional
        If True, angles are in degrees (default=True)

    Returns
    -------
    dict
        Fitted parameters (mu, kappa) and distribution object

    Examples
    --------
    >>> # Generate von Mises data
    >>> true_mu, true_kappa = 0, 2
    >>> angles = np.random.vonmises(true_mu, true_kappa, 200)
    >>>
    >>> # Fit distribution
    >>> fit = von_mises_fit(angles, degrees=False)
    >>> print(f"Estimated mu: {fit['mu']:.3f}")
    >>> print(f"Estimated kappa: {fit['kappa']:.3f}")
    """
    angles = np.asarray(angles)

    if degrees:
        angles = np.deg2rad(angles)

    # Estimate mu (circular mean)
    sin_sum = np.sum(np.sin(angles))
    cos_sum = np.sum(np.cos(angles))
    mu = np.arctan2(sin_sum, cos_sum)

    # Estimate kappa (concentration parameter)
    R = np.sqrt(sin_sum**2 + cos_sum**2) / len(angles)

    # Approximation for kappa
    if R < 0.53:
        kappa = 2 * R + R**3 + 5 * R**5 / 6
    elif R < 0.85:
        kappa = -0.4 + 1.39 * R + 0.43 / (1 - R)
    else:
        kappa = 1 / (2 * (1 - R))

    if degrees:
        mu_degrees = np.rad2deg(mu)
        if mu_degrees < 0:
            mu_degrees += 360
    else:
        mu_degrees = mu

    return {
        "mu": mu_degrees if degrees else mu,
        "kappa": kappa,
        "mean_resultant_length": R,
    }


def rose_diagram_data(angles: np.ndarray, n_bins: int = 16, degrees: bool = True) -> dict:
    """
    Prepare data for rose diagram (circular histogram).

    Parameters
    ----------
    angles : array_like
        Angles
    n_bins : int, optional
        Number of bins (default=16)
    degrees : bool, optional
        If True, angles are in degrees (default=True)

    Returns
    -------
    dict
        Bin centers, counts, and frequencies

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> # Wind directions
    >>> winds = np.random.vonmises(np.deg2rad(45), 2, 200)
    >>> rose_data = rose_diagram_data(winds, n_bins=16, degrees=False)
    >>>
    >>> # Plot rose diagram
    >>> fig = plt.figure(figsize=(8, 8))
    >>> ax = fig.add_subplot(111, projection='polar')
    >>> ax.bar(rose_data['bin_centers_rad'], rose_data['frequencies'],
    ...       width=rose_data['bin_width'], alpha=0.7)
    >>> ax.set_theta_zero_location('N')
    >>> ax.set_theta_direction(-1)
    >>> plt.show()
    """
    angles = np.asarray(angles)

    if degrees:
        angles_rad = np.deg2rad(angles)
    else:
        angles_rad = angles

    # Ensure angles are in [0, 2π)
    angles_rad = np.mod(angles_rad, 2 * np.pi)

    # Create bins
    bin_edges = np.linspace(0, 2 * np.pi, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = 2 * np.pi / n_bins

    # Histogram
    counts, _ = np.histogram(angles_rad, bins=bin_edges)
    frequencies = counts / len(angles)

    return {
        "bin_centers_rad": bin_centers,
        "bin_centers_deg": np.rad2deg(bin_centers),
        "bin_width": bin_width,
        "counts": counts,
        "frequencies": frequencies,
        "n_bins": n_bins,
    }


def watson_u2_test(angles1: np.ndarray, angles2: np.ndarray, degrees: bool = True) -> dict:
    """
    Watson's U² test for comparing two circular distributions.

    Tests if two samples come from the same circular distribution using
    Watson's U² statistic with chi-square approximation for p-values.

    .. note::
        This implementation uses an asymptotic chi-square approximation
        for p-values which may be less accurate for small samples (n < 20).
        For small samples, consider using permutation tests or bootstrap methods.

    Parameters
    ----------
    angles1, angles2 : array_like
        Two samples of angles
    degrees : bool, optional
        If True, angles are in degrees (default=True)

    Returns
    -------
    dict
        Dictionary containing:
        - 'statistic': Watson's U² test statistic
        - 'p_value': Approximate p-value (asymptotic chi-square)
        - 'n1', 'n2': Sample sizes
        - 'note': Warning about approximation

    Notes
    -----
    The Watson U² test is a non-parametric test for comparing two circular
    distributions. The null hypothesis is that both samples come from the
    same distribution.

    The p-value is calculated using an asymptotic chi-square approximation
    with 2 degrees of freedom. This approximation is valid for larger samples
    but may be inaccurate for small samples.

    References
    ----------
    .. [1] Mardia, K. V., & Jupp, P. E. (2000). Directional Statistics.
           John Wiley & Sons.

    Examples
    --------
    >>> import numpy as np
    >>> # Two samples from similar distributions
    >>> sample1 = np.random.vonmises(0, 2, 50)
    >>> sample2 = np.random.vonmises(0, 2, 50)
    >>> result = watson_u2_test(sample1, sample2, degrees=False)
    >>> print(f"U²={result['statistic']:.4f}, p={result['p_value']:.4f}")
    """

    angles1 = np.asarray(angles1)
    angles2 = np.asarray(angles2)

    if degrees:
        angles1 = np.deg2rad(angles1)
        angles2 = np.deg2rad(angles2)

    # Ensure angles in [0, 2π)
    angles1 = np.mod(angles1, 2 * np.pi)
    angles2 = np.mod(angles2, 2 * np.pi)

    n1 = len(angles1)
    n2 = len(angles2)
    n = n1 + n2

    # Combined sample
    all_angles = np.concatenate([angles1, angles2])
    sorted_angles = np.sort(all_angles)

    # Calculate U² statistic using ECDFs
    F1 = np.searchsorted(np.sort(angles1), sorted_angles, side="right") / n1
    F2 = np.searchsorted(np.sort(angles2), sorted_angles, side="right") / n2

    U2 = (n1 * n2 / n**2) * np.sum((F1 - F2) ** 2)

    # Approximate p-value using asymptotic chi-square distribution
    p_value = 1 - stats.chi2.cdf(2 * n * U2, df=2)

    return {
        "statistic": U2,
        "p_value": p_value,
        "n1": n1,
        "n2": n2,
        "note": "Approximate p-value; use cautiously for small samples",
    }


def angular_distance(angle1: float, angle2: float, degrees: bool = True) -> float:
    """
    Calculate the shortest angular distance between two angles.

    Parameters
    ----------
    angle1, angle2 : float
        Two angles
    degrees : bool, optional
        If True, angles are in degrees (default=True)

    Returns
    -------
    float
        Shortest angular distance

    Examples
    --------
    >>> # Distance between 10° and 350° is 20°, not 340°
    >>> dist = angular_distance(10, 350, degrees=True)
    >>> print(f"Angular distance: {dist:.1f}°")
    """
    if degrees:
        angle1 = np.deg2rad(angle1)
        angle2 = np.deg2rad(angle2)

    # Calculate difference
    diff = np.abs(angle1 - angle2)

    # Take shorter path around circle
    dist = min(diff, 2 * np.pi - diff)

    if degrees:
        dist = np.rad2deg(dist)

    return dist


def kuiper_test(angles1: np.ndarray, angles2: np.ndarray, degrees: bool = True) -> dict:
    """
    Kuiper's test for comparing two circular distributions.

    Similar to Watson's U² but uses maximum deviation instead of integral.
    More sensitive to differences in location than dispersion.

    Parameters
    ----------
    angles1, angles2 : array_like
        Two samples of angles
    degrees : bool, optional
        If True, angles are in degrees (default=True)

    Returns
    -------
    dict
        Test results including Kuiper statistic and approximate p-value

    Examples
    --------
    >>> sample1 = np.random.vonmises(0, 2, 50)
    >>> sample2 = np.random.vonmises(np.pi/6, 2, 50)
    >>> result = kuiper_test(sample1, sample2, degrees=False)
    >>> print(f"V = {result['statistic']:.4f}, p = {result['p_value']:.4f}")
    """

    angles1 = np.asarray(angles1)
    angles2 = np.asarray(angles2)

    if degrees:
        angles1 = np.deg2rad(angles1)
        angles2 = np.deg2rad(angles2)

    # Ensure angles in [0, 2π)
    angles1 = np.mod(angles1, 2 * np.pi)
    angles2 = np.mod(angles2, 2 * np.pi)

    n1 = len(angles1)
    n2 = len(angles2)

    # Sort angles
    sorted1 = np.sort(angles1)
    sorted2 = np.sort(angles2)

    # Combined sorted angles
    all_angles = np.sort(np.concatenate([angles1, angles2]))

    # ECDFs
    F1 = np.searchsorted(sorted1, all_angles, side="right") / n1
    F2 = np.searchsorted(sorted2, all_angles, side="right") / n2

    # Kuiper statistic: D+ + D-
    D_plus = np.max(F1 - F2)
    D_minus = np.max(F2 - F1)
    V = D_plus + D_minus

    # Approximate p-value (Kuiper 1960)
    # For large samples: sqrt(n1*n2/(n1+n2)) * V ~ Kuiper distribution
    n_eff = np.sqrt(n1 * n2 / (n1 + n2))
    V_scaled = n_eff * V

    # Approximation using exponential tail
    # This is a rough approximation
    p_value = 2 * np.exp(-2 * V_scaled**2)
    p_value = min(p_value, 1.0)

    return {
        "statistic": V,
        "V_scaled": V_scaled,
        "p_value": p_value,
        "D_plus": D_plus,
        "D_minus": D_minus,
        "n1": n1,
        "n2": n2,
        "note": "Approximate p-value using asymptotic distribution",
    }


def wheeler_watson_test(angles1: np.ndarray, angles2: np.ndarray, degrees: bool = True) -> dict:
    """
    Wheeler-Watson test for common mean direction.

    Tests if two circular samples have the same mean direction.
    More powerful than Watson U² for detecting location shifts.

    Parameters
    ----------
    angles1, angles2 : array_like
        Two samples of angles
    degrees : bool, optional
        If True, angles are in degrees (default=True)

    Returns
    -------
    dict
        Test results including W statistic and approximate p-value

    Examples
    --------
    >>> # Two samples with similar concentration but different means
    >>> sample1 = np.random.vonmises(0, 3, 40)
    >>> sample2 = np.random.vonmises(np.pi/4, 3, 40)
    >>> result = wheeler_watson_test(sample1, sample2, degrees=False)
    >>> print(f"W = {result['statistic']:.4f}, p = {result['p_value']:.4f}")
    """

    angles1 = np.asarray(angles1)
    angles2 = np.asarray(angles2)

    if degrees:
        angles1 = np.deg2rad(angles1)
        angles2 = np.deg2rad(angles2)

    n1 = len(angles1)
    n2 = len(angles2)

    # Calculate resultant vectors for each sample
    C1 = np.sum(np.cos(angles1))
    S1 = np.sum(np.sin(angles1))
    R1 = np.sqrt(C1**2 + S1**2)

    C2 = np.sum(np.cos(angles2))
    S2 = np.sum(np.sin(angles2))
    R2 = np.sqrt(C2**2 + S2**2)

    # Combined sample
    C_total = C1 + C2
    S_total = S1 + S2
    R_total = np.sqrt(C_total**2 + S_total**2)

    # Wheeler-Watson W statistic
    # W = 2(R1 + R2 - R_total)
    W = 2 * (R1 + R2 - R_total)

    # For large samples with high concentration,
    # W approximately follows chi-square(2) distribution
    p_value = 1 - stats.chi2.cdf(W, df=2)

    return {
        "statistic": W,
        "p_value": p_value,
        "R1": R1,
        "R2": R2,
        "R_total": R_total,
        "n1": n1,
        "n2": n2,
        "note": "Approximate p-value; assumes high concentration",
    }
