"""
Nonlinear time-series analysis

Chaos theory, recurrence plots, and nonlinear dynamics.
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform


def time_delay_embedding(data: np.ndarray, delay: int, dimension: int) -> np.ndarray:
    """
    Time-delay embedding for phase space reconstruction.

    Parameters
    ----------
    data : array_like
        Input time series
    delay : int
        Time delay
    dimension : int
        Embedding dimension

    Returns
    -------
    ndarray
        Embedded data (n_points, dimension)

    Examples
    --------
    >>> # Lorenz system
    >>> t = np.linspace(0, 100, 10000)
    >>> data = np.sin(t) + 0.1*np.sin(11*t)  # Quasi-periodic
    >>> embedded = time_delay_embedding(data, delay=10, dimension=3)
    >>> print(f"Embedded shape: {embedded.shape}")
    """
    data = np.asarray(data)
    n = len(data) - (dimension - 1) * delay

    if n <= 0:
        raise ValueError("Data too short for given delay and dimension")

    embedded = np.zeros((n, dimension))
    for i in range(dimension):
        embedded[:, i] = data[i * delay : i * delay + n]

    return embedded


def mutual_information(
    data: np.ndarray, max_delay: int = 50, bins: int = 50
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate mutual information for optimal delay selection.

    First minimum of MI suggests good delay for embedding.

    Parameters
    ----------
    data : array_like
        Input time series
    max_delay : int, optional
        Maximum delay to test (default=50)
    bins : int, optional
        Number of bins for histogram (default=50)

    Returns
    -------
    delays : ndarray
        Delay values
    mi : ndarray
        Mutual information values

    Examples
    --------
    >>> data = np.sin(np.linspace(0, 100, 1000))
    >>> delays, mi = mutual_information(data, max_delay=50)
    >>> optimal_delay = delays[np.argmin(mi[1:])+1]  # First minimum
    >>> print(f"Optimal delay: {optimal_delay}")
    """
    data = np.asarray(data)

    delays = np.arange(0, max_delay + 1)
    mi = np.zeros(len(delays))

    # Normalize data
    data_norm = (data - np.min(data)) / (np.max(data) - np.min(data))

    for i, delay in enumerate(delays):
        if delay == 0:
            mi[i] = np.inf
            continue

        # Create delayed series
        x = data_norm[:-delay]
        y = data_norm[delay:]

        # 2D histogram
        hist_2d, _, _ = np.histogram2d(x, y, bins=bins)
        hist_2d = hist_2d / np.sum(hist_2d)  # Normalize

        # Marginal distributions
        px = np.sum(hist_2d, axis=1)
        py = np.sum(hist_2d, axis=0)

        # Mutual information
        mi_val = 0
        for ix in range(bins):
            for iy in range(bins):
                if hist_2d[ix, iy] > 0 and px[ix] > 0 and py[iy] > 0:
                    mi_val += hist_2d[ix, iy] * np.log(hist_2d[ix, iy] / (px[ix] * py[iy]))

        mi[i] = mi_val

    return delays, mi


def false_nearest_neighbors(
    data: np.ndarray, max_dimension: int = 10, delay: int = 1, rtol: float = 15.0, atol: float = 2.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    False nearest neighbors method for determining embedding dimension.

    Parameters
    ----------
    data : array_like
        Input time series
    max_dimension : int, optional
        Maximum embedding dimension to test (default=10)
    delay : int, optional
        Time delay (default=1)
    rtol : float, optional
        Relative tolerance (default=15.0)
    atol : float, optional
        Absolute tolerance (default=2.0)

    Returns
    -------
    dimensions : ndarray
        Embedding dimensions tested
    fnn_percent : ndarray
        Percentage of false nearest neighbors

    Examples
    --------
    >>> data = np.sin(np.linspace(0, 100, 1000))
    >>> dims, fnn = false_nearest_neighbors(data, max_dimension=10)
    >>> optimal_dim = dims[np.where(fnn < 0.01)[0][0]]
    >>> print(f"Optimal dimension: {optimal_dim}")
    """
    data = np.asarray(data)

    dimensions = np.arange(1, max_dimension + 1)
    fnn_percent = np.zeros(len(dimensions))

    # Standard deviation of data
    std_data = np.std(data)

    for i, dim in enumerate(dimensions):
        # Embed in d dimensions
        embedded_d = time_delay_embedding(data, delay, int(dim))

        if dim == max_dimension:
            fnn_percent[i] = 0
            continue

        # Embed in d+1 dimensions
        embedded_d1 = time_delay_embedding(data, delay, int(dim) + 1)

        n_points = len(embedded_d)
        false_neighbors = 0

        # For each point, find nearest neighbor
        for j in range(n_points):
            # Find nearest neighbor in d dimensions
            distances = np.sqrt(np.sum((embedded_d - embedded_d[j]) ** 2, axis=1))
            distances[j] = np.inf  # Exclude self
            nearest_idx = np.argmin(distances)
            nearest_dist_d = distances[nearest_idx]

            if nearest_dist_d == 0:
                continue

            # Check in d+1 dimensions
            dist_d1 = np.linalg.norm(embedded_d1[j] - embedded_d1[nearest_idx])

            # Relative increase in distance
            ratio = np.abs(dist_d1 - nearest_dist_d) / nearest_dist_d

            # Check if false neighbor
            if ratio > rtol or dist_d1 / std_data > atol:
                false_neighbors += 1

        fnn_percent[i] = (false_neighbors / n_points) * 100

    return dimensions, fnn_percent


def recurrence_plot(
    data: np.ndarray,
    dimension: int = 1,
    delay: int = 1,
    threshold: float | None = None,
    method: str = "distance",
) -> np.ndarray:
    """
    Calculate recurrence plot matrix.

    Visualizes recurring patterns in time series.

    Parameters
    ----------
    data : array_like
        Input time series
    dimension : int, optional
        Embedding dimension (default=1)
    delay : int, optional
        Time delay (default=1)
    threshold : float, optional
        Recurrence threshold (if None, use 10% of max distance)
    method : str, optional
        Distance method: 'distance' or 'correlation' (default='distance')

    Returns
    -------
    ndarray
        Recurrence matrix (binary)

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> t = np.linspace(0, 50, 500)
    >>> data = np.sin(t) + 0.5*np.sin(3*t)
    >>> R = recurrence_plot(data, dimension=3, delay=10)
    >>> plt.figure(figsize=(8, 8))
    >>> plt.imshow(R, cmap='binary', origin='lower')
    >>> plt.xlabel('Time')
    >>> plt.ylabel('Time')
    >>> plt.title('Recurrence Plot')
    >>> plt.show()
    """
    data = np.asarray(data)

    # Embed if needed
    if dimension > 1:
        embedded = time_delay_embedding(data, delay, dimension)
    else:
        embedded = data.reshape(-1, 1)

    # Calculate distance matrix using dispatch
    DISTANCE_METRICS = {
        "distance": lambda: squareform(pdist(embedded, metric="euclidean")),
        "correlation": lambda: squareform(pdist(embedded, metric="correlation")),
    }

    if method not in DISTANCE_METRICS:
        valid_methods = ", ".join(f"'{m}'" for m in DISTANCE_METRICS.keys())
        raise ValueError(f"Unknown method '{method}'. Valid options: {valid_methods}")

    distances = DISTANCE_METRICS[method]()

    # Threshold
    if threshold is None:
        threshold = 0.1 * np.max(distances)

    # Recurrence matrix
    R = (distances <= threshold).astype(int)

    return R


def recurrence_quantification_analysis(R: np.ndarray) -> dict:
    """
    Recurrence Quantification Analysis (RQA) measures.

    Parameters
    ----------
    R : array_like
        Recurrence matrix from recurrence_plot()

    Returns
    -------
    dict
        RQA measures

    Examples
    --------
    >>> data = np.sin(np.linspace(0, 50, 500))
    >>> R = recurrence_plot(data, dimension=3, delay=10)
    >>> rqa = recurrence_quantification_analysis(R)
    >>> print(f"Recurrence rate: {rqa['recurrence_rate']:.3f}")
    >>> print(f"Determinism: {rqa['determinism']:.3f}")
    """
    R = np.asarray(R)
    N = R.shape[0]

    # Remove main diagonal
    R_offdiag = R.copy()
    np.fill_diagonal(R_offdiag, 0)

    # Recurrence Rate (RR)
    recurrence_rate = np.sum(R_offdiag) / (N * (N - 1))

    # Find diagonal lines (determinism)
    min_line_length = 2
    diagonal_lengths = []

    for offset in range(-N + 1, N):
        diag = np.diagonal(R, offset=offset)
        if len(diag) < min_line_length:
            continue

        # Find consecutive sequences
        consecutive = []
        count = 0
        for val in diag:
            if val == 1:
                count += 1
            else:
                if count >= min_line_length:
                    consecutive.append(count)
                count = 0
        if count >= min_line_length:
            consecutive.append(count)

        diagonal_lengths.extend(consecutive)

    determinism: float
    avg_line_length: float
    max_line_length: float
    entropy: float

    if len(diagonal_lengths) > 0:
        # Determinism (DET)
        determinism = float(np.sum(diagonal_lengths) / np.sum(R_offdiag))

        # Average diagonal line length (L)
        avg_line_length = float(np.mean(diagonal_lengths))

        # Maximum diagonal line length
        max_line_length = float(np.max(diagonal_lengths))

        # Entropy of diagonal line lengths
        hist, _ = np.histogram(
            diagonal_lengths, bins=np.arange(min(diagonal_lengths), max(diagonal_lengths) + 2)
        )
        prob = hist / np.sum(hist)
        prob = prob[prob > 0]
        entropy = float(-np.sum(prob * np.log(prob)))
    else:
        determinism = 0.0
        avg_line_length = 0.0
        max_line_length = 0.0
        entropy = 0.0

    return {
        "recurrence_rate": recurrence_rate,
        "determinism": determinism,
        "avg_diagonal_line": avg_line_length,
        "max_diagonal_line": max_line_length,
        "entropy": entropy,
    }


def lyapunov_exponent(
    data: np.ndarray, dimension: int = 3, delay: int = 1, n_iterations: int = 500
) -> float:
    """
    Estimate largest Lyapunov exponent (simplified method).

    Positive Lyapunov exponent indicates chaos.

    Parameters
    ----------
    data : array_like
        Input time series
    dimension : int, optional
        Embedding dimension (default=3)
    delay : int, optional
        Time delay (default=1)
    n_iterations : int, optional
        Number of iterations (default=500)

    Returns
    -------
    float
        Largest Lyapunov exponent estimate

    Examples
    --------
    >>> # Chaotic system should have positive Lyapunov exponent
    >>> t = np.linspace(0, 100, 10000)
    >>> data = np.sin(t) * (1 + 0.1*np.sin(11*t))
    >>> lyap = lyapunov_exponent(data, dimension=3, delay=10)
    >>> print(f"Lyapunov exponent: {lyap:.4f}")
    >>> if lyap > 0:
    ...     print("System is chaotic")
    """
    data = np.asarray(data)

    # Embed data
    embedded = time_delay_embedding(data, delay, dimension)
    n_points = len(embedded)

    if n_points < n_iterations + 10:
        n_iterations = n_points - 10

    lyap_sum = 0
    n_valid = 0

    # Track divergence of nearby trajectories
    for i in range(n_iterations):
        # Find nearest neighbor
        distances = np.sqrt(np.sum((embedded - embedded[i]) ** 2, axis=1))
        distances[i] = np.inf
        nearest_idx = np.argmin(distances)

        initial_dist = distances[nearest_idx]

        if initial_dist == 0:
            continue

        # Track evolution for a few steps
        steps = int(min(10, n_points - max(int(i), int(nearest_idx)) - 1))

        for step in range(1, steps):
            if i + step >= n_points or nearest_idx + step >= n_points:
                break

            evolved_dist = np.linalg.norm(embedded[i + step] - embedded[nearest_idx + step])

            if evolved_dist > 0:
                lyap_sum += np.log(evolved_dist / initial_dist)
                n_valid += 1
                break

    if n_valid > 0:
        return lyap_sum / n_valid
    else:
        return 0.0


def correlation_dimension(
    data: np.ndarray, dimension: int = 3, delay: int = 1, r_values: np.ndarray | None = None
) -> dict:
    """
    Estimate correlation dimension (fractal dimension).

    Parameters
    ----------
    data : array_like
        Input time series
    dimension : int, optional
        Embedding dimension (default=3)
    delay : int, optional
        Time delay (default=1)
    r_values : array_like, optional
        Radius values to test

    Returns
    -------
    dict
        Correlation dimension estimate

    Examples
    --------
    >>> data = np.sin(np.linspace(0, 100, 1000))
    >>> result = correlation_dimension(data, dimension=3)
    >>> print(f"Correlation dimension: {result['correlation_dim']:.2f}")
    """
    data = np.asarray(data)

    # Embed data
    embedded = time_delay_embedding(data, delay, dimension)
    n_points = len(embedded)

    # Calculate distances
    distances = pdist(embedded, metric="euclidean")

    # Range of radii
    if r_values is None:
        r_min = np.percentile(distances, 1)
        r_max = np.percentile(distances, 50)
        r_values = np.logspace(np.log10(r_min), np.log10(r_max), 20)

    # Correlation integral
    C_r = np.zeros(len(r_values))
    for i, r in enumerate(r_values):
        C_r[i] = np.sum(distances < r) / len(distances)

    # Estimate dimension from slope
    # D = d(log C(r)) / d(log r)
    log_r = np.log(r_values[C_r > 0])
    log_C = np.log(C_r[C_r > 0])

    if len(log_r) > 2:
        # Linear fit
        slope = np.polyfit(log_r, log_C, 1)[0]
    else:
        slope = 0

    return {
        "correlation_dim": slope,
        "r_values": r_values,
        "correlation_integral": C_r,
    }
