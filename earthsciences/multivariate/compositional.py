"""
Compositional data analysis

Functions for analyzing compositional (closed) data using Aitchison's
log-ratio transformations. Essential for geochemistry and other earth science
applications where data sum to a constant (e.g., 100% for percentages).
"""

import warnings

import numpy as np


def alr(data: np.ndarray, denominator: int | None = None) -> np.ndarray:
    """
    Additive Log-Ratio (ALR) transformation.

    Transforms compositional data to real space by taking log-ratios
    with respect to a reference component (denominator).

    Parameters
    ----------
    data : array_like
        Compositional data, shape (n_samples, n_components)
        Each row should sum to 1 (or constant)
    denominator : int, optional
        Index of reference component (default=last component)

    Returns
    -------
    ndarray
        ALR-transformed data, shape (n_samples, n_components-1)

    Notes
    -----
    The ALR transformation is:
    alr(x) = log(x_i / x_D) for i = 1, ..., D-1

    where x_D is the denominator component.

    Examples
    --------
    >>> # Geochemical composition (SiO2, Al2O3, FeO)
    >>> comp = np.array([[0.60, 0.30, 0.10],
    ...                  [0.55, 0.35, 0.10],
    ...                  [0.50, 0.40, 0.10]])
    >>> alr_data = alr(comp)
    >>> print(alr_data.shape)  # (3, 2) - one less dimension
    """
    data = np.asarray(data, dtype=float)

    if data.ndim == 1:
        data = data.reshape(1, -1)

    # Check for valid compositional data
    if np.any(data <= 0):
        raise ValueError("All components must be positive for log-ratio transformation")

    # Normalize if not already summing to 1
    row_sums = np.sum(data, axis=1, keepdims=True)
    if not np.allclose(row_sums, 1.0):
        warnings.warn("Data normalized to sum to 1")
        data = data / row_sums

    if denominator is None:
        denominator = data.shape[1] - 1

    # Extract denominator component
    denom_col = data[:, denominator]

    # Compute ALR for all components except denominator
    alr_data = np.log(data / denom_col[:, np.newaxis])
    alr_data = np.delete(alr_data, denominator, axis=1)

    return alr_data


def alr_inv(alr_data: np.ndarray, denominator: int | None = None) -> np.ndarray:
    """
    Inverse ALR transformation.

    Transforms ALR-transformed data back to compositional space.

    Parameters
    ----------
    alr_data : array_like
        ALR-transformed data, shape (n_samples, n_components-1)
    denominator : int, optional
        Index where denominator was originally (default=last)

    Returns
    -------
    ndarray
        Compositional data, shape (n_samples, n_components)

    Examples
    --------
    >>> comp = np.array([[0.60, 0.30, 0.10]])
    >>> alr_data = alr(comp)
    >>> comp_back = alr_inv(alr_data)
    >>> np.allclose(comp, comp_back)
    True
    """
    alr_data = np.asarray(alr_data, dtype=float)

    if alr_data.ndim == 1:
        alr_data = alr_data.reshape(1, -1)

    if denominator is None:
        denominator = alr_data.shape[1]

    # Add back the denominator (which is 0 in ALR space, since log(1) = 0)
    exp_data = np.exp(alr_data)

    # Insert denominator column (value = 1)
    exp_data = np.insert(exp_data, denominator, 1.0, axis=1)

    # Close to sum to 1
    data = exp_data / np.sum(exp_data, axis=1, keepdims=True)

    return data


def clr(data: np.ndarray) -> np.ndarray:
    """
    Centered Log-Ratio (CLR) transformation.

    Transforms compositional data by taking log-ratios with respect
    to the geometric mean. More symmetric than ALR but produces
    singular covariance matrices.

    Parameters
    ----------
    data : array_like
        Compositional data, shape (n_samples, n_components)
        Each row should sum to 1 (or constant)

    Returns
    -------
    ndarray
        CLR-transformed data, same shape as input

    Notes
    -----
    The CLR transformation is:
    clr(x) = log(x_i / g(x))

    where g(x) is the geometric mean of all components.

    Examples
    --------
    >>> comp = np.array([[0.60, 0.30, 0.10]])
    >>> clr_data = clr(comp)
    >>> print(np.sum(clr_data))  # Sum is zero (centered)
    """
    data = np.asarray(data, dtype=float)

    if data.ndim == 1:
        data = data.reshape(1, -1)

    # Check for valid compositional data
    if np.any(data <= 0):
        raise ValueError("All components must be positive for log-ratio transformation")

    # Normalize if not already summing to 1
    row_sums = np.sum(data, axis=1, keepdims=True)
    if not np.allclose(row_sums, 1.0):
        warnings.warn("Data normalized to sum to 1")
        data = data / row_sums

    # Geometric mean for each sample
    geom_mean = np.exp(np.mean(np.log(data), axis=1, keepdims=True))

    # CLR transformation
    clr_data = np.log(data / geom_mean)

    return clr_data


def clr_inv(clr_data: np.ndarray) -> np.ndarray:
    """
    Inverse CLR transformation.

    Transforms CLR-transformed data back to compositional space.

    Parameters
    ----------
    clr_data : array_like
        CLR-transformed data

    Returns
    -------
    ndarray
        Compositional data

    Examples
    --------
    >>> comp = np.array([[0.60, 0.30, 0.10]])
    >>> clr_data = clr(comp)
    >>> comp_back = clr_inv(clr_data)
    >>> np.allclose(comp, comp_back)
    True
    """
    clr_data = np.asarray(clr_data, dtype=float)

    if clr_data.ndim == 1:
        clr_data = clr_data.reshape(1, -1)

    # Exponentiate
    exp_data = np.exp(clr_data)

    # Close to sum to 1
    data = exp_data / np.sum(exp_data, axis=1, keepdims=True)

    return data


def ilr(data: np.ndarray, basis: np.ndarray | None = None) -> np.ndarray:
    """
    Isometric Log-Ratio (ILR) transformation.

    Transforms compositional data to real space using an orthonormal basis.
    Preserves distances and produces non-singular covariance matrices.
    This is the most statistically appropriate transformation.

    Parameters
    ----------
    data : array_like
        Compositional data, shape (n_samples, n_components)
    basis : array_like, optional
        Orthonormal basis for transformation, shape (n_components, n_components-1)
        If None, uses default sequential binary partition

    Returns
    -------
    ndarray
        ILR-transformed data, shape (n_samples, n_components-1)

    Notes
    -----
    The ILR transformation requires an orthonormal basis in the simplex.
    By default, uses Gram-Schmidt orthonormalization of CLR-transformed data.

    Examples
    --------
    >>> comp = np.array([[0.60, 0.30, 0.10],
    ...                  [0.55, 0.35, 0.10]])
    >>> ilr_data = ilr(comp)
    >>> print(ilr_data.shape)  # (2, 2) - one less dimension
    """
    data = np.asarray(data, dtype=float)

    if data.ndim == 1:
        data = data.reshape(1, -1)

    # Check for valid compositional data
    if np.any(data <= 0):
        raise ValueError("All components must be positive for log-ratio transformation")

    # Normalize
    row_sums = np.sum(data, axis=1, keepdims=True)
    if not np.allclose(row_sums, 1.0):
        warnings.warn("Data normalized to sum to 1")
        data = data / row_sums

    n_components = data.shape[1]

    if basis is None:
        # Create default basis using sequential binary partition
        # This is the Helmert sub-matrix approach
        basis = _default_ilr_basis(n_components)

    # Transform via CLR and then project onto orthonormal basis
    clr_data = clr(data)
    ilr_data = clr_data @ basis

    return ilr_data


def ilr_inv(ilr_data: np.ndarray, basis: np.ndarray | None = None) -> np.ndarray:
    """
    Inverse ILR transformation.

    Transforms ILR-transformed data back to compositional space.

    Parameters
    ----------
    ilr_data : array_like
        ILR-transformed data, shape (n_samples, n_components-1)
    basis : array_like, optional
        Same orthonormal basis used in forward transformation

    Returns
    -------
    ndarray
        Compositional data

    Examples
    --------
    >>> comp = np.array([[0.60, 0.30, 0.10]])
    >>> ilr_data = ilr(comp)
    >>> comp_back = ilr_inv(ilr_data)
    >>> np.allclose(comp, comp_back)
    True
    """
    ilr_data = np.asarray(ilr_data, dtype=float)

    if ilr_data.ndim == 1:
        ilr_data = ilr_data.reshape(1, -1)

    n_components = ilr_data.shape[1] + 1

    if basis is None:
        basis = _default_ilr_basis(n_components)

    # Project back to CLR space
    clr_data = ilr_data @ basis.T

    # CLR inverse
    data = clr_inv(clr_data)

    return data


def _default_ilr_basis(n_components: int) -> np.ndarray:
    """
    Create default orthonormal basis for ILR transformation.

    Uses Gram-Schmidt orthonormalization approach.

    Parameters
    ----------
    n_components : int
        Number of compositional components

    Returns
    -------
    ndarray
        Orthonormal basis, shape (n_components, n_components-1)
    """
    # Create Helmert sub-matrix (sequential binary partition)
    basis = np.zeros((n_components, n_components - 1))

    for i in range(n_components - 1):
        # Numerator: first i+1 components
        basis[: i + 1, i] = 1.0 / (i + 1)
        # Denominator: (i+1)-th component
        basis[i + 1, i] = -1.0
        # Normalize
        basis[:, i] *= np.sqrt((i + 1) / (i + 2))

    return basis


def aitchison_distance(x: np.ndarray, y: np.ndarray) -> float:
    """
    Aitchison distance between two compositions.

    The Aitchison distance is the Euclidean distance between
    CLR-transformed compositions.

    Parameters
    ----------
    x, y : array_like
        Compositional data (1D arrays)

    Returns
    -------
    float
        Aitchison distance

    Examples
    --------
    >>> x = np.array([0.6, 0.3, 0.1])
    >>> y = np.array([0.5, 0.4, 0.1])
    >>> dist = aitchison_distance(x, y)
    >>> print(f"Distance: {dist:.3f}")
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Transform to CLR
    clr_x = clr(x.reshape(1, -1)).flatten()
    clr_y = clr(y.reshape(1, -1)).flatten()

    # Euclidean distance in CLR space
    distance = np.linalg.norm(clr_x - clr_y)

    return distance


def closure(data: np.ndarray, total: float = 1.0) -> np.ndarray:
    """
    Closure operation (normalization to constant sum).

    Forces compositional data to sum to a specified constant.

    Parameters
    ----------
    data : array_like
        Data to close
    total : float, optional
        Target sum (default=1.0)

    Returns
    -------
    ndarray
        Closed data

    Examples
    --------
    >>> data = np.array([[60, 30, 10], [55, 35, 10]])
    >>> closed = closure(data, total=1.0)
    >>> print(closed)
    [[0.6  0.3  0.1 ]
     [0.55 0.35 0.1 ]]
    """
    data = np.asarray(data, dtype=float)

    if data.ndim == 1:
        return total * data / np.sum(data)
    else:
        return total * data / np.sum(data, axis=1, keepdims=True)


def perturbation(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Perturbation operation (compositional equivalent of addition).

    Aitchison's perturbation operator for compositional data.

    Parameters
    ----------
    x, y : array_like
        Compositional data

    Returns
    -------
    ndarray
        Perturbed composition (closed to sum to 1)

    Examples
    --------
    >>> x = np.array([0.6, 0.3, 0.1])
    >>> y = np.array([0.5, 0.4, 0.1])
    >>> z = perturbation(x, y)
    >>> print(z)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Element-wise multiplication and closure
    result = x * y
    return closure(result)


def powering(x: np.ndarray, alpha: float) -> np.ndarray:
    """
    Powering operation (compositional equivalent of scalar multiplication).

    Aitchison's powering operator for compositional data.

    Parameters
    ----------
    x : array_like
        Compositional data
    alpha : float
        Power coefficient

    Returns
    -------
    ndarray
        Powered composition (closed to sum to 1)

    Examples
    --------
    >>> x = np.array([0.6, 0.3, 0.1])
    >>> z = powering(x, 2.0)
    >>> print(z)
    """
    x = np.asarray(x, dtype=float)

    # Raise to power and closure
    result = x**alpha
    return closure(result)


def compositional_mean(data: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Aitchison mean of compositional data.

    Geometric mean followed by closure.

    Parameters
    ----------
    data : array_like
        Compositional data
    axis : int, optional
        Axis along which to compute mean (default=0)

    Returns
    -------
    ndarray
        Mean composition

    Examples
    --------
    >>> data = np.array([[0.60, 0.30, 0.10],
    ...                  [0.55, 0.35, 0.10]])
    >>> mean_comp = compositional_mean(data)
    >>> print(mean_comp)
    """
    data = np.asarray(data, dtype=float)

    # Geometric mean
    log_mean = np.mean(np.log(data), axis=axis)
    geom_mean = np.exp(log_mean)

    # Closure
    return closure(geom_mean)


def variation_matrix(data: np.ndarray) -> np.ndarray:
    """
    Variation matrix for compositional data.

    The variation matrix contains variances of all log-ratios.

    Parameters
    ----------
    data : array_like
        Compositional data, shape (n_samples, n_components)

    Returns
    -------
    ndarray
        Variation matrix, shape (n_components, n_components)

    Notes
    -----
    Element (i,j) contains var(log(x_i / x_j))
    Diagonal elements are zero by definition.

    Examples
    --------
    >>> data = np.array([[0.60, 0.30, 0.10],
    ...                  [0.55, 0.35, 0.10]])
    >>> var_matrix = variation_matrix(data)
    >>> print(var_matrix.shape)
    (3, 3)
    """
    data = np.asarray(data, dtype=float)

    if data.ndim == 1:
        data = data.reshape(1, -1)

    n_components = data.shape[1]
    log_data = np.log(data)

    # Compute all pairwise log-ratio variances
    var_mat = np.zeros((n_components, n_components))

    for i in range(n_components):
        for j in range(n_components):
            if i != j:
                log_ratio = log_data[:, i] - log_data[:, j]
                var_mat[i, j] = np.var(log_ratio, ddof=1)

    return var_mat
