"""
Input validation utilities for earthsciences package.

Minimal, focused validation that fails fast on invalid input.
"""

import numpy as np


def validate_array(arr, name="array", ndim=None, min_length=None):
    """
    Validate and convert to numpy array.

    Parameters
    ----------
    arr : array_like
        Array to validate
    name : str
        Parameter name for error messages
    ndim : int, optional
        Required number of dimensions
    min_length : int, optional
        Minimum length for 1D arrays

    Returns
    -------
    np.ndarray
        Validated array

    Raises
    ------
    ValueError, TypeError
        If validation fails
    """
    try:
        arr = np.asarray(arr)
    except Exception as e:
        raise TypeError(f"{name} must be array-like: {e}")

    if arr.size == 0:
        raise ValueError(f"{name} cannot be empty")

    if ndim is not None and arr.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}D, got {arr.ndim}D")

    if min_length is not None and len(arr) < min_length:
        raise ValueError(f"{name} must have at least {min_length} elements, got {len(arr)}")

    if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
        raise ValueError(f"{name} contains NaN or inf values")

    return arr


def validate_same_length(*arrays, names=None):
    """
    Validate that arrays have the same length.

    Parameters
    ----------
    *arrays : array_like
        Arrays to validate
    names : tuple of str, optional
        Names for error messages

    Raises
    ------
    ValueError
        If arrays have different lengths
    """
    if names is None:
        names = tuple(f"array{i}" for i in range(len(arrays)))

    if len(arrays) < 2:
        return

    lengths = [len(arr) for arr in arrays]
    if len(set(lengths)) > 1:
        msg = ", ".join(f"{name}={length}" for name, length in zip(names, lengths))
        raise ValueError(f"Arrays must have same length: {msg}")


def validate_coordinates(x, y, values=None):
    """
    Validate coordinate arrays for spatial operations.

    Parameters
    ----------
    x, y : array_like
        Coordinate arrays
    values : array_like, optional
        Values at coordinates

    Returns
    -------
    tuple
        Validated (x, y) or (x, y, values)
    """
    x = validate_array(x, "x", ndim=1)
    y = validate_array(y, "y", ndim=1)
    validate_same_length(x, y, names=("x", "y"))

    if values is not None:
        values = validate_array(values, "values", ndim=1)
        validate_same_length(x, y, values, names=("x", "y", "values"))
        return x, y, values

    return x, y


def validate_angles(angles, name="angles"):
    """
    Validate angular data.

    Parameters
    ----------
    angles : array_like
        Angular measurements
    name : str
        Parameter name for error messages

    Returns
    -------
    np.ndarray
        Validated angles
    """
    return validate_array(angles, name)
