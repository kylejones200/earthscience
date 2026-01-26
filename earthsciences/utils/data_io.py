"""
Data input/output utilities

Essential file I/O functions for common earth sciences formats.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Union


def load_csv(filepath, skiprows=0, delimiter=',', names=None):
    """
    Load CSV file.
    
    Parameters
    ----------
    filepath : str
        Path to CSV file
    skiprows : int, optional
        Rows to skip
    delimiter : str, optional
        Column delimiter
    names : list, optional
        Column names
        
    Returns
    -------
    DataFrame
    """
    return pd.read_csv(filepath, skiprows=skiprows, delimiter=delimiter, names=names)


def save_csv(data, filepath, header=True, index=False):
    """
    Save data to CSV.
    
    Parameters
    ----------
    data : DataFrame or ndarray
        Data to save
    filepath : str
        Output file path
    header : bool, optional
        Include header
    index : bool, optional
        Include row index
    """
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data)
    data.to_csv(filepath, header=header, index=index)


def load_xyz(filepath):
    """
    Load XYZ format data.
    
    Parameters
    ----------
    filepath : str
        Path to XYZ file
        
    Returns
    -------
    dict
        Dictionary with x, y, z coordinates
    """
    data = np.loadtxt(filepath)
    return {
        'x': data[:, 0],
        'y': data[:, 1],
        'z': data[:, 2] if data.shape[1] > 2 else None,
    }


def save_xyz(x, y, z, filepath):
    """
    Save data in XYZ format.
    
    Parameters
    ----------
    x, y : array_like
        Coordinates
    z : array_like, optional
        Values
    filepath : str
        Output file path
    """
    if z is not None:
        data = np.column_stack([x, y, z])
    else:
        data = np.column_stack([x, y])
    np.savetxt(filepath, data, fmt='%.6f')


def save_numpy(data, filepath):
    """
    Save data in NumPy format (.npy or .npz).
    
    Parameters
    ----------
    data : ndarray or dict
        Data to save
    filepath : str
        Output file path
    """
    if isinstance(data, dict):
        np.savez(filepath, **data)
    else:
        np.save(filepath, data)


def load_numpy(filepath):
    """
    Load NumPy format data.
    
    Parameters
    ----------
    filepath : str
        Path to .npy or .npz file
        
    Returns
    -------
    ndarray or dict
    """
    if filepath.endswith('.npz'):
        loaded = np.load(filepath)
        return {key: loaded[key] for key in loaded.files}
    else:
        return np.load(filepath)
