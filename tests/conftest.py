"""
Pytest configuration and shared fixtures
"""

import numpy as np
import pytest


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    np.random.seed(42)
    return np.random.randn(100)


@pytest.fixture
def sample_data_2d():
    """Generate 2D sample data for testing."""
    np.random.seed(42)
    return np.random.randn(100, 5)


@pytest.fixture
def sample_xy_data():
    """Generate x, y paired data for testing."""
    np.random.seed(42)
    x = np.linspace(0, 10, 50)
    y = 2 * x + 1 + np.random.randn(50) * 0.5
    return x, y


@pytest.fixture
def sample_spatial_data():
    """Generate spatial point data for testing."""
    np.random.seed(42)
    x = np.random.rand(30) * 10
    y = np.random.rand(30) * 10
    values = np.sin(x) + np.cos(y)
    return x, y, values


@pytest.fixture
def sample_angles():
    """Generate circular data for testing."""
    np.random.seed(42)
    return np.array([10, 350, 5, 15, 355, 0, 8, 12, 358, 3])


@pytest.fixture
def sample_time_series():
    """Generate time series data for testing."""
    np.random.seed(42)
    t = np.linspace(0, 1, 1000)
    signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 10 * t)
    return t, signal
