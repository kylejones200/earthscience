"""
Tests for input validation utilities.
"""

import numpy as np
import pytest

from earthsciences.utils.validation import (
    validate_angles,
    validate_array,
    validate_coordinates,
    validate_same_length,
)


class TestValidateArray:
    """Test validate_array function."""

    def test_basic_validation(self):
        """Test basic array validation."""
        arr = validate_array([1, 2, 3], "test")
        assert isinstance(arr, np.ndarray)
        assert len(arr) == 3

    def test_empty_array_raises(self):
        """Test that empty arrays raise ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_array([], "test")

    def test_nan_detection(self):
        """Test NaN detection."""
        with pytest.raises(ValueError, match="NaN or inf"):
            validate_array([1, 2, np.nan], "test")

    def test_inf_detection(self):
        """Test infinity detection."""
        with pytest.raises(ValueError, match="NaN or inf"):
            validate_array([1, 2, np.inf], "test")

    def test_ndim_validation(self):
        """Test dimensionality validation."""
        with pytest.raises(ValueError, match="must be 1D"):
            validate_array([[1, 2], [3, 4]], "test", ndim=1)

        # Should work with correct ndim
        arr = validate_array([[1, 2], [3, 4]], "test", ndim=2)
        assert arr.ndim == 2

    def test_min_length_validation(self):
        """Test minimum length validation."""
        with pytest.raises(ValueError, match="at least 5 elements"):
            validate_array([1, 2, 3], "test", min_length=5)

        # Should work with sufficient length
        arr = validate_array([1, 2, 3, 4, 5], "test", min_length=5)
        assert len(arr) == 5


class TestValidateSameLength:
    """Test validate_same_length function."""

    def test_same_length_passes(self):
        """Test that arrays with same length pass."""
        validate_same_length([1, 2, 3], [4, 5, 6], [7, 8, 9])

    def test_different_length_raises(self):
        """Test that different lengths raise ValueError."""
        with pytest.raises(ValueError, match="must have same length"):
            validate_same_length([1, 2], [3, 4, 5])

    def test_custom_names(self):
        """Test custom array names in error messages."""
        with pytest.raises(ValueError, match="x="):
            validate_same_length([1, 2], [3, 4, 5], names=("x", "y"))


class TestValidateCoordinates:
    """Test validate_coordinates function."""

    def test_valid_coordinates(self):
        """Test valid coordinate arrays."""
        x, y = validate_coordinates([1, 2, 3], [4, 5, 6])
        assert len(x) == len(y) == 3

    def test_with_values(self):
        """Test coordinate validation with values."""
        x, y, values = validate_coordinates([1, 2], [3, 4], [5, 6])
        assert len(x) == len(y) == len(values) == 2

    def test_different_lengths_raises(self):
        """Test that different lengths raise ValueError."""
        with pytest.raises(ValueError, match="must have same length"):
            validate_coordinates([1, 2], [3, 4, 5])

    def test_nan_rejected(self):
        """Test that NaN values are rejected."""
        with pytest.raises(ValueError, match="NaN or inf"):
            validate_coordinates([1, 2, np.nan], [3, 4, 5])

    def test_inf_rejected(self):
        """Test that infinity values are rejected."""
        with pytest.raises(ValueError, match="NaN or inf"):
            validate_coordinates([1, 2, np.inf], [3, 4, 5])


class TestValidateAngles:
    """Test validate_angles function."""

    def test_valid_angles_degrees(self):
        """Test valid angles in degrees."""
        angles = validate_angles([0, 90, 180, 270])
        assert len(angles) == 4

    def test_valid_angles_radians(self):
        """Test valid angles in radians."""
        angles = validate_angles([0, np.pi / 2, np.pi, 3 * np.pi / 2])
        assert len(angles) == 4

    def test_nan_rejected(self):
        """Test that NaN values are rejected."""
        with pytest.raises(ValueError, match="NaN or inf"):
            validate_angles([0, 90, np.nan])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
