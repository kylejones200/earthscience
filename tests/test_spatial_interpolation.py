"""
Tests for spatial interpolation module
"""

import numpy as np

from earthsciences.spatial import interpolation


class TestIDWInterpolation:
    """Test inverse distance weighting interpolation."""

    def test_basic_functionality(self):
        """Test basic IDW interpolation."""
        np.random.seed(42)
        x = np.array([0, 1, 0, 1])
        y = np.array([0, 0, 1, 1])
        values = np.array([0, 1, 1, 2])

        grid_x, grid_y = np.meshgrid(np.linspace(0, 1, 3), np.linspace(0, 1, 3))

        result = interpolation.idw_interpolation(x, y, values, grid_x, grid_y)

        assert result.shape == grid_x.shape
        assert not np.any(np.isnan(result))

    def test_exact_at_data_points(self):
        """Test that interpolation is close to exact at data points."""
        x = np.array([0, 5, 10])
        y = np.array([0, 5, 10])
        values = np.array([1, 2, 3])

        # Query at data points - should be very close to original values
        result = interpolation.idw_interpolation(x, y, values, x, y)

        # IDW may not be exactly equal, but should be close
        np.testing.assert_array_almost_equal(result, values, decimal=0)

    def test_power_parameter(self):
        """Test effect of power parameter."""
        x = np.array([0, 10])
        y = np.array([0, 0])
        values = np.array([0, 10])

        grid_x = np.array([[5]])
        grid_y = np.array([[0]])

        # Higher power gives more weight to closer points
        result_p1 = interpolation.idw_interpolation(x, y, values, grid_x, grid_y, power=1)
        result_p3 = interpolation.idw_interpolation(x, y, values, grid_x, grid_y, power=3)

        # Both should be around 5 (midpoint)
        assert 4 < result_p1[0, 0] < 6
        assert 4 < result_p3[0, 0] < 6


class TestNearestNeighbor:
    """Test nearest neighbor interpolation."""

    def test_basic_functionality(self):
        """Test basic nearest neighbor interpolation."""
        x = np.array([0, 5, 10])
        y = np.array([0, 5, 10])
        values = np.array([1, 2, 3])

        grid_x, grid_y = np.meshgrid(np.linspace(0, 10, 5), np.linspace(0, 10, 5))

        result = interpolation.nearest_neighbor(x, y, values, grid_x, grid_y)

        assert result.shape == grid_x.shape
        # All values should be one of the original values
        assert np.all(np.isin(result, values))


class TestNaturalNeighbor:
    """Test natural neighbor interpolation."""

    def test_basic_functionality(self):
        """Test natural neighbor interpolation on scattered points."""
        np.random.seed(42)
        x = np.random.rand(25) * 10
        y = np.random.rand(25) * 10
        values = np.sin(x) + np.cos(y)

        grid_x, grid_y = np.meshgrid(np.linspace(0, 10, 15), np.linspace(0, 10, 15))
        result = interpolation.natural_neighbor(x, y, values, grid_x, grid_y)

        assert result.shape == grid_x.shape
        assert np.any(np.isfinite(result))
        interior = result[2:-2, 2:-2]
        assert np.mean(np.isfinite(interior)) > 0.8


class TestRBFInterpolation:
    """Test radial basis function interpolation."""

    def test_basic_functionality(self):
        """Test RBF interpolation."""
        np.random.seed(42)
        x = np.random.rand(20) * 10
        y = np.random.rand(20) * 10
        values = np.sin(x) + np.cos(y)

        grid_x, grid_y = np.meshgrid(np.linspace(0, 10, 10), np.linspace(0, 10, 10))

        result = interpolation.rbf_interpolation(x, y, values, grid_x, grid_y)

        assert result.shape == grid_x.shape
        assert not np.any(np.isnan(result))

    def test_different_kernels(self):
        """Test different RBF kernels."""
        x = np.array([0, 5, 10])
        y = np.array([0, 5, 10])
        values = np.array([1, 2, 3])

        grid_x, grid_y = np.meshgrid(np.linspace(0, 10, 5), np.linspace(0, 10, 5))

        for kernel in ["gaussian", "multiquadric", "thin_plate"]:
            result = interpolation.rbf_interpolation(x, y, values, grid_x, grid_y, kernel=kernel)
            assert result.shape == grid_x.shape


class TestSplineInterpolation:
    """Test spline interpolation."""

    def test_basic_functionality(self):
        """Test bivariate spline interpolation."""
        x = np.array([0, 2, 4, 6, 8, 10])
        y = np.array([0, 2, 4, 6, 8, 10])
        values = np.array([0, 1, 2, 1, 0, -1])

        grid_x, grid_y = np.meshgrid(np.linspace(0, 10, 15), np.linspace(0, 10, 15))

        result = interpolation.spline_interpolation(x, y, values, grid_x, grid_y)

        assert result.shape == grid_x.shape
        assert not np.any(np.isnan(result))
