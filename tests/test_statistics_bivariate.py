"""
Tests for bivariate statistics module
"""

import numpy as np
import pytest

from earthsciences.statistics import bivariate


class TestCorrelation:
    """Test correlation function."""

    def test_perfect_positive_correlation(self):
        """Test perfect positive correlation."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])

        r, p = bivariate.correlation(x, y, method="pearson")

        assert np.isclose(r, 1.0)
        assert p < 0.05

    def test_perfect_negative_correlation(self):
        """Test perfect negative correlation."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([10, 8, 6, 4, 2])

        r, p = bivariate.correlation(x, y, method="pearson")

        assert np.isclose(r, -1.0)
        assert p < 0.05

    def test_no_correlation(self):
        """Test uncorrelated data."""
        np.random.seed(42)
        x = np.random.randn(100)
        y = np.random.randn(100)

        r, p = bivariate.correlation(x, y, method="pearson")

        # Should be close to 0
        assert abs(r) < 0.3

    def test_spearman_correlation(self, sample_xy_data):
        """Test Spearman correlation."""
        x, y = sample_xy_data
        r, p = bivariate.correlation(x, y, method="spearman")

        assert -1 <= r <= 1
        assert 0 <= p <= 1

    def test_kendall_correlation(self, sample_xy_data):
        """Test Kendall correlation."""
        x, y = sample_xy_data
        r, p = bivariate.correlation(x, y, method="kendall")

        assert -1 <= r <= 1
        assert 0 <= p <= 1

    def test_nan_handling(self):
        """Test handling of NaN values."""
        x = np.array([1, 2, np.nan, 4, 5])
        y = np.array([2, 4, 6, 8, 10])

        r, p = bivariate.correlation(x, y)

        # Should work with NaN removed
        assert not np.isnan(r)

    def test_insufficient_data(self):
        """Test that insufficient data raises error."""
        x = np.array([1, 2])
        y = np.array([2, 4])

        with pytest.raises(ValueError, match="at least 3"):
            bivariate.correlation(x, y)

    def test_invalid_method(self, sample_xy_data):
        """Test invalid correlation method."""
        x, y = sample_xy_data
        with pytest.raises(ValueError, match="Unknown method"):
            bivariate.correlation(x, y, method="invalid")


class TestLinearRegression:
    """Test linear_regression function."""

    def test_perfect_fit(self):
        """Test regression with perfect linear relationship."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])  # y = 2x

        result = bivariate.linear_regression(x, y)

        assert np.isclose(result["slope"], 2.0)
        assert np.isclose(result["intercept"], 0.0)
        assert np.isclose(result["r_squared"], 1.0)

    def test_with_intercept(self):
        """Test regression with non-zero intercept."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([3, 5, 7, 9, 11])  # y = 2x + 1

        result = bivariate.linear_regression(x, y)

        assert np.isclose(result["slope"], 2.0)
        assert np.isclose(result["intercept"], 1.0)

    def test_returns_all_keys(self, sample_xy_data):
        """Test that all expected keys are returned."""
        x, y = sample_xy_data
        result = bivariate.linear_regression(x, y)

        expected_keys = [
            "slope",
            "intercept",
            "r_value",
            "r_squared",
            "p_value",
            "std_err",
            "residuals",
            "predicted",
        ]

        for key in expected_keys:
            assert key in result

    def test_residuals_sum_to_zero(self, sample_xy_data):
        """Test that residuals sum to approximately zero."""
        x, y = sample_xy_data
        result = bivariate.linear_regression(x, y)

        assert np.abs(np.sum(result["residuals"])) < 1e-10

    def test_insufficient_data(self):
        """Test that insufficient data raises error."""
        x = np.array([1])
        y = np.array([2])

        with pytest.raises(ValueError, match="at least 2"):
            bivariate.linear_regression(x, y)


class TestRMARegression:
    """Test rma_regression function."""

    def test_basic_functionality(self, sample_xy_data):
        """Test RMA regression basic functionality."""
        x, y = sample_xy_data
        result = bivariate.rma_regression(x, y)

        assert "slope" in result
        assert "intercept" in result
        assert "r_value" in result

    def test_differs_from_ols(self):
        """Test that RMA differs from OLS."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2.1, 3.9, 6.2, 7.8, 10.1])

        ols_result = bivariate.linear_regression(x, y)
        rma_result = bivariate.rma_regression(x, y)

        # RMA and OLS should give different slopes
        assert not np.isclose(ols_result["slope"], rma_result["slope"])


class TestPolynomialFit:
    """Test polynomial_fit function."""

    def test_linear_fit(self):
        """Test that degree 1 matches linear regression."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])

        result = bivariate.polynomial_fit(x, y, degree=1)

        assert len(result["coefficients"]) == 2
        assert np.isclose(result["r_squared"], 1.0)

    def test_quadratic_fit(self):
        """Test quadratic fit."""
        x = np.linspace(0, 10, 20)
        y = 2 * x**2 - 3 * x + 1

        result = bivariate.polynomial_fit(x, y, degree=2)

        # Should fit perfectly
        assert np.isclose(result["r_squared"], 1.0)
        # Check coefficients (highest degree first)
        assert np.isclose(result["coefficients"][0], 2.0)

    def test_insufficient_data(self):
        """Test that insufficient data raises error."""
        x = np.array([1, 2, 3])
        y = np.array([2, 4, 6])

        with pytest.raises(ValueError):
            bivariate.polynomial_fit(x, y, degree=5)


class TestMovingAverage:
    """Test moving_average function."""

    def test_basic_functionality(self):
        """Test moving average basic functionality."""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        smoothed = bivariate.moving_average(data, window=3)

        assert len(smoothed) == len(data)

    def test_smoothing_effect(self):
        """Test that moving average smooths data."""
        data = np.array([1, 10, 1, 10, 1, 10, 1, 10])
        smoothed = bivariate.moving_average(data, window=3)

        # Smoothed should have lower variance
        assert np.var(smoothed) < np.var(data)

    def test_invalid_window(self):
        """Test that invalid window raises error."""
        data = np.array([1, 2, 3, 4, 5])

        with pytest.raises(ValueError):
            bivariate.moving_average(data, window=0)


class TestDetrend:
    """Test detrend function."""

    def test_linear_detrend(self):
        """Test linear detrending."""
        x = np.linspace(0, 10, 100)
        trend = 2 * x + 5
        noise = np.random.randn(100) * 0.1
        data = trend + noise

        detrended = bivariate.detrend(data, method="linear")

        # Mean should be close to 0 after detrending
        assert np.abs(np.mean(detrended)) < 0.5

    def test_constant_detrend(self):
        """Test constant detrending (mean removal)."""
        data = np.random.randn(100) + 10

        detrended = bivariate.detrend(data, method="constant")

        # Mean should be approximately 0
        assert np.abs(np.mean(detrended)) < 1e-10


class TestConfidenceInterval:
    """Test confidence_interval function."""

    def test_95_confidence(self):
        """Test 95% confidence interval."""
        np.random.seed(42)
        data = np.random.randn(100) + 5

        lower, upper = bivariate.confidence_interval(data, confidence=0.95)

        mean = np.mean(data)
        assert lower < mean < upper

    def test_wider_confidence(self):
        """Test that 99% CI is wider than 95% CI."""
        np.random.seed(42)
        data = np.random.randn(100)

        lower_95, upper_95 = bivariate.confidence_interval(data, confidence=0.95)
        lower_99, upper_99 = bivariate.confidence_interval(data, confidence=0.99)

        width_95 = upper_95 - lower_95
        width_99 = upper_99 - lower_99

        assert width_99 > width_95
