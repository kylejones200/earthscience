"""
Tests for univariate statistics module
"""

import numpy as np
import pytest

from earthsciences.statistics import univariate


class TestDescriptiveStats:
    """Test descriptive_stats function."""

    def test_basic_functionality(self, sample_data):
        """Test that descriptive_stats returns all expected keys."""
        result = univariate.descriptive_stats(sample_data)

        expected_keys = [
            "mean",
            "median",
            "mode",
            "std",
            "variance",
            "range",
            "min",
            "max",
            "Q1",
            "Q2",
            "Q3",
            "IQR",
            "skewness",
            "kurtosis",
            "count",
        ]

        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_with_known_values(self):
        """Test with data that has known statistical properties."""
        data = np.array([1, 2, 3, 4, 5])
        result = univariate.descriptive_stats(data)

        assert result["mean"] == 3.0
        assert result["median"] == 3.0
        assert result["min"] == 1.0
        assert result["max"] == 5.0
        assert result["count"] == 5

    def test_with_nan_values(self):
        """Test handling of NaN values."""
        data = np.array([1, 2, np.nan, 4, 5])
        result = univariate.descriptive_stats(data, remove_nan=True)

        assert result["count"] == 4
        assert result["mean"] == 3.0

    def test_empty_array_raises_error(self):
        """Test that empty array raises ValueError."""
        with pytest.raises(ValueError, match="Empty array"):
            univariate.descriptive_stats(np.array([]))

    def test_all_nan_raises_error(self):
        """Test that all NaN values raises error."""
        data = np.array([np.nan, np.nan, np.nan])
        with pytest.raises(ValueError):
            univariate.descriptive_stats(data, remove_nan=True)


class TestPercentiles:
    """Test percentiles function."""

    def test_default_quartiles(self, sample_data):
        """Test default quartile calculation."""
        result = univariate.percentiles(sample_data)
        assert len(result) == 3  # Q1, Q2, Q3

    def test_single_percentile(self, sample_data):
        """Test single percentile."""
        result = univariate.percentiles(sample_data, q=50)
        assert isinstance(result, (float, np.floating))

    def test_custom_percentiles(self, sample_data):
        """Test custom percentiles."""
        result = univariate.percentiles(sample_data, q=[10, 90])
        assert len(result) == 2


class TestZScore:
    """Test z_score function."""

    def test_normalization(self, sample_data):
        """Test that z-scores have mean~0 and std~1."""
        z = univariate.z_score(sample_data)

        assert np.abs(np.mean(z)) < 1e-10
        assert np.abs(np.std(z, ddof=1) - 1.0) < 1e-10

    def test_with_nan(self):
        """Test handling of NaN values."""
        data = np.array([1, 2, np.nan, 4, 5])
        z = univariate.z_score(data, remove_nan=True)

        # Should have NaN in same position
        assert np.isnan(z[2])
        # Other values should be normalized
        assert not np.isnan(z[0])


class TestSkewnessKurtosis:
    """Test skewness and kurtosis functions."""

    def test_normal_distribution(self):
        """Test that normal distribution has skewness~0 and kurtosis~0."""
        np.random.seed(42)
        data = np.random.randn(10000)

        skew = univariate.skewness(data)
        kurt = univariate.kurtosis(data, fisher=True)

        # Should be close to 0 for normal distribution
        assert abs(skew) < 0.1
        assert abs(kurt) < 0.1

    def test_positive_skew(self):
        """Test positive skew with lognormal data."""
        np.random.seed(42)
        data = np.random.lognormal(0, 1, 1000)

        skew = univariate.skewness(data)
        assert skew > 0  # Right-skewed


class TestCoefficientOfVariation:
    """Test coefficient_of_variation function."""

    def test_known_value(self):
        """Test with known CV value."""
        data = np.array([10, 12, 14, 16, 18])
        cv = univariate.coefficient_of_variation(data)

        # Manual calculation: std/mean
        expected_cv = np.std(data, ddof=1) / np.mean(data)
        assert np.isclose(cv, expected_cv)

    def test_zero_mean(self):
        """Test that zero mean returns infinity."""
        data = np.array([-1, 0, 1])
        cv = univariate.coefficient_of_variation(data)
        assert cv == np.inf


class TestModeEstimate:
    """Test mode_estimate function."""

    def test_simple_mode(self):
        """Test simple mode method."""
        data = np.array([1, 2, 2, 3, 3, 3, 4, 4])
        mode = univariate.mode_estimate(data, method="simple")
        assert mode == 3

    def test_kernel_mode(self, sample_data):
        """Test kernel density estimation mode."""
        mode = univariate.mode_estimate(sample_data, method="kernel")
        # Should return a float
        assert isinstance(mode, (float, np.floating))

    def test_invalid_method(self, sample_data):
        """Test that invalid method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown method"):
            univariate.mode_estimate(sample_data, method="invalid")
