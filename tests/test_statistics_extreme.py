"""
Tests for extreme value statistics module
"""

import numpy as np
import pytest

from earthsciences.statistics import extreme_values


class TestGEV:
    """Test Generalized Extreme Value distribution."""

    def test_fit_gev(self):
        """Test fitting GEV distribution."""
        np.random.seed(42)
        data = np.random.gumbel(loc=10, scale=2, size=100)

        result = extreme_values.fit_gev(data)

        assert "location" in result
        assert "scale" in result
        assert "shape" in result
        assert result["scale"] > 0

    def test_gev_return_level(self):
        """Test GEV return level calculation."""
        params = {"location": 10, "scale": 2, "shape": 0.1}

        return_level = extreme_values.gev_return_level(params, return_period=100)

        assert return_level > params["location"]

    def test_gev_return_level_matches_return_level(self):
        """gev_return_level should agree with return_level for the same parameters."""
        params = {"location": 10, "scale": 2, "shape": 0.1}
        assert extreme_values.gev_return_level(params, 50) == extreme_values.return_level(
            params, 50
        )

    def test_gev_return_level_invalid_period(self):
        """Return period must be greater than one year."""
        params = {"location": 10, "scale": 2, "shape": 0.0}
        with pytest.raises(ValueError, match="greater than 1"):
            extreme_values.gev_return_level(params, return_period=1)
        with pytest.raises(ValueError, match="greater than 1"):
            extreme_values.return_level(params, return_period=0.5)


class TestGPD:
    """Test Generalized Pareto Distribution."""

    def test_fit_gpd(self):
        """Test fitting GPD to exceedances."""
        np.random.seed(42)
        data = np.random.exponential(scale=2, size=200)
        threshold = np.percentile(data, 90)

        result = extreme_values.fit_gpd(data, threshold=threshold)

        assert "scale" in result
        assert "shape" in result
        assert result["scale"] > 0

    def test_threshold_selection(self):
        """Test automatic threshold selection."""
        np.random.seed(42)
        data = np.random.exponential(scale=1, size=300)

        threshold = extreme_values.select_threshold(data)

        assert threshold > np.min(data)
        assert threshold < np.max(data)


class TestBlockMaxima:
    """Test block maxima method."""

    def test_extract_block_maxima(self):
        """Test extracting block maxima."""
        np.random.seed(42)
        data = np.random.randn(365)  # One year of daily data

        maxima = extreme_values.block_maxima(data, block_size=30)

        assert len(maxima) == 12  # 12 months
        assert np.all(maxima >= np.min(data))

    def test_annual_maxima(self):
        """Test annual maxima extraction."""
        np.random.seed(42)
        data = np.random.randn(1000)

        annual_max = extreme_values.annual_maxima(data, observations_per_year=100)

        assert len(annual_max) == 10


class TestReturnPeriod:
    """Test return period calculations."""

    def test_empirical_return_period(self):
        """Test empirical return period."""
        data = np.array([1, 2, 3, 4, 5, 10, 15, 20])

        result = extreme_values.empirical_return_period(data)

        assert "values" in result
        assert "return_periods" in result
        assert len(result["return_periods"]) == len(result["values"])


class TestPOT:
    """Test Peaks Over Threshold method."""

    def test_pot_analysis(self):
        """Test POT analysis."""
        np.random.seed(42)
        data = np.random.exponential(scale=1, size=500)
        threshold = np.percentile(data, 95)

        result = extreme_values.pot_analysis(data, threshold=threshold)

        assert "exceedances" in result
        assert "n_exceedances" in result or "count" in result
        assert len(result["exceedances"]) < len(data)


class TestExtremeValueIndex:
    """Test extreme value index estimation."""

    def test_hill_estimator(self):
        """Test Hill estimator for tail index."""
        np.random.seed(42)
        data = np.random.pareto(a=2, size=200)

        xi = extreme_values.hill_estimator(data, k=50)

        assert xi > 0  # Pareto has positive tail index


class TestReturnLevelPlot:
    """Test return level plot data."""

    def test_return_level_plot_data(self):
        """Test generating return level plot data."""
        np.random.seed(42)
        data = np.random.gumbel(loc=5, scale=1, size=100)

        result = extreme_values.return_level_plot(data)

        assert "return_periods" in result
        assert "return_levels" in result
        assert len(result["return_periods"]) == len(result["return_levels"])
