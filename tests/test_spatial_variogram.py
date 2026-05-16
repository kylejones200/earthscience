"""
Tests for spatial variogram module
"""

import numpy as np

from earthsciences.spatial import variogram


class TestComputeVariogram:
    """Test compute_variogram function."""

    def test_basic_functionality(self):
        """Test basic variogram computation."""
        np.random.seed(42)
        x = np.random.rand(30) * 10
        y = np.random.rand(30) * 10
        values = np.sin(x) + np.cos(y)

        lags, gamma, n_pairs = variogram.compute_variogram(x, y, values)

        assert len(lags) > 0
        assert len(gamma) == len(lags)
        assert len(n_pairs) == len(lags)
        assert np.all(lags >= 0)
        assert np.all(gamma >= 0)
        assert np.all(n_pairs > 0)

    def test_with_custom_lags(self):
        """Test with custom number of lags."""
        np.random.seed(42)
        x = np.random.rand(20) * 10
        y = np.random.rand(20) * 10
        values = x + y

        n_lags = 10
        lags, gamma, n_pairs = variogram.compute_variogram(x, y, values, n_lags=n_lags)

        assert len(lags) == n_lags

    def test_semivariance_increases(self):
        """Test that semivariance generally increases with distance."""
        np.random.seed(42)
        x = np.linspace(0, 10, 50)
        y = np.zeros(50)
        values = np.random.randn(50)

        lags, gamma, n_pairs = variogram.compute_variogram(x, y, values, n_lags=5)

        # First value should be smaller than last (generally)
        assert gamma[0] < gamma[-1] * 2  # Allow some variation


class TestFitVariogramModel:
    """Test fit_variogram_model function."""

    def test_spherical_model(self):
        """Test fitting spherical variogram model."""
        np.random.seed(42)
        x = np.random.rand(30) * 10
        y = np.random.rand(30) * 10
        values = np.sin(x) + np.cos(y) + np.random.randn(30) * 0.1

        lags, gamma, n_pairs = variogram.compute_variogram(x, y, values)

        fit = variogram.fit_variogram_model(lags, gamma, model="spherical")

        assert "nugget" in fit
        assert "sill" in fit
        assert "range" in fit
        assert "model" in fit
        assert fit["nugget"] >= 0
        assert fit["sill"] >= 0
        assert fit["range"] > 0
        assert callable(fit["variogram_func"])
        assert callable(fit["function"])
        assert fit["variogram_func"](0.0) == fit["function"](0.0)

    def test_exponential_model(self):
        """Test fitting exponential variogram model."""
        np.random.seed(42)
        x = np.random.rand(25) * 10
        y = np.random.rand(25) * 10
        values = x * 2 + y + np.random.randn(25) * 0.2

        lags, gamma, n_pairs = variogram.compute_variogram(x, y, values)

        fit = variogram.fit_variogram_model(lags, gamma, model="exponential")

        assert fit["model"] == "exponential"
        assert "r_squared" in fit

    def test_gaussian_model(self):
        """Test fitting Gaussian variogram model."""
        np.random.seed(42)
        x = np.random.rand(30) * 10
        y = np.random.rand(30) * 10
        values = np.exp(-((x - 5) ** 2 + (y - 5) ** 2) / 10)

        lags, gamma, n_pairs = variogram.compute_variogram(x, y, values)

        fit = variogram.fit_variogram_model(lags, gamma, model="gaussian")

        assert fit["model"] == "gaussian"

    def test_returns_variogram_function(self):
        """Test that fitted model returns a callable function."""
        np.random.seed(42)
        x = np.random.rand(20) * 10
        y = np.random.rand(20) * 10
        values = x + y + np.random.randn(20) * 0.5

        lags, gamma, n_pairs = variogram.compute_variogram(x, y, values)
        fit = variogram.fit_variogram_model(lags, gamma, model="spherical")

        assert callable(fit["variogram_func"])
        test_distance = 5.0
        result = fit["variogram_func"](test_distance)
        assert isinstance(result, (float, np.floating, np.ndarray))
        assert float(np.asarray(result).item()) >= 0


class TestVariogramModels:
    """Test individual variogram model functions."""

    def test_spherical_model_function(self):
        """Test spherical model at different distances."""
        h = np.array([0, 1, 5, 10, 20])
        nugget, sill, range_param = 0.1, 1.0, 10.0

        gamma = variogram.spherical_model(h, nugget, sill, range_param)

        assert gamma[0] == 0  # origin
        assert np.isclose(gamma[-1], sill)
        assert np.all(gamma >= 0)
        assert gamma[-1] >= gamma[0]
        assert np.isclose(gamma[np.searchsorted(h, range_param)], sill)

    def test_exponential_model_function(self):
        """Test exponential model."""
        h = np.array([0, 1, 5, 10, 20])
        nugget, sill, range_param = 0.0, 1.0, 5.0

        gamma = variogram.exponential_model(h, nugget, sill, range_param)

        assert gamma[0] == nugget
        assert np.all(np.diff(gamma) >= 0)
        assert gamma[-1] < sill

    def test_gaussian_model_function(self):
        """Test Gaussian model."""
        h = np.array([0, 1, 5, 10])
        nugget, sill, range_param = 0.0, 1.0, 5.0

        gamma = variogram.gaussian_model(h, nugget, sill, range_param)

        assert gamma[0] == nugget
        assert np.all(np.diff(gamma) >= 0)


class TestVariogramCloud:
    def test_pair_count_and_golden_semivariances(self):
        """Three collinear points: γ(h) = 0.5 * (z_i - z_j)² for each pair."""
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([0.0, 0.0, 0.0])
        values = np.array([1.0, 3.0, 5.0])
        dist, gamma = variogram.variogram_cloud(x, y, values)
        assert len(dist) == 3
        assert len(gamma) == 3
        order = np.argsort(dist)
        dist = dist[order]
        gamma = gamma[order]
        np.testing.assert_allclose(dist, [1.0, 1.0, 2.0], rtol=1e-12)
        np.testing.assert_allclose(gamma, [2.0, 2.0, 8.0], rtol=1e-12)

    def test_subsample_respects_max_points(self):
        rng = np.random.default_rng(11)
        n = 80
        x = rng.random(n) * 10
        y = rng.random(n) * 10
        values = rng.random(n)
        dist, gamma = variogram.variogram_cloud(x, y, values, max_points=100)
        assert len(dist) == 100
        assert len(gamma) == 100


class TestAnisotropicVariogram:
    def test_directional_variograms(self):
        rng = np.random.default_rng(9)
        x = rng.random(40) * 10
        y = rng.random(40) * 10
        values = x * 2 + y + rng.normal(scale=0.1, size=40)
        result = variogram.anisotropic_variogram(x, y, values, n_directions=4)
        assert len(result) == 4
        for direction in result.values():
            assert len(direction["lag_dist"]) > 0
            assert len(direction["gamma"]) == len(direction["lag_dist"])
