"""
Tests for statistics resampling module
"""

import numpy as np

from earthsciences.statistics import resampling


class TestBootstrap:
    """Test bootstrap resampling."""

    def test_basic_functionality(self):
        """Test basic bootstrap."""
        np.random.seed(42)
        data = np.random.randn(50)

        result = resampling.bootstrap(data, statistic=np.mean, n_bootstrap=1000)

        assert "estimate" in result
        assert "confidence_interval" in result or "ci" in result
        assert "bootstrap_samples" in result or "samples" in result

    def test_confidence_interval(self):
        """Test that CI contains true mean."""
        np.random.seed(42)
        data = np.random.randn(100) + 5.0  # Mean = 5

        result = resampling.bootstrap(data, statistic=np.mean, n_bootstrap=2000)

        ci_key = "confidence_interval" if "confidence_interval" in result else "ci"
        ci = result[ci_key]

        # CI should contain true mean
        assert ci[0] < 5.0 < ci[1]

    def test_custom_statistic(self):
        """Test bootstrap with custom statistic."""
        data = np.random.randn(100)

        def custom_stat(x):
            return np.percentile(x, 75)

        result = resampling.bootstrap(data, statistic=custom_stat, n_bootstrap=500)

        assert "estimate" in result


class TestJackknife:
    """Test jackknife resampling."""

    def test_basic_functionality(self):
        """Test basic jackknife."""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        result = resampling.jackknife(data, statistic=np.mean)

        assert "estimate" in result
        assert "bias" in result
        assert "standard_error" in result or "se" in result

    def test_bias_calculation(self):
        """Test jackknife bias calculation."""
        np.random.seed(42)
        data = np.random.randn(50)

        result = resampling.jackknife(data, statistic=np.std)

        assert "bias" in result
        assert isinstance(result["bias"], (int, float, np.number))


class TestPermutationTest:
    """Test permutation tests."""

    def test_two_sample_test(self):
        """Test permutation test for two samples."""
        np.random.seed(42)
        group1 = np.random.randn(30) + 1.0
        group2 = np.random.randn(30)

        result = resampling.permutation_test(group1, group2, n_permutations=1000)

        assert "p_value" in result or "pvalue" in result
        assert "statistic" in result or "test_statistic" in result

    def test_detects_difference(self):
        """Test that it detects real differences."""
        np.random.seed(42)
        group1 = np.random.randn(50) + 5.0  # Mean = 5
        group2 = np.random.randn(50)  # Mean = 0

        result = resampling.permutation_test(group1, group2, n_permutations=2000)

        p_key = "p_value" if "p_value" in result else "pvalue"
        assert result[p_key] < 0.05


class TestCrossValidation:
    """Test cross-validation methods."""

    def test_kfold(self):
        """Test k-fold cross-validation."""
        X = np.random.randn(100, 5)
        y = np.random.randn(100)

        result = resampling.cross_validate(X, y, k=5)

        assert "scores" in result or "cv_scores" in result
        assert "mean_score" in result or len(result["scores"]) == 5

    def test_leave_one_out(self):
        """Test leave-one-out cross-validation."""
        X = np.random.randn(20, 3)
        y = np.random.randn(20)

        result = resampling.cross_validate(X, y, method="loo")

        assert "scores" in result or "cv_scores" in result


class TestMonteCarloSimulation:
    """Test Monte Carlo simulation."""

    def test_basic_simulation(self):
        """Test Monte Carlo simulation."""

        def simulation_func():
            return np.random.randn(10).mean()

        result = resampling.monte_carlo(simulation_func, n_simulations=1000)

        assert "results" in result or "simulations" in result
        assert len(result["results" if "results" in result else "simulations"]) == 1000

    def test_confidence_interval(self):
        """Test Monte Carlo confidence interval."""

        def simulation_func():
            return np.random.randn(50).mean()

        result = resampling.monte_carlo(simulation_func, n_simulations=2000)

        assert "mean" in result or "estimate" in result
        assert "confidence_interval" in result or "ci" in result
