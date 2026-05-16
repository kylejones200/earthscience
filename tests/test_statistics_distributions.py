"""Tests for probability distribution fitting helpers."""

import numpy as np
import pytest

from earthsciences.statistics import distributions


class TestFitDistribution:
    def test_normal_fit_on_gaussian_data(self):
        np.random.seed(0)
        data = np.random.normal(loc=2.0, scale=0.5, size=500)
        result = distributions.fit_distribution(data, "norm")
        assert "params" in result
        assert result["aic"] < result["bic"] or result["bic"] > 0
        assert len(result["params"]) == 2

    def test_unknown_distribution_raises(self):
        with pytest.raises(ValueError, match="Unknown distribution"):
            distributions.fit_distribution(np.array([1.0, 2.0]), "not_a_dist")


class TestNormalityAndRandom:
    def test_shapiro_on_normal_data(self):
        np.random.seed(1)
        data = np.random.normal(size=100)
        statistic, p_value = distributions.test_normality(data, method="shapiro")
        assert 0 <= statistic <= 1
        assert 0 <= p_value <= 1

    def test_kstest_method(self):
        data = np.random.default_rng(2).normal(size=80)
        statistic, p_value = distributions.test_normality(data, method="kstest")
        assert statistic >= 0
        assert 0 <= p_value <= 1

    def test_generate_random_normal(self):
        samples = distributions.generate_random("norm", size=50, params=(1.0, 0.2), seed=0)
        assert len(samples) == 50
        assert np.std(samples) > 0

    def test_qq_plot_data_matches_sample_length(self):
        data = np.random.default_rng(3).normal(size=60)
        theoretical, sample = distributions.qq_plot_data(data, "norm")
        assert len(theoretical) == len(sample) == 60

    def test_histogram_bins_methods(self):
        data = np.random.default_rng(4).normal(size=200)
        for method in ("auto", "sturges", "scott", "fd", "sqrt"):
            n_bins = distributions.histogram_bins(data, method=method)
            assert n_bins >= 1

    def test_lognormal_stats(self):
        data = np.random.default_rng(5).lognormal(mean=0.0, sigma=0.4, size=150)
        stats = distributions.lognormal_stats(data)
        assert stats["geometric_mean"] > 0
        assert stats["arithmetic_mean"] > stats["geometric_mean"]
