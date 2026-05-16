"""
Tests for statistics hypothesis testing module
"""

import numpy as np

from earthsciences.statistics import hypothesis_tests


class TestTTest:
    """Test t-tests."""

    def test_one_sample(self):
        """Test one-sample t-test."""
        np.random.seed(42)
        data = np.random.randn(30) + 5.0

        result = hypothesis_tests.t_test_1samp(data, popmean=5.0)

        assert "statistic" in result
        assert "p_value" in result or "pvalue" in result

    def test_two_sample_independent(self):
        """Test two-sample independent t-test."""
        np.random.seed(42)
        group1 = np.random.randn(25)
        group2 = np.random.randn(30) + 0.5

        result = hypothesis_tests.t_test_2samp(group1, group2)

        assert "statistic" in result
        assert "p_value" in result or "pvalue" in result

    def test_paired_samples(self):
        """Test paired t-test."""
        np.random.seed(42)
        before = np.random.randn(20) + 10
        after = before + np.random.randn(20) * 0.5 + 1.0

        result = hypothesis_tests.t_test_paired(before, after)

        assert "statistic" in result
        assert "p_value" in result or "pvalue" in result


class TestANOVA:
    """Test ANOVA."""

    def test_one_way_anova(self):
        """Test one-way ANOVA."""
        np.random.seed(42)
        group1 = np.random.randn(20)
        group2 = np.random.randn(20) + 0.5
        group3 = np.random.randn(20) + 1.0

        result = hypothesis_tests.anova_oneway(group1, group2, group3)

        assert "F_statistic" in result or "statistic" in result
        assert "p_value" in result or "pvalue" in result


class TestChiSquare:
    """Test chi-square tests."""

    def test_chi_square_goodness_of_fit(self):
        """Test chi-square goodness of fit."""
        observed = np.array([10, 15, 20, 25])
        expected = np.array([15, 15, 20, 20])

        result = hypothesis_tests.chi_square_test(observed, expected)

        assert "statistic" in result
        assert "p_value" in result or "pvalue" in result

    def test_chi_square_independence(self):
        """Test chi-square test of independence."""
        contingency_table = np.array([[10, 15], [20, 25]])

        result = hypothesis_tests.chi_square_independence(contingency_table)

        assert "statistic" in result
        assert "p_value" in result or "pvalue" in result


class TestKolmogorovSmirnov:
    """Test Kolmogorov-Smirnov test."""

    def test_one_sample(self):
        """Test one-sample KS test."""
        np.random.seed(42)
        data = np.random.randn(100)

        result = hypothesis_tests.ks_test_1samp(data, distribution="norm")

        assert "statistic" in result
        assert "p_value" in result or "pvalue" in result

    def test_two_sample(self):
        """Test two-sample KS test."""
        np.random.seed(42)
        sample1 = np.random.randn(50)
        sample2 = np.random.randn(60) + 0.3

        result = hypothesis_tests.ks_test_2samp(sample1, sample2)

        assert "statistic" in result
        assert "p_value" in result or "pvalue" in result


class TestMannWhitneyU:
    """Test Mann-Whitney U test."""

    def test_basic_functionality(self):
        """Test Mann-Whitney U test."""
        np.random.seed(42)
        group1 = np.random.rand(30)
        group2 = np.random.rand(40) + 0.2

        result = hypothesis_tests.mann_whitney_u(group1, group2)

        assert "statistic" in result
        assert "p_value" in result or "pvalue" in result


class TestWilcoxonTest:
    """Test Wilcoxon tests."""

    def test_signed_rank(self):
        """Test Wilcoxon signed-rank test."""
        np.random.seed(42)
        before = np.random.randn(25)
        after = before + np.random.randn(25) * 0.3 + 0.5

        result = hypothesis_tests.wilcoxon_test(before, after)

        assert "statistic" in result
        assert "p_value" in result or "pvalue" in result


class TestKruskalWallis:
    """Test Kruskal-Wallis test."""

    def test_basic_functionality(self):
        """Test Kruskal-Wallis H test."""
        np.random.seed(42)
        group1 = np.random.randn(20)
        group2 = np.random.randn(25) + 0.5
        group3 = np.random.randn(30) + 1.0

        result = hypothesis_tests.kruskal_wallis(group1, group2, group3)

        assert "statistic" in result or "H" in result
        assert "p_value" in result or "pvalue" in result
