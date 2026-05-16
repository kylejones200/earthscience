"""
Tests for PCA module
"""

import numpy as np

from earthsciences.multivariate import pca


class TestPrincipalComponentAnalysis:
    """Test principal_component_analysis function."""

    def test_basic_functionality(self, sample_data_2d):
        """Test PCA basic functionality."""
        result = pca.principal_component_analysis(sample_data_2d, n_components=3)

        expected_keys = [
            "scores",
            "loadings",
            "explained_variance",
            "explained_variance_ratio",
            "cumulative_variance",
            "eigenvalues",
            "mean",
            "pca_object",
        ]

        for key in expected_keys:
            assert key in result

    def test_scores_shape(self, sample_data_2d):
        """Test that scores have correct shape."""
        n_samples, n_features = sample_data_2d.shape
        n_components = 3

        result = pca.principal_component_analysis(sample_data_2d, n_components=n_components)

        assert result["scores"].shape == (n_samples, n_components)

    def test_loadings_shape(self, sample_data_2d):
        """Test that loadings have correct shape."""
        n_features = sample_data_2d.shape[1]
        n_components = 3

        result = pca.principal_component_analysis(sample_data_2d, n_components=n_components)

        assert result["loadings"].shape == (n_features, n_components)

    def test_cumulative_variance_increasing(self, sample_data_2d):
        """Test that cumulative variance is increasing."""
        result = pca.principal_component_analysis(sample_data_2d)

        cum_var = result["cumulative_variance"]
        assert np.all(np.diff(cum_var) >= 0)  # Should be non-decreasing

    def test_variance_sums_to_one(self, sample_data_2d):
        """Test that explained variance ratios sum to approximately 1."""
        result = pca.principal_component_analysis(sample_data_2d)

        total_var = np.sum(result["explained_variance_ratio"])
        assert np.isclose(total_var, 1.0, atol=1e-10)

    def test_without_standardization(self, sample_data_2d):
        """Test PCA without standardization."""
        result = pca.principal_component_analysis(sample_data_2d, standardize=False)

        assert "scores" in result
        assert "loadings" in result


class TestPCABiplot:
    """Test pca_biplot function."""

    def test_returns_dict(self, sample_data_2d):
        """Test that biplot returns dict."""
        result = pca.pca_biplot(sample_data_2d)

        expected_keys = ["scores", "loadings", "feature_names", "explained_variance"]

        for key in expected_keys:
            assert key in result

    def test_scores_shape(self, sample_data_2d):
        """Test biplot scores shape."""
        result = pca.pca_biplot(sample_data_2d, pc1=0, pc2=1)

        n_samples = sample_data_2d.shape[0]
        assert result["scores"].shape == (n_samples, 2)

    def test_custom_feature_names(self, sample_data_2d):
        """Test with custom feature names."""
        feature_names = ["A", "B", "C", "D", "E"]
        result = pca.pca_biplot(sample_data_2d, feature_names=feature_names)

        assert result["feature_names"] == feature_names


class TestScreePlotData:
    """Test scree_plot_data function."""

    def test_returns_dict(self, sample_data_2d):
        """Test that scree plot data returns dict."""
        result = pca.scree_plot_data(sample_data_2d)

        expected_keys = ["explained_variance", "cumulative_variance", "eigenvalues"]

        for key in expected_keys:
            assert key in result

    def test_variance_lengths_match(self, sample_data_2d):
        """Test that variance arrays have same length."""
        result = pca.scree_plot_data(sample_data_2d)

        n_features = sample_data_2d.shape[1]
        assert len(result["explained_variance"]) == n_features
        assert len(result["cumulative_variance"]) == n_features
        assert len(result["eigenvalues"]) == n_features


class TestFactorAnalysis:
    """Test factor_analysis function."""

    def test_basic_functionality(self, sample_data_2d):
        """Test factor analysis basic functionality."""
        result = pca.factor_analysis(sample_data_2d, n_factors=2)

        expected_keys = ["scores", "loadings", "noise_variance", "fa_object"]

        for key in expected_keys:
            assert key in result

    def test_scores_shape(self, sample_data_2d):
        """Test factor analysis scores shape."""
        n_samples = sample_data_2d.shape[0]
        n_factors = 2

        result = pca.factor_analysis(sample_data_2d, n_factors=n_factors)

        assert result["scores"].shape == (n_samples, n_factors)


class TestRotationVarimax:
    """Test rotation_varimax function."""

    def test_basic_functionality(self, sample_data_2d):
        """Test varimax rotation basic functionality."""
        pca_result = pca.principal_component_analysis(sample_data_2d, n_components=3)

        rotated, rotation_matrix = pca.rotation_varimax(pca_result["loadings"])

        assert rotated.shape == pca_result["loadings"].shape
        assert rotation_matrix.shape == (3, 3)

    def test_rotation_matrix_orthogonal(self, sample_data_2d):
        """Test that rotation matrix is orthogonal."""
        pca_result = pca.principal_component_analysis(sample_data_2d, n_components=3)

        rotated, rotation_matrix = pca.rotation_varimax(pca_result["loadings"])

        # R^T @ R should be identity
        identity = rotation_matrix.T @ rotation_matrix
        assert np.allclose(identity, np.eye(3), atol=1e-10)
