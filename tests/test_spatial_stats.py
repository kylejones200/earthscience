"""Tests for spatial autocorrelation utilities."""

import numpy as np

from earthsciences.spatial.spatial_stats import (
    gearys_c,
    getis_ord_g,
    local_morans_i,
    morans_i,
    semivariogram,
    spatial_correlogram,
    spatial_weights_matrix,
)


class TestSpatialWeights:
    def test_inverse_distance_weights(self):
        coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        w = spatial_weights_matrix(coords, method="inverse_distance")
        assert w.shape == (3, 3)
        assert np.allclose(w.sum(axis=1), 1.0)
        assert np.allclose(np.diag(w), 0.0)

    def test_knn_weights(self):
        coords = np.random.default_rng(0).random((12, 2))
        w = spatial_weights_matrix(coords, method="knn", k_neighbors=3)
        assert w.shape == (12, 12)
        assert np.all(w.sum(axis=1) > 0)

    def test_threshold_weights(self):
        coords = np.array([[0.0, 0.0], [0.5, 0.0], [5.0, 5.0]])
        w = spatial_weights_matrix(coords, method="threshold", distance_threshold=1.0)
        assert w[0, 1] > 0
        assert w[0, 2] == 0


class TestMoransI:
    def test_clustered_pattern_positive_i(self):
        rng = np.random.default_rng(1)
        coords = rng.random((40, 2)) * 10
        values = coords[:, 0] + coords[:, 1]
        result = morans_i(values, coords)
        assert "I" in result
        assert result["I"] > result["EI"]
        assert 0 <= result["p_value"] <= 1


class TestGearysC:
    def test_returns_statistic(self):
        coords = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        values = np.array([1.0, 2.0, 3.0, 4.0])
        result = gearys_c(values, coords)
        assert "C" in result
        assert "p_value" in result


class TestLocalMoransI:
    def test_lisa_output_length(self):
        rng = np.random.default_rng(5)
        coords = rng.random((25, 2)) * 10
        values = rng.normal(size=25)
        result = local_morans_i(values, coords)
        assert len(result["Ii"]) == 25
        assert len(result["significant"]) == 25


class TestGetisOrdG:
    def test_hotspot_statistic(self):
        coords = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [5, 5]], dtype=float)
        values = np.array([10, 12, 11, 13, 1], dtype=float)
        result = getis_ord_g(values, coords)
        assert "Gi_star" in result
        assert len(result["hot_spots"]) == 5


class TestCorrelogramAndSemivariogram:
    def test_spatial_correlogram_lags(self):
        rng = np.random.default_rng(6)
        coords = rng.random((30, 2)) * 10
        values = coords[:, 0] + rng.normal(scale=0.1, size=30)
        result = spatial_correlogram(values, coords, n_lags=5)
        assert len(result["distances"]) == 5
        assert len(result["morans_i"]) == 5

    def test_semivariogram_increases_with_lag(self):
        rng = np.random.default_rng(7)
        coords = rng.random((40, 2)) * 10
        values = np.sin(coords[:, 0]) + rng.normal(scale=0.05, size=40)
        lags, gamma = semivariogram(values, coords, n_lags=6)
        assert len(lags) == len(gamma)
        assert gamma[-1] >= gamma[0]
