"""Tests for spatial point pattern analysis."""

import numpy as np
import pytest

from earthsciences.spatial.point_patterns import (
    clark_evans_statistic,
    f_function,
    g_function,
    nearest_neighbor_distance,
    point_density,
    ripley_k,
)


class TestNearestNeighbor:
    def test_two_point_distance(self):
        x = np.array([0.0, 1.0])
        y = np.array([0.0, 0.0])
        d = nearest_neighbor_distance(x, y, k=1)
        assert d[0] == pytest.approx(1.0)
        assert d[1] == pytest.approx(1.0)

    def test_kth_neighbor(self):
        rng = np.random.default_rng(0)
        x = rng.random(20)
        y = rng.random(20)
        d1 = nearest_neighbor_distance(x, y, k=1)
        d2 = nearest_neighbor_distance(x, y, k=2)
        assert np.all(d2 >= d1)


class TestRipleyK:
    def test_two_points_at_unit_separation(self):
        x = np.array([0.0, 1.0])
        y = np.array([0.0, 0.0])
        dist = np.array([0.5, 1.5])
        k = ripley_k(x, y, dist, area=1.0)
        assert k[0] == pytest.approx(0.0)
        assert k[1] == pytest.approx(1.0)

    def test_increases_with_distance(self):
        rng = np.random.default_rng(1)
        x = rng.random(30) * 10
        y = rng.random(30) * 10
        dist = np.linspace(0.5, 8.0, 10)
        k = ripley_k(x, y, dist)
        assert np.all(np.diff(k) >= 0)


class TestClarkEvans:
    def test_clustered_points_low_r(self):
        rng = np.random.default_rng(2)
        x = np.concatenate([rng.normal(5, 0.3, 25), rng.normal(15, 0.3, 25)])
        y = np.concatenate([rng.normal(5, 0.3, 25), rng.normal(15, 0.3, 25)])
        r, z = clark_evans_statistic(x, y, area=400.0)
        assert r < 1.0
        assert isinstance(z, float)


class TestGAndF:
    def test_g_function_bounds(self):
        rng = np.random.default_rng(3)
        x = rng.random(40) * 5
        y = rng.random(40) * 5
        dist = np.linspace(0.01, 3.0, 15)
        g = g_function(x, y, dist)
        assert np.all(g >= 0)
        assert np.all(g <= 1)
        assert g[-1] == pytest.approx(1.0, rel=0.05)

    def test_f_function_monotone(self):
        rng = np.random.default_rng(4)
        x = rng.random(35) * 8
        y = rng.random(35) * 8
        dist = np.linspace(0.05, 4.0, 12)
        f = f_function(x, y, dist)
        assert np.all(np.diff(f) >= -1e-10)


class TestPointDensity:
    def test_density_grid_shape(self):
        rng = np.random.default_rng(8)
        x = rng.random(30) * 4
        y = rng.random(30) * 4
        gx, gy = np.meshgrid(np.linspace(0, 4, 8), np.linspace(0, 4, 8))
        density = point_density(x, y, gx, gy)
        assert density.shape == gx.shape
        assert np.all(density >= 0)
