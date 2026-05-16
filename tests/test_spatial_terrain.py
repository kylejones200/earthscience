"""Tests for DEM terrain analysis."""

import numpy as np
import pytest

from earthsciences.spatial.terrain_analysis import (
    calculate_flow_accumulation,
    curvature,
    hillshade,
    slope_aspect,
    stream_power_index,
    terrain_ruggedness_index,
    topographic_position_index,
    topographic_wetness_index,
)


@pytest.fixture
def planar_dem():
    """Tilted plane: elevation increases east and north."""
    rows, cols = 20, 20
    dem = np.add.outer(np.arange(rows, dtype=float), np.arange(cols, dtype=float))
    return dem


@pytest.fixture
def bowl_dem():
    x = np.linspace(-4, 4, 25)
    y = np.linspace(-4, 4, 25)
    xx, yy = np.meshgrid(x, y)
    return 100.0 - 0.5 * (xx**2 + yy**2)


class TestSlopeAspect:
    def test_planar_slope_positive(self, planar_dem):
        slope, aspect = slope_aspect(planar_dem, cell_size=1.0, degrees=True)
        assert np.nanmax(slope) > 0
        assert np.nanmin(slope) >= 0
        assert aspect.shape == slope.shape

    def test_bowl_max_slope_near_rim(self, bowl_dem):
        slope, _ = slope_aspect(bowl_dem, cell_size=0.5)
        center = slope[12, 12]
        edge = slope[0, 12]
        assert edge > center


class TestCurvatureAndHillshade:
    def test_curvature_keys(self, bowl_dem):
        curv = curvature(bowl_dem, cell_size=1.0)
        assert {"profile", "planform", "mean"}.issubset(curv.keys())
        assert curv["mean"].shape == bowl_dem.shape

    def test_hillshade_range(self, bowl_dem):
        hs = hillshade(bowl_dem, cell_size=1.0, azimuth=315.0, altitude=45.0)
        assert hs.shape == bowl_dem.shape
        assert np.nanmin(hs) >= 0
        assert np.nanmax(hs) <= 255


class TestHydrologyIndices:
    def test_twi_positive(self, bowl_dem):
        twi = topographic_wetness_index(bowl_dem, cell_size=1.0)
        assert twi.shape == bowl_dem.shape
        assert np.nanmax(twi) > 0

    def test_flow_accumulation_nonnegative(self, bowl_dem):
        acc = calculate_flow_accumulation(bowl_dem, cell_size=1.0)
        assert np.all(acc >= 0)

    def test_stream_power_index(self, bowl_dem):
        spi = stream_power_index(bowl_dem, cell_size=1.0)
        assert spi.shape == bowl_dem.shape


class TestTerrainIndices:
    def test_terrain_ruggedness_index(self, bowl_dem):
        tri = terrain_ruggedness_index(bowl_dem, window_size=3)
        assert tri.shape == bowl_dem.shape
        assert np.nanmax(tri) >= 0

    def test_topographic_position_index(self, bowl_dem):
        tpi = topographic_position_index(bowl_dem, inner_radius=1, outer_radius=4)
        assert tpi.shape == bowl_dem.shape
