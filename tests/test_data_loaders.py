"""Tests for earthsciences.data loaders."""

from pathlib import Path

import pytest

from earthsciences.data import geochem_loaders, loaders


@pytest.fixture
def agdb_fixture_dir(monkeypatch):
    """Point AGDB loaders at minimal test fixtures."""
    fixture_dir = Path(__file__).parent / "fixtures" / "agdb"
    monkeypatch.setattr(geochem_loaders, "get_agdb_path", lambda: fixture_dir)
    return fixture_dir


class TestGeochemLoaders:
    def test_load_agdb_geology(self, agdb_fixture_dir):
        geol = geochem_loaders.load_agdb_geology()
        assert len(geol) == 4
        assert {"LATITUDE", "LONGITUDE"}.issubset(geol.columns)

    def test_load_agdb_chemistry_filters_elements(self, agdb_fixture_dir):
        chem = geochem_loaders.load_agdb_chemistry(["Cu"])
        assert set(chem["SPECIES"].unique()) == {"Cu"}
        assert len(chem) == 4

    def test_prepare_spatial_data(self, agdb_fixture_dir):
        spatial = geochem_loaders.prepare_spatial_data(["Cu"], min_samples=1)
        assert "LATITUDE" in spatial.columns
        assert "Cu" in spatial.columns
        assert len(spatial) >= 2

    def test_load_stream_sediments(self, agdb_fixture_dir):
        streams = geochem_loaders.load_stream_sediments(["Cu"])
        assert len(streams) >= 1
        assert streams["SAMPLE_SOURCE"].str.contains("stream", case=False).any()

    def test_get_element_stats(self, agdb_fixture_dir):
        stats = geochem_loaders.get_element_stats("Cu")
        assert stats["element"] == "Cu"
        assert stats["count"] == 4
        assert stats["median"] == pytest.approx(56.1, rel=0.01)


class TestElevationLoaders:
    def test_load_etopo2022_invalid_bbox(self):
        pytest.importorskip("xarray")
        with pytest.raises(ValueError, match="Longitude"):
            loaders.load_etopo2022((200, 10, 210, 20))

    def test_load_etopo2022_unknown_resolution(self):
        pytest.importorskip("xarray")
        with pytest.raises(ValueError, match="Unknown resolution"):
            loaders.load_etopo2022((-150, 60, -149, 61), resolution="invalid")

    def test_load_srtm_missing_tile(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="SRTM tile not found"):
            loaders.load_srtm(37.0, -122.0, data_dir=str(tmp_path))

    def test_load_gtopo30_missing_directory(self, tmp_path):
        missing = tmp_path / "empty_gtopo"
        with pytest.raises(FileNotFoundError, match="GTOPO30"):
            loaders.load_gtopo30((-125, 30, -115, 40), data_dir=str(missing))
