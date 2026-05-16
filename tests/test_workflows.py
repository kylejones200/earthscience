"""Tests for config-driven geostatistics workflows."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from earthsciences.config.parser import load_config, validate_config
from earthsciences.workflows.pipeline import GeostatsPipeline
from earthsciences.workflows.runner import ConfigRunner


@pytest.fixture
def pipeline_config(tmp_path: Path) -> Path:
    """Minimal valid pipeline config and CSV input."""
    data_file = Path(__file__).parent / "fixtures" / "spatial_points.csv"
    config = {
        "project": {"name": "Test run", "output_dir": str(tmp_path / "results")},
        "data": {
            "input_file": str(data_file),
            "x_column": "x",
            "y_column": "y",
            "z_column": "value",
        },
        "variogram": {"n_lags": 5, "models": ["spherical"], "auto_fit": True},
        "kriging": {
            "method": "ordinary",
            "grid": {
                "x_min": 0.0,
                "x_max": 4.0,
                "y_min": 0.0,
                "y_max": 4.0,
                "resolution": 1.0,
            },
        },
        "validation": {
            "cross_validation": True,
            "n_folds": 3,
            "metrics": ["rmse", "mae"],
        },
        "visualization": {"style": "default", "plots": [], "dpi": 72},
        "output": {"save_predictions": False, "save_plots": False, "save_report": False},
    }
    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text(yaml.dump(config), encoding="utf-8")
    return config_path


class TestWorkflowConfig:
    def test_validate_config(self, pipeline_config):
        valid, message = validate_config(pipeline_config)
        assert valid is True
        assert "valid" in message.lower()

    def test_load_config(self, pipeline_config):
        config = load_config(pipeline_config)
        assert config.project.name == "Test run"
        assert config.data.z_column == "value"


class TestGeostatsPipeline:
    def test_load_data(self, pipeline_config):
        config = load_config(pipeline_config)
        pipeline = GeostatsPipeline(config)
        pipeline.load_data()
        assert pipeline.data is not None
        assert len(pipeline.data["x"]) == 10
        assert pipeline.results["data_stats"]["n_points"] == 10

    def test_variogram_and_kriging(self, pipeline_config):
        config = load_config(pipeline_config)
        pipeline = GeostatsPipeline(config)
        pipeline.load_data()
        pipeline.fit_variogram()
        assert pipeline.variogram_model is not None
        pipeline.krige()
        assert pipeline.kriging_result is not None
        assert np.isfinite(pipeline.kriging_result["Z_pred"]).any()

    def test_cross_validation(self, pipeline_config):
        config = load_config(pipeline_config)
        pipeline = GeostatsPipeline(config)
        pipeline.load_data()
        pipeline.fit_variogram()
        pipeline.validate()
        assert "rmse" in pipeline.results["validation"]


class TestConfigRunner:
    def test_runner_returns_results(self, pipeline_config):
        runner = ConfigRunner(pipeline_config, verbose=False)
        results = runner.run()
        assert "data_stats" in results
        assert "variogram" in results
        assert Path(results["output_dir"]).exists()
