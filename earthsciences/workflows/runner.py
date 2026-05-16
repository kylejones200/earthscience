"""
Configuration-driven workflow runner.
"""

import logging
from pathlib import Path

from earthsciences.config.parser import load_config
from earthsciences.utils.logging_config import log_section
from earthsciences.workflows.pipeline import GeostatsPipeline

logger = logging.getLogger(__name__)


class ConfigRunner:
    """
    Run complete geostatistical workflow from configuration file.

    Parameters
    ----------
    config_path : str or Path
        Path to YAML configuration file
    verbose : bool, optional
        Print verbose output (default: True)

    Examples
    --------
    >>> runner = ConfigRunner('my_analysis.yaml')
    >>> results = runner.run()
    >>> print(f"Analysis complete. R² = {results['validation']['r2']:.3f}")
    """

    def __init__(self, config_path, verbose=True):
        self.config_path = Path(config_path).expanduser()
        self.config = load_config(config_path)
        self.pipeline = GeostatsPipeline(self.config)
        self.verbose = verbose

    def run(self):
        """
        Execute full geostatistical workflow.

        Returns
        -------
        results : dict
            Dictionary containing all analysis results including:
            - data_stats: Input data statistics
            - preprocessing: Preprocessing summary
            - variogram: Variogram model parameters
            - kriging: Kriging predictions summary
            - validation: Cross-validation metrics
            - output_dir: Path to output directory
            - plots: List of generated plot paths
        """
        if self.verbose:
            log_section("GEOSTATISTICAL ANALYSIS")
            logger.info("Config file: %s", self.config_path)
            logger.info("Project: %s", self.config.project.name)

        # Run full pipeline
        results = self.pipeline.run_full_pipeline()

        return results

    def get_config(self):
        """Get configuration object."""
        return self.config

    def get_pipeline(self):
        """Get pipeline object."""
        return self.pipeline
