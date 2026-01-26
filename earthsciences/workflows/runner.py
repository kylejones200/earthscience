"""
Configuration-driven workflow runner.
"""

from pathlib import Path
from earthsciences.config.parser import load_config
from earthsciences.workflows.pipeline import GeostatsPipeline


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
            print(f"\n{'='*70}")
            print(f"GEOSTATISTICAL ANALYSIS")
            print(f"{'='*70}")
            print(f"Config file: {self.config_path}")
            print(f"Project: {self.config.project.name}")
            print(f"{'='*70}\n")
        
        # Run full pipeline
        results = self.pipeline.run_full_pipeline()
        
        return results
    
    def get_config(self):
        """Get configuration object."""
        return self.config
    
    def get_pipeline(self):
        """Get pipeline object."""
        return self.pipeline
