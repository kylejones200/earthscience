"""
Template configuration file generators.
"""

import yaml
from pathlib import Path
from typing import Union


def generate_template(output_path: Union[str, Path] = "analysis_config.yaml", 
                     template_type: str = "full"):
    """
    Generate template configuration file.
    
    Parameters
    ----------
    output_path : str or Path
        Output file path
    template_type : str
        Template type: 'full', 'minimal', 'kriging', 'validation'
    """
    templates = {
        'full': _full_template(),
        'minimal': _minimal_template(),
        'kriging': _kriging_template(),
        'validation': _validation_template()
    }
    
    if template_type not in templates:
        raise ValueError(f"Unknown template type: {template_type}. "
                        f"Choose from: {list(templates.keys())}")
    
    template = templates[template_type]
    output_path = Path(output_path).expanduser()
    
    with open(output_path, 'w') as f:
        yaml.safe_dump(template, f, default_flow_style=False, sort_keys=False)
    
    print(f"✓ Template created: {output_path}")
    print(f"  Edit the file and run: earthsciences run --config {output_path}")


def _full_template() -> dict:
    """Full template with all options."""
    return {
        'project': {
            'name': 'My Geostatistical Analysis',
            'output_dir': './results',
            'description': 'Description of the analysis project'
        },
        'data': {
            'input_file': 'data/my_data.csv',
            'x_column': 'LONGITUDE',
            'y_column': 'LATITUDE',
            'z_column': 'VALUE'
        },
        'preprocessing': {
            'remove_outliers': True,
            'outlier_method': 'iqr',
            'outlier_threshold': 3.0,
            'transform': 'log',
            'handle_negatives': True
        },
        'variogram': {
            'n_lags': 15,
            'max_lag': None,
            'lag_tolerance': 0.5,
            'models': ['spherical', 'exponential', 'gaussian'],
            'auto_fit': True,
            'fit_method': 'wls',
            'direction': None,
            'tolerance': None
        },
        'kriging': {
            'method': 'ordinary',
            'drift_terms': None,
            'neighborhood': {
                'max_neighbors': 25,
                'min_neighbors': 3,
                'search_radius': None,
                'use_quadrant_search': False
            },
            'grid': {
                'x_min': -180.0,
                'x_max': -120.0,
                'y_min': 50.0,
                'y_max': 72.0,
                'resolution': 0.1
            },
            'compute_variance': True
        },
        'validation': {
            'cross_validation': True,
            'n_folds': 5,
            'method': 'kfold',
            'metrics': ['rmse', 'mae', 'r2'],
            'stratified': False
        },
        'visualization': {
            'style': 'minimalist',
            'plots': ['variogram', 'kriging_map', 'variance_map', 'cross_validation', 'histogram'],
            'colormap': 'viridis',
            'dpi': 300,
            'figsize': [10, 8],
            'contour_levels': 20
        },
        'output': {
            'save_predictions': True,
            'save_variance': True,
            'save_plots': True,
            'save_report': True,
            'formats': ['png', 'csv'],
            'compression': False
        }
    }


def _minimal_template() -> dict:
    """Minimal template with only required fields."""
    return {
        'project': {
            'name': 'Quick Analysis',
            'output_dir': './results'
        },
        'data': {
            'input_file': 'data/my_data.csv',
            'x_column': 'x',
            'y_column': 'y',
            'z_column': 'value'
        },
        'kriging': {
            'grid': {
                'x_min': 0.0,
                'x_max': 100.0,
                'y_min': 0.0,
                'y_max': 100.0,
                'resolution': 1.0
            }
        }
    }


def _kriging_template() -> dict:
    """Template focused on kriging parameters."""
    return {
        'project': {
            'name': 'Kriging Analysis',
            'output_dir': './kriging_results'
        },
        'data': {
            'input_file': 'data/sample_data.csv',
            'x_column': 'x',
            'y_column': 'y',
            'z_column': 'value'
        },
        'variogram': {
            'n_lags': 15,
            'models': ['spherical', 'exponential'],
            'auto_fit': True
        },
        'kriging': {
            'method': 'ordinary',
            'neighborhood': {
                'max_neighbors': 25,
                'min_neighbors': 5,
                'search_radius': 50.0
            },
            'grid': {
                'x_min': 0.0,
                'x_max': 100.0,
                'y_min': 0.0,
                'y_max': 100.0,
                'resolution': 1.0
            },
            'compute_variance': True
        },
        'visualization': {
            'plots': ['variogram', 'kriging_map', 'variance_map'],
            'colormap': 'viridis'
        }
    }


def _validation_template() -> dict:
    """Template focused on validation and assessment."""
    return {
        'project': {
            'name': 'Validation Study',
            'output_dir': './validation_results'
        },
        'data': {
            'input_file': 'data/validation_data.csv',
            'x_column': 'x',
            'y_column': 'y',
            'z_column': 'value'
        },
        'variogram': {
            'n_lags': 15,
            'models': ['spherical', 'exponential', 'gaussian'],
            'auto_fit': True
        },
        'kriging': {
            'method': 'ordinary',
            'grid': {
                'x_min': 0.0,
                'x_max': 100.0,
                'y_min': 0.0,
                'y_max': 100.0,
                'resolution': 2.0
            }
        },
        'validation': {
            'cross_validation': True,
            'n_folds': 10,
            'method': 'kfold',
            'metrics': ['rmse', 'mae', 'r2', 'mse', 'mape'],
            'stratified': True
        },
        'visualization': {
            'plots': ['cross_validation', 'scatter_actual_predicted', 'histogram', 'qq_plot'],
            'style': 'minimalist'
        }
    }


def list_templates() -> list:
    """List available template types."""
    return ['full', 'minimal', 'kriging', 'validation']
