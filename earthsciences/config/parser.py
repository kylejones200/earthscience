"""
Configuration file parser and validator.
"""

import yaml
from pathlib import Path
from typing import Union
from .schema import AnalysisConfig


def load_config(config_path: Union[str, Path]) -> AnalysisConfig:
    """
    Load and validate configuration file.
    
    Parameters
    ----------
    config_path : str or Path
        Path to YAML configuration file
        
    Returns
    -------
    AnalysisConfig
        Validated configuration object
        
    Raises
    ------
    FileNotFoundError
        If config file doesn't exist
    ValueError
        If config file is invalid
    yaml.YAMLError
        If YAML syntax is invalid
        
    Examples
    --------
    >>> config = load_config('my_analysis.yaml')
    >>> print(config.project.name)
    'My Analysis'
    """
    config_path = Path(config_path).expanduser()
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML syntax in {config_path}: {e}")
    
    if config_dict is None:
        raise ValueError(f"Empty config file: {config_path}")
    
    try:
        config = AnalysisConfig(**config_dict)
    except Exception as e:
        raise ValueError(f"Invalid configuration: {e}")
    
    return config


def validate_config(config_path: Union[str, Path]) -> tuple[bool, str]:
    """
    Validate configuration file without loading full analysis.
    
    Parameters
    ----------
    config_path : str or Path
        Path to YAML configuration file
        
    Returns
    -------
    is_valid : bool
        True if config is valid
    message : str
        Validation message or error description
        
    Examples
    --------
    >>> valid, msg = validate_config('my_analysis.yaml')
    >>> print(msg)
    '✓ Configuration is valid'
    """
    try:
        config = load_config(config_path)
        return True, "✓ Configuration is valid"
    except FileNotFoundError as e:
        return False, f"✗ File not found: {e}"
    except ValueError as e:
        return False, f"✗ Validation error: {e}"
    except Exception as e:
        return False, f"✗ Unexpected error: {e}"


def config_to_dict(config: AnalysisConfig) -> dict:
    """
    Convert config object to dictionary.
    
    Parameters
    ----------
    config : AnalysisConfig
        Configuration object
        
    Returns
    -------
    dict
        Configuration as dictionary
    """
    return config.dict()


def config_to_yaml(config: AnalysisConfig, output_path: Union[str, Path]):
    """
    Save config object to YAML file.
    
    Parameters
    ----------
    config : AnalysisConfig
        Configuration object
    output_path : str or Path
        Output file path
    """
    output_path = Path(output_path).expanduser()
    config_dict = config_to_dict(config)
    
    with open(output_path, 'w') as f:
        yaml.safe_dump(config_dict, f, default_flow_style=False, sort_keys=False)
