"""
Configuration system for earthsciences package.

Enables config-driven workflows using YAML files.
"""

from .parser import load_config, AnalysisConfig
from .templates import generate_template

__all__ = ['load_config', 'AnalysisConfig', 'generate_template']
