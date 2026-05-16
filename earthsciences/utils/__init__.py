"""
Utility functions and helpers

Common utilities used across the library.
"""

from .logging_config import log_block, log_section, log_step, setup_logging
from .plot_style import clean_plot_style, setup_figure, use_earthsciences_style
from .validation import (
    validate_angles,
    validate_array,
    validate_coordinates,
    validate_same_length,
)

__all__ = [
    "validate_array",
    "validate_same_length",
    "validate_coordinates",
    "validate_angles",
    "setup_logging",
    "log_section",
    "log_step",
    "log_block",
    "use_earthsciences_style",
    "clean_plot_style",
    "setup_figure",
]
