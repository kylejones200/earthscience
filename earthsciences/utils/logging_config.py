"""
Logging configuration helpers for scripts, examples, and library modules.
"""

from __future__ import annotations

import logging
import sys

DEFAULT_FORMAT = "%(message)s"


def setup_logging(
    level: int = logging.INFO,
    *,
    name: str | None = None,
) -> logging.Logger:
    """Configure process-wide logging for CLI entry points and examples."""
    logging.basicConfig(
        level=level,
        format=DEFAULT_FORMAT,
        stream=sys.stdout,
        force=True,
    )
    return logging.getLogger(name or __name__)


def log_section(title: str, *, logger: logging.Logger | None = None) -> None:
    """Log a major section header: === title ===."""
    (logger or logging.getLogger()).info("=== %s ===", title)


def log_step(title: str, *, logger: logging.Logger | None = None) -> None:
    """Log a subsection header: --- title ---."""
    (logger or logging.getLogger()).info("--- %s ---", title)


def log_block(message: str, *, logger: logging.Logger | None = None) -> None:
    """Log a multi-line message (e.g. formatted tables)."""
    (logger or logging.getLogger()).info("%s", message)
