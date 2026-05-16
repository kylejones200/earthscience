"""Tests for logging helpers."""

import logging

from earthsciences.utils.logging_config import log_section, log_step, setup_logging


def test_setup_logging_default_level():
    setup_logging(level=logging.WARNING)
    assert logging.getLogger().level == logging.WARNING


def test_log_section_emits(caplog):
    logger = logging.getLogger("earthsciences.test")
    with caplog.at_level(logging.INFO, logger="earthsciences.test"):
        log_section("Test section", logger=logger)
    assert "Test section" in caplog.text


def test_log_step_emits(caplog):
    logger = logging.getLogger("earthsciences.test")
    with caplog.at_level(logging.INFO, logger="earthsciences.test"):
        log_step("Step one", logger=logger)
    assert "Step one" in caplog.text
