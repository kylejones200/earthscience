"""Tests for geochronology decay and auxiliary dating methods."""

import numpy as np
import pytest

from earthsciences.geochronology import (
    DECAY_CONSTANTS,
    calculate_age,
    cosmogenic_exposure_age,
    fission_track_age,
    radioactive_decay,
    radiocarbon_age,
    radiocarbon_calibration,
    th_u_dating,
)


class TestRadioactiveDecay:
    def test_one_half_life_c14(self):
        remaining = radioactive_decay(100.0, 5730.0, half_life=5730.0)
        assert remaining == pytest.approx(50.0, rel=1e-10)

    def test_decay_constant_matches_half_life(self):
        lam = np.log(2) / 5730.0
        assert radioactive_decay(80.0, 5730.0, decay_constant=lam) == pytest.approx(40.0, rel=1e-10)

    def test_requires_half_life_or_decay_constant(self):
        with pytest.raises(ValueError, match="half_life or decay_constant"):
            radioactive_decay(1.0, 1.0)


class TestCalculateAge:
    def test_u238_pb206_age(self):
        age = calculate_age(100.0, 50.0, DECAY_CONSTANTS["U238"])
        expected = (1 / DECAY_CONSTANTS["U238"]) * np.log(1.5)
        assert age == pytest.approx(expected, rel=1e-10)

    def test_nonpositive_parent_raises(self):
        with pytest.raises(ValueError, match="N_parent"):
            calculate_age(0.0, 10.0, DECAY_CONSTANTS["U238"])


class TestRadiocarbon:
    def test_half_modern_carbon_age(self):
        age, unc = radiocarbon_age(0.5, N_modern=1.0)
        assert age == pytest.approx(5730.0, rel=1e-6)
        assert unc > 0

    def test_calibration_returns_interval(self):
        result = radiocarbon_calibration(5000.0, 50.0)
        assert "median_cal_age" in result
        lo, hi = result["range_1sigma"]
        assert lo <= result["median_cal_age"] <= hi


class TestOtherDatingMethods:
    def test_fission_track_age_positive(self):
        age, unc = fission_track_age(1.5e6, 3.0e6, 28.0e6, 350.0)
        assert age > 0
        assert unc >= 0

    def test_cosmogenic_exposure_age(self):
        age = cosmogenic_exposure_age(1.0e6, 10.0)
        assert age == pytest.approx(1.0e5, rel=1e-10)

    def test_th_u_dating(self):
        age = th_u_dating(0.5)
        assert age > 0
