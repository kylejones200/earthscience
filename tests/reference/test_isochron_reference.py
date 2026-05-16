"""
Isochron reference: synthetic Rb-Sr and U-Pb systems with known slope and age.
"""

import numpy as np
import pytest

from earthsciences.geochronology import DECAY_CONSTANTS, isochron_dating
from earthsciences.geochronology.radiometric import concordia_diagram_age

# Rb-Sr: slope m = exp(λt) − 1 at 50 Ma (Steiger & Jäger 1977 λ_Rb).
RB_LAMBDA = DECAY_CONSTANTS["Rb87"]
RB_AGE_YR = 50e6
RB_SLOPE = np.exp(RB_LAMBDA * RB_AGE_YR) - 1
RB_PARENT = np.array([0.10, 0.30, 0.50, 0.70, 0.90])
RB_INITIAL = 0.70400
RB_DAUGHTER = RB_INITIAL + RB_SLOPE * RB_PARENT

# U-Pb concordia: near-concordant zircon-like ratios at ~400 Ma.
U238_L = DECAY_CONSTANTS["U238"]
U235_L = DECAY_CONSTANTS["U235"]
UPB_AGE_YR = 400e6
UPB_RATIO_206_238 = np.exp(U238_L * UPB_AGE_YR) - 1
UPB_RATIO_207_235 = np.exp(U235_L * UPB_AGE_YR) - 1


class TestRbSrIsochronReference:
    def test_synthetic_50_ma_isochron(self):
        ref = np.ones_like(RB_PARENT)
        result = isochron_dating(RB_PARENT, RB_DAUGHTER, ref, RB_LAMBDA)
        assert result["age"] == pytest.approx(RB_AGE_YR, rel=1e-9)
        assert result["slope"] == pytest.approx(RB_SLOPE, rel=1e-9)
        assert result["initial_ratio"] == pytest.approx(RB_INITIAL, rel=1e-9)
        assert result["mswd"] == pytest.approx(0.0, abs=1e-12)

    def test_isochron_linearity_residuals(self):
        """Perfect synthetic data lies on the isochron with zero residuals."""
        ref = np.ones_like(RB_PARENT)
        result = isochron_dating(RB_PARENT, RB_DAUGHTER, ref, RB_LAMBDA)
        predicted = result["slope"] * RB_PARENT + result["intercept"]
        np.testing.assert_allclose(predicted, RB_DAUGHTER, rtol=1e-12)


class TestUPbConcordiaReference:
    def test_concordant_ratios_give_matching_ages(self):
        r206 = np.full(3, UPB_RATIO_206_238)
        r207 = np.full(3, UPB_RATIO_207_235)
        result = concordia_diagram_age(r206, r207)
        assert result["age_206_238"] == pytest.approx(UPB_AGE_YR, rel=1e-6)
        assert result["age_207_235"] == pytest.approx(UPB_AGE_YR, rel=1e-6)
        assert result["concordance"] == pytest.approx(1.0, rel=1e-9)
        assert not result["discordant"]
