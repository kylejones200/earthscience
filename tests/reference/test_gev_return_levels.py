"""
GEV return levels compared to scipy.stats.genextreme (R evd equivalent).
"""

import numpy as np
import pytest
from scipy import stats

from earthsciences.statistics.extreme_values import gev_return_level, return_level

# Gumbel (ξ=0): μ − σ ln(−ln(1 − 1/T)); Hosking & Wallis (1997) flood-frequency form.
GUMBEL_MU = 10.0
GUMBEL_SIGMA = 2.0
T_100 = 100.0
P_100 = 1.0 - 1.0 / T_100
GOLDEN_GUMBEL_100Y = GUMBEL_MU - GUMBEL_SIGMA * np.log(-np.log(P_100))

# Fréchet-type positive shape (published-style worked example).
FRECHET_PARAMS = {"location": 50.0, "scale": 12.0, "shape": 0.15}
T_50 = 50.0


class TestGEVReturnLevelReference:
    def test_gumbel_100_year_matches_scipy_ppf(self):
        """100-year level for ξ=0 matches scipy genextreme.ppf."""
        z_scipy = stats.genextreme.ppf(P_100, 0.0, loc=GUMBEL_MU, scale=GUMBEL_SIGMA)
        params = {"location": GUMBEL_MU, "scale": GUMBEL_SIGMA, "shape": 0.0}
        z_lib = return_level(params, T_100)
        assert z_scipy == pytest.approx(GOLDEN_GUMBEL_100Y, rel=1e-12)
        assert z_lib == pytest.approx(GOLDEN_GUMBEL_100Y, rel=1e-12)
        assert gev_return_level(params, T_100) == pytest.approx(GOLDEN_GUMBEL_100Y, rel=1e-12)

    def test_positive_shape_50_year_matches_scipy(self):
        """Non-zero ξ return level matches scipy for the same parameterization."""
        c = -FRECHET_PARAMS["shape"]
        z_scipy = stats.genextreme.ppf(
            1.0 - 1.0 / T_50,
            c,
            loc=FRECHET_PARAMS["location"],
            scale=FRECHET_PARAMS["scale"],
        )
        z_lib = return_level(FRECHET_PARAMS, T_50)
        assert z_lib == pytest.approx(float(z_scipy), rel=1e-10)

    def test_return_level_increases_with_return_period(self):
        """Larger return period ⇒ higher return level (fixed parameters)."""
        params = {"location": GUMBEL_MU, "scale": GUMBEL_SIGMA, "shape": 0.0}
        z10 = return_level(params, 10.0)
        z100 = return_level(params, 100.0)
        z1000 = return_level(params, 1000.0)
        assert z10 < z100 < z1000
