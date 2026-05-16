"""
Variogram model limits: h=0, h→∞, sill, nugget (Matheron standard models).
"""

import numpy as np
import pytest

from earthsciences.spatial.variogram import exponential_model, gaussian_model, spherical_model


class TestSphericalVariogramReference:
    NUGGET, SILL, RANGE = 0.2, 1.0, 5.0

    def test_origin_and_sill_limits(self):
        h = np.array([0.0, self.RANGE, 1e6])
        gamma = spherical_model(h, self.NUGGET, self.SILL, self.RANGE)
        assert gamma[0] == pytest.approx(0.0)
        assert gamma[1] == pytest.approx(self.SILL)
        assert gamma[2] == pytest.approx(self.SILL)

    def test_mid_range_closed_form(self):
        """γ(a/2) = C0 + (C − C0)(1.5·0.5 − 0.5·0.5³) for spherical model."""
        h = np.array([self.RANGE / 2])
        gamma = spherical_model(h, self.NUGGET, self.SILL, self.RANGE)
        expected = self.NUGGET + (self.SILL - self.NUGGET) * (1.5 * 0.5 - 0.5 * 0.5**3)
        assert float(gamma[0]) == pytest.approx(expected, rel=1e-12)


class TestExponentialVariogramReference:
    def test_origin_and_asymptotic_sill(self):
        nugget, sill, a = 0.2, 1.0, 5.0
        h = np.array([0.0, a, 1e6])
        gamma = exponential_model(h, nugget, sill, a)
        assert gamma[0] == pytest.approx(0.0)
        assert gamma[1] == pytest.approx(nugget + (sill - nugget) * (1 - np.exp(-3)), rel=1e-12)
        assert gamma[2] == pytest.approx(sill, rel=1e-6)

    def test_practical_range_definition(self):
        """At h = range, γ ≈ 0.95(sill − nugget) + nugget (3·h/a exponent)."""
        nugget, sill, a = 0.0, 1.0, 10.0
        gamma = float(exponential_model(np.array([a]), nugget, sill, a)[0])
        expected = 1.0 - np.exp(-3)
        assert gamma == pytest.approx(expected, rel=1e-12)


class TestGaussianVariogramReference:
    def test_origin_and_large_lag(self):
        nugget, sill, a = 0.1, 1.0, 8.0
        h = np.array([0.0, 4 * a])
        gamma = gaussian_model(h, nugget, sill, a)
        assert gamma[0] == pytest.approx(0.0)
        assert gamma[1] == pytest.approx(sill, rel=1e-4)
