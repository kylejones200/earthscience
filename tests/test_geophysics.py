"""Tests for geophysical corrections and forward models."""

import numpy as np
import pytest

from earthsciences.geophysics import (
    bouguer_correction,
    free_air_correction,
    magnetic_anomaly_sphere,
    nmo_correction,
    reduction_to_pole,
)


class TestGravityCorrections:
    def test_free_air_correction_gradient(self):
        g_obs = np.array([980_000.0, 980_000.0])
        elev = np.array([0.0, 100.0])
        corrected = free_air_correction(g_obs, elev)
        assert corrected[1] < corrected[0]
        assert corrected[1] - corrected[0] == pytest.approx(-0.3086 * 100.0, rel=1e-4)

    def test_bouguer_correction_keys(self):
        g_obs = np.array([980_000.0, 980_100.0])
        elev = np.array([100.0, 200.0])
        lat = np.array([45.0, 45.0])
        result = bouguer_correction(g_obs, elev, lat, density=2670)
        assert "bouguer_anomaly" in result
        assert len(result["bouguer_anomaly"]) == 2
        assert len(result["free_air_correction"]) == 2


class TestMagnetics:
    def test_sphere_anomaly_finite_and_nonzero(self):
        x = np.linspace(-500, 500, 11)
        y = np.zeros_like(x)
        z = np.zeros_like(x)
        anomaly = magnetic_anomaly_sphere(
            x,
            y,
            z,
            x0=0,
            y0=0,
            z0=200,
            radius=100,
            susceptibility=0.05,
            inclination=60,
            declination=0,
        )
        assert np.all(np.isfinite(anomaly))
        assert np.max(np.abs(anomaly)) > 0

    def test_reduction_to_pole_changes_amplitude(self):
        field = np.array([100.0, -50.0, 30.0])
        rtp = reduction_to_pole(field, inclination=60.0, declination=10.0)
        assert len(rtp) == len(field)
        assert not np.allclose(rtp, field)


class TestSeismic:
    def test_nmo_correction_reduces_to_zero_offset_time(self):
        velocity = 2000.0
        offsets = np.array([0.0, 500.0, 1000.0])
        traveltimes = np.sqrt(1.0**2 + (offsets / velocity) ** 2)
        t0 = nmo_correction(traveltimes, offsets, velocity)
        assert np.all(t0 <= traveltimes)
        assert t0[0] == pytest.approx(1.0, rel=1e-6)
