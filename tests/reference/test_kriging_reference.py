"""
Ordinary kriging on a symmetric 3-point equilateral layout with hand-solved weights.
"""

import numpy as np
import pytest
from scipy.linalg import solve
from scipy.spatial.distance import cdist

from earthsciences.spatial.kriging import ordinary_kriging
from earthsciences.spatial.variogram import spherical_model

# Equilateral triangle, side 1; spherical γ with nugget=0, sill=1, range=2.
X = np.array([0.0, 1.0, 0.5])
Y = np.array([0.0, 0.0, np.sqrt(3) / 2])
Z = np.array([10.0, 20.0, 30.0])
CENTROID_X = np.array([[0.5]])
CENTROID_Y = np.array([[np.sqrt(3) / 6]])
GOLDEN_WEIGHTS = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])
GOLDEN_PREDICTION = 20.0


def _spherical_covariance(distances: np.ndarray) -> np.ndarray:
    return 1.0 - spherical_model(np.asarray(distances), 0.0, 1.0, 2.0)


def _hand_ordinary_kriging_weights() -> np.ndarray:
    pts = np.column_stack([X, Y])
    k_data = _spherical_covariance(cdist(pts, pts))
    k_sys = np.ones((4, 4))
    k_sys[:3, :3] = k_data
    k_sys[3, 3] = 0.0
    cent = np.array([[0.5, np.sqrt(3) / 6]])
    k_target = _spherical_covariance(cdist(cent, pts)[0])
    rhs = np.ones(4)
    rhs[:3] = k_target
    solution = solve(k_sys, rhs)
    return solution[:3]


class TestOrdinaryKrigingReference:
    def test_symmetric_weights_sum_to_one(self):
        weights = _hand_ordinary_kriging_weights()
        np.testing.assert_allclose(weights, GOLDEN_WEIGHTS, rtol=1e-10)
        assert weights.sum() == pytest.approx(1.0)

    def test_centroid_prediction_matches_hand_solution(self):
        def variogram_func(h):
            return spherical_model(np.asarray(h), 0.0, 1.0, 2.0)

        pred = ordinary_kriging(X, Y, Z, CENTROID_X, CENTROID_Y, variogram_func)
        assert float(pred.ravel()[0]) == pytest.approx(GOLDEN_PREDICTION, rel=1e-10)
        weights = _hand_ordinary_kriging_weights()
        assert float(np.dot(weights, Z)) == pytest.approx(GOLDEN_PREDICTION, rel=1e-10)
