"""
Tests for directional spherical statistics module
"""

import numpy as np

from earthsciences.directional import spherical


class TestSphericalMean:
    """Test spherical mean calculation."""

    def test_basic_functionality(self):
        """Test spherical mean calculation."""
        # Points near north pole
        theta = np.array([0.1, 0.1, 0.15, 0.1])
        phi = np.array([0, np.pi / 2, np.pi, 3 * np.pi / 2])

        mean_theta, mean_phi = spherical.spherical_mean(theta, phi)

        assert 0 <= mean_theta <= np.pi
        assert 0 <= mean_phi < 2 * np.pi

    def test_antipodal_points(self):
        """Test with antipodal points."""
        theta = np.array([0.1, np.pi - 0.1])
        phi = np.array([0, 0])

        mean_theta, mean_phi = spherical.spherical_mean(theta, phi)

        assert not np.isnan(mean_theta)
        assert not np.isnan(mean_phi)


class TestResultantLength:
    """Test resultant length for spherical data."""

    def test_concentrated_data(self):
        """Test resultant length for concentrated data."""
        theta = np.random.randn(50) * 0.1 + np.pi / 4
        phi = np.random.randn(50) * 0.1 + np.pi / 2

        R = spherical.resultant_length(theta, phi)

        assert 0 <= R <= 1
        assert R > 0.8  # Should be concentrated


class TestSphericalVariance:
    """Test spherical variance."""

    def test_basic_functionality(self):
        """Test spherical variance calculation."""
        np.random.seed(42)
        theta = np.random.rand(30) * np.pi
        phi = np.random.rand(30) * 2 * np.pi

        variance = spherical.spherical_variance(theta, phi)

        assert variance >= 0
        assert variance <= 2


class TestFisherDistribution:
    """Test Fisher distribution on sphere."""

    def test_pdf_calculation(self):
        """Test Fisher distribution PDF."""
        theta = np.pi / 4
        phi = np.pi / 2
        kappa = 10.0

        pdf = spherical.fisher_pdf(
            theta, phi, mean_theta=np.pi / 4, mean_phi=np.pi / 2, kappa=kappa
        )

        assert pdf > 0

    def test_random_sampling(self):
        """Test sampling from Fisher distribution."""
        n_samples = 100
        mean_theta = np.pi / 3
        mean_phi = np.pi
        kappa = 5.0

        theta, phi = spherical.fisher_rvs(mean_theta, mean_phi, kappa, size=n_samples)

        assert len(theta) == n_samples
        assert len(phi) == n_samples
        assert np.all((theta >= 0) & (theta <= np.pi))
        assert np.all((phi >= 0) & (phi < 2 * np.pi))


class TestSphericalDistance:
    """Test great circle distance."""

    def test_same_point(self):
        """Test distance between same point."""
        theta1, phi1 = np.pi / 4, np.pi / 2

        dist = spherical.great_circle_distance(theta1, phi1, theta1, phi1)

        assert np.isclose(dist, 0.0, atol=1e-10)

    def test_antipodal_points(self):
        """Test distance between antipodal points (on unit sphere)."""
        # True antipodal points: theta1=pi/4, phi1=0 and theta2=3*pi/4, phi2=pi
        theta1, phi1 = np.pi / 4, 0
        theta2, phi2 = 3 * np.pi / 4, np.pi

        dist = spherical.great_circle_distance(theta1, phi1, theta2, phi2, degrees=False)

        # Antipodal points on unit sphere are pi apart
        assert np.isclose(dist, np.pi, atol=0.01)


class TestSphericalKDE:
    """Test kernel density estimation on sphere."""

    def test_basic_functionality(self):
        """Test spherical KDE."""
        np.random.seed(42)
        theta = np.random.rand(50) * np.pi
        phi = np.random.rand(50) * 2 * np.pi

        result = spherical.spherical_kde(theta, phi, bandwidth=0.3)

        assert "density" in result or "pdf" in result
