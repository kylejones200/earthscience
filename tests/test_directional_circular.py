"""
Tests for circular statistics module
"""

import numpy as np

from earthsciences.directional import circular


class TestCircularMean:
    """Test circular_mean function."""

    def test_north_direction(self):
        """Test mean of angles around north (0°)."""
        angles = np.array([350, 0, 10])
        mean = circular.circular_mean(angles, degrees=True)

        # Should be close to 0
        assert abs(mean) < 5 or abs(mean - 360) < 5

    def test_east_direction(self):
        """Test mean around east (90°)."""
        angles = np.array([85, 90, 95])
        mean = circular.circular_mean(angles, degrees=True)

        assert 85 <= mean <= 95

    def test_radians(self):
        """Test with radians."""
        angles = np.array([0, np.pi / 4, np.pi / 2])
        mean = circular.circular_mean(angles, degrees=False)

        assert 0 <= mean <= np.pi


class TestCircularVariance:
    """Test circular_variance function."""

    def test_concentrated_data(self):
        """Test that concentrated data has low variance."""
        angles = np.array([0, 1, 2, 358, 359])
        var = circular.circular_variance(angles, degrees=True)

        # Should be close to 0
        assert var < 0.1

    def test_uniform_data(self):
        """Test that uniform data has high variance."""
        angles = np.linspace(0, 350, 36)
        var = circular.circular_variance(angles, degrees=True)

        # Should be close to 1
        assert var > 0.9


class TestCircularStd:
    """Test circular_std function."""

    def test_returns_positive(self, sample_angles):
        """Test that std is positive."""
        std = circular.circular_std(sample_angles, degrees=True)
        assert std >= 0


class TestRayleighTest:
    """Test rayleigh_test function."""

    def test_returns_dict(self, sample_angles):
        """Test that rayleigh_test returns dict with expected keys."""
        result = circular.rayleigh_test(sample_angles, degrees=True)

        expected_keys = [
            "statistic",
            "p_value",
            "mean_resultant_length",
            "mean_direction",
            "significant",
        ]

        for key in expected_keys:
            assert key in result

    def test_concentrated_data(self):
        """Test with concentrated data (should reject uniformity)."""
        # Von Mises distribution, highly concentrated
        angles = np.random.vonmises(0, 10, 100)
        result = circular.rayleigh_test(angles, degrees=False)

        # Should reject null hypothesis of uniformity
        assert result["p_value"] < 0.05
        assert result["significant"] == True

    def test_p_value_range(self, sample_angles):
        """Test that p-value is in valid range."""
        result = circular.rayleigh_test(sample_angles, degrees=True)
        assert 0 <= result["p_value"] <= 1


class TestVonMisesFit:
    """Test von_mises_fit function."""

    def test_basic_functionality(self, sample_angles):
        """Test von Mises fitting returns expected keys."""
        result = circular.von_mises_fit(sample_angles, degrees=True)

        assert "mu" in result
        assert "kappa" in result
        assert "mean_resultant_length" in result

    def test_mu_range_degrees(self, sample_angles):
        """Test that mu is in valid range for degrees."""
        result = circular.von_mises_fit(sample_angles, degrees=True)
        assert 0 <= result["mu"] < 360

    def test_kappa_positive(self, sample_angles):
        """Test that kappa is positive."""
        result = circular.von_mises_fit(sample_angles, degrees=True)
        assert result["kappa"] >= 0


class TestRoseDiagramData:
    """Test rose_diagram_data function."""

    def test_returns_dict(self, sample_angles):
        """Test that rose_diagram_data returns dict."""
        result = circular.rose_diagram_data(sample_angles, n_bins=16, degrees=True)

        expected_keys = [
            "bin_centers_rad",
            "bin_centers_deg",
            "bin_width",
            "counts",
            "frequencies",
            "n_bins",
        ]

        for key in expected_keys:
            assert key in result

    def test_bin_count(self, sample_angles):
        """Test that correct number of bins is created."""
        n_bins = 16
        result = circular.rose_diagram_data(sample_angles, n_bins=n_bins, degrees=True)

        assert len(result["counts"]) == n_bins
        assert len(result["frequencies"]) == n_bins

    def test_frequencies_sum_to_one(self, sample_angles):
        """Test that frequencies sum to 1."""
        result = circular.rose_diagram_data(sample_angles, degrees=True)

        assert np.isclose(np.sum(result["frequencies"]), 1.0)


class TestAngularDistance:
    """Test angular_distance function."""

    def test_short_distance(self):
        """Test that angular distance takes shorter path."""
        dist = circular.angular_distance(10, 350, degrees=True)

        # Should be 20°, not 340°
        assert np.isclose(dist, 20.0)

    def test_opposite_directions(self):
        """Test opposite directions."""
        dist = circular.angular_distance(0, 180, degrees=True)

        assert np.isclose(dist, 180.0)

    def test_same_angle(self):
        """Test that same angle gives zero distance."""
        dist = circular.angular_distance(45, 45, degrees=True)

        assert np.isclose(dist, 0.0)


class TestWatsonU2Test:
    """Test watson_u2_test function."""

    def test_basic_functionality(self):
        """Test Watson U² test returns proper structure."""
        sample1 = np.random.vonmises(0, 2, 50)
        sample2 = np.random.vonmises(0, 2, 50)

        result = circular.watson_u2_test(sample1, sample2, degrees=False)

        assert "statistic" in result
        assert "p_value" in result
        assert "n1" in result
        assert "n2" in result
        assert result["n1"] == 50
        assert result["n2"] == 50

    def test_identical_samples(self):
        """Test with identical samples should give high p-value."""
        np.random.seed(42)
        sample = np.random.vonmises(0, 2, 50)

        result = circular.watson_u2_test(sample, sample, degrees=False)

        # Identical samples should have U² ≈ 0 and p-value ≈ 1
        assert result["statistic"] < 0.01
        assert result["p_value"] > 0.9

    def test_different_distributions(self):
        """Test with very different distributions should give low p-value."""
        np.random.seed(42)
        sample1 = np.random.vonmises(0, 5, 100)  # Concentrated at 0
        sample2 = np.random.vonmises(np.pi, 5, 100)  # Concentrated at π

        result = circular.watson_u2_test(sample1, sample2, degrees=False)

        # Very different samples should have significant U² and low p-value
        assert result["p_value"] < 0.05

    def test_degrees_input(self):
        """Test that degrees input works correctly."""
        sample1 = np.array([0, 90, 180, 270])
        sample2 = np.array([45, 135, 225, 315])

        result = circular.watson_u2_test(sample1, sample2, degrees=True)

        assert "statistic" in result
        assert "p_value" in result
