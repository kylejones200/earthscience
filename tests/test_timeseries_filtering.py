"""
Tests for timeseries filtering module
"""

import numpy as np

from earthsciences.timeseries import filtering


class TestButterworthFilter:
    """Test Butterworth filter."""

    def test_lowpass(self):
        """Test lowpass filter."""
        np.random.seed(42)
        t = np.linspace(0, 1, 500)
        signal = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 50 * t)

        filtered = filtering.butterworth_filter(signal, cutoff=10, fs=500, filter_type="lowpass")

        assert len(filtered) == len(signal)
        assert not np.array_equal(filtered, signal)

    def test_highpass(self):
        """Test highpass filter."""
        np.random.seed(42)
        signal = np.sin(2 * np.pi * 5 * np.linspace(0, 1, 500))

        filtered = filtering.butterworth_filter(signal, cutoff=10, fs=500, filter_type="highpass")

        assert len(filtered) == len(signal)

    def test_bandpass(self):
        """Test bandpass filter."""
        signal = np.random.randn(1000)

        filtered = filtering.butterworth_filter(
            signal, cutoff=[5, 20], fs=100, filter_type="bandpass"
        )

        assert len(filtered) == len(signal)

    def test_bandstop(self):
        """Test bandstop filter."""
        signal = np.random.randn(500)

        filtered = filtering.butterworth_filter(
            signal, cutoff=[45, 55], fs=1000, filter_type="bandstop"
        )

        assert len(filtered) == len(signal)


class TestMovingAverage:
    """Test moving average filter."""

    def test_basic_functionality(self):
        """Test basic moving average."""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        result = filtering.moving_average(data, window=3)

        assert len(result) > 0
        assert len(result) <= len(data)

    def test_smoothing_effect(self):
        """Test that moving average smooths data."""
        np.random.seed(42)
        data = np.sin(np.linspace(0, 10, 100)) + np.random.randn(100) * 0.5

        smoothed = filtering.moving_average(data, window=5)

        # Smoothed should have less variation
        assert np.std(np.diff(smoothed)) < np.std(np.diff(data))


class TestMedianFilter:
    """Test median filter."""

    def test_removes_outliers(self):
        """Test that median filter removes outliers."""
        data = np.ones(50)
        data[25] = 100  # Outlier

        filtered = filtering.median_filter(data, window=5)

        # Outlier should be reduced
        assert filtered[25] < 50


class TestSavitzkyGolay:
    """Test Savitzky-Golay filter."""

    def test_basic_functionality(self):
        """Test Savitzky-Golay smoothing."""
        np.random.seed(42)
        x = np.linspace(0, 10, 100)
        y = np.sin(x) + np.random.randn(100) * 0.1

        smoothed = filtering.savitzky_golay(y, window=11, order=3)

        assert len(smoothed) == len(y)
        # Should be smoother
        assert np.std(smoothed) < np.std(y)

    def test_polynomial_preservation(self):
        """Test that it preserves polynomial trends."""
        x = np.linspace(0, 10, 50)
        y = 2 * x**2 + 3 * x + 1

        smoothed = filtering.savitzky_golay(y, window=11, order=2)

        # Should preserve quadratic
        np.testing.assert_array_almost_equal(y, smoothed, decimal=5)


class TestKalmanFilter:
    """Test Kalman filter."""

    def test_basic_functionality(self):
        """Test basic Kalman filtering."""
        np.random.seed(42)
        true_values = np.cumsum(np.random.randn(100) * 0.1)
        measurements = true_values + np.random.randn(100) * 1.0

        filtered = filtering.kalman_filter(measurements)

        assert len(filtered) == len(measurements)

        # Filtered should be smoother than measurements
        assert np.std(np.diff(filtered)) < np.std(np.diff(measurements))


class TestWienerFilter:
    """Test Wiener filter."""

    def test_noise_reduction(self):
        """Test Wiener filter noise reduction."""
        np.random.seed(42)
        signal = np.sin(2 * np.pi * 5 * np.linspace(0, 1, 200))
        noisy = signal + np.random.randn(200) * 0.5

        filtered = filtering.wiener_filter(noisy)

        assert len(filtered) == len(noisy)

        # Should be closer to original signal
        mse_noisy = np.mean((noisy - signal) ** 2)
        mse_filtered = np.mean((filtered - signal) ** 2)
        assert mse_filtered < mse_noisy
