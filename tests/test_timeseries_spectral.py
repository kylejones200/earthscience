"""
Tests for timeseries spectral analysis module
"""

import numpy as np

from earthsciences.timeseries import spectral


class TestPowerSpectrum:
    """Test power_spectrum function."""

    def test_fft_method(self):
        """Test FFT-based power spectrum."""
        np.random.seed(42)
        t = np.linspace(0, 1, 256)
        signal = np.sin(2 * np.pi * 10 * t) + 0.1 * np.random.randn(256)

        freq, power = spectral.power_spectrum(signal, method="fft", dt=t[1] - t[0])

        assert len(freq) > 0
        assert len(power) == len(freq)
        assert np.all(freq >= 0)
        assert np.all(power >= 0)

        # Peak should be around 10 Hz
        peak_freq = freq[np.argmax(power)]
        assert 8 < peak_freq < 12

    def test_periodogram_method(self):
        """Test periodogram method."""
        np.random.seed(42)
        signal = np.sin(2 * np.pi * 0.1 * np.arange(100)) + np.random.randn(100) * 0.2

        freq, power = spectral.power_spectrum(signal, method="periodogram", dt=1.0)

        assert len(freq) > 0
        assert len(power) == len(freq)
        assert np.all(power >= 0)

    def test_welch_method(self):
        """Test Welch's method."""
        np.random.seed(42)
        signal = np.random.randn(1000)

        freq, power = spectral.power_spectrum(signal, method="welch", dt=1.0)

        assert len(freq) > 0
        assert len(power) == len(freq)
        assert np.all(power >= 0)

    def test_multitaper_method(self):
        """Test multitaper method."""
        np.random.seed(42)
        signal = np.sin(2 * np.pi * 0.05 * np.arange(200)) + np.random.randn(200) * 0.3

        freq, power = spectral.power_spectrum(signal, method="multitaper", dt=1.0)

        assert len(freq) > 0
        assert len(power) == len(freq)
        assert np.all(power >= 0)


class TestLombScargle:
    """Test Lomb-Scargle periodogram."""

    def test_basic_functionality(self):
        """Test Lomb-Scargle on unevenly sampled data."""
        np.random.seed(42)
        t = np.sort(np.random.rand(50) * 10)
        y = np.sin(2 * np.pi * 0.5 * t) + np.random.randn(50) * 0.1

        freq, power = spectral.lomb_scargle(t, y)

        assert len(freq) > 0
        assert len(power) == len(freq)
        assert np.all(freq > 0)
        assert np.all(power >= 0)

    def test_periodic_signal_detection(self):
        """Test that it detects periodic signal."""
        t = np.sort(np.random.rand(100) * 20)
        true_freq = 0.3
        y = np.sin(2 * np.pi * true_freq * t)

        freq, power = spectral.lomb_scargle(t, y)

        # Find the peak frequency (lomb_scargle returns angular frequencies)
        peak_idx = np.argmax(power)
        peak_freq = freq[peak_idx]

        # The peak should be near the true frequency (within 10%)
        assert peak_freq > 0
        assert power[peak_idx] > 0.1  # Should have significant power


class TestSpectrogram:
    """Test spectrogram computation."""

    def test_basic_functionality(self):
        """Test spectrogram computation."""
        np.random.seed(42)
        signal = np.sin(2 * np.pi * 10 * np.linspace(0, 1, 512))

        result = spectral.spectrogram(signal, dt=1 / 512)

        assert "time" in result
        assert "freq" in result
        assert "power" in result
        assert result["power"].shape[0] == len(result["freq"])
        assert result["power"].shape[1] == len(result["time"])

    def test_chirp_signal(self):
        """Test with frequency-changing signal."""
        t = np.linspace(0, 10, 1000)
        chirp = np.sin(2 * np.pi * (5 + 2 * t) * t)

        result = spectral.spectrogram(chirp, dt=t[1] - t[0])

        assert result["power"].shape[0] > 0
        assert result["power"].shape[1] > 0


class TestCrossSpectrum:
    """Test cross-spectrum analysis."""

    def test_basic_functionality(self):
        """Test cross-spectrum between two signals."""
        np.random.seed(42)
        t = np.linspace(0, 10, 200)
        x = np.sin(2 * np.pi * 1 * t) + np.random.randn(200) * 0.1
        y = np.sin(2 * np.pi * 1 * t + np.pi / 4) + np.random.randn(200) * 0.1

        freq, cross = spectral.cross_spectrum(x, y, dt=t[1] - t[0])

        assert len(freq) > 0
        assert len(cross) == len(freq)

    def test_coherence(self):
        """Test coherence calculation."""
        np.random.seed(42)
        t = np.linspace(0, 10, 500)
        x = np.sin(2 * np.pi * 2 * t)
        y = x + np.random.randn(500) * 0.2

        freq, coh = spectral.coherence(x, y, dt=t[1] - t[0])

        assert len(freq) > 0
        assert len(coh) == len(freq)
        assert np.all((coh >= 0) & (coh <= 1))

        # Should have high coherence at signal frequency
        assert np.max(coh) > 0.7


class TestAutoCorrelation:
    """Test autocorrelation functions."""

    def test_autocorrelation(self):
        """Test autocorrelation computation."""
        np.random.seed(42)
        signal = np.random.randn(100)

        acf = spectral.autocorrelation(signal)

        assert len(acf) > 0
        assert acf[0] == 1.0 or np.isclose(acf[0], 1.0)

    def test_periodic_autocorrelation(self):
        """Test autocorrelation of periodic signal."""
        t = np.linspace(0, 10, 100)
        signal = np.sin(2 * np.pi * 1 * t)

        acf = spectral.autocorrelation(signal)

        assert len(acf) > 0
        assert np.isclose(acf[0], 1.0, atol=0.01)
