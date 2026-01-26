"""
Tests for timeseries signals module
"""

import pytest
import numpy as np
from earthsciences.timeseries import signals


class TestGenerateTestSignal:
    """Test generate_test_signal function."""
    
    def test_sine_signal(self):
        """Test sine signal generation."""
        t, sig = signals.generate_test_signal('sine', duration=1.0, sampling_rate=1000)
        
        assert len(t) == 1000
        assert len(sig) == 1000
        assert -1.5 <= sig.min() <= sig.max() <= 1.5
    
    def test_mixed_signal(self):
        """Test mixed signal generation."""
        t, sig = signals.generate_test_signal('mixed', duration=1.0, sampling_rate=1000)
        
        assert len(t) == 1000
        assert len(sig) == 1000
    
    def test_chirp_signal(self):
        """Test chirp signal generation."""
        t, sig = signals.generate_test_signal('chirp', duration=1.0, sampling_rate=1000)
        
        assert len(t) == 1000
        assert len(sig) == 1000
    
    def test_pulse_signal(self):
        """Test pulse signal generation."""
        t, sig = signals.generate_test_signal('pulse', duration=1.0, sampling_rate=1000)
        
        assert len(t) == 1000
        assert len(sig) == 1000
    
    def test_noise_signal(self):
        """Test noise signal generation."""
        t, sig = signals.generate_test_signal('noise', duration=1.0, sampling_rate=1000)
        
        assert len(t) == 1000
        assert len(sig) == 1000
    
    def test_with_noise(self):
        """Test adding noise to signal."""
        t1, sig1 = signals.generate_test_signal('sine', noise_level=0.0, seed=42)
        t2, sig2 = signals.generate_test_signal('sine', noise_level=0.5, seed=42)
        
        # Signal with noise should have higher variance
        assert np.var(sig2) > np.var(sig1)
    
    def test_seed_reproducibility(self):
        """Test that seed produces reproducible results."""
        t1, sig1 = signals.generate_test_signal('sine', noise_level=0.1, seed=42)
        t2, sig2 = signals.generate_test_signal('sine', noise_level=0.1, seed=42)
        
        np.testing.assert_array_equal(sig1, sig2)
    
    def test_invalid_signal_type(self):
        """Test that invalid signal type raises error."""
        with pytest.raises(ValueError, match="Unknown signal type"):
            signals.generate_test_signal('invalid')


class TestDetrend:
    """Test detrend function."""
    
    def test_linear_detrend(self):
        """Test linear detrending."""
        t = np.linspace(0, 1, 100)
        trend = 2*t + 5
        signal = trend + np.sin(2*np.pi*5*t)
        
        detrended = signals.detrend(signal, method='linear')
        
        # After detrending, should be mostly sinusoidal
        assert np.abs(np.mean(detrended)) < 1.0
    
    def test_constant_detrend(self):
        """Test constant detrending (mean removal)."""
        signal = np.sin(2*np.pi*5*np.linspace(0, 1, 100)) + 10
        
        detrended = signals.detrend(signal, method='constant')
        
        assert np.abs(np.mean(detrended)) < 1e-10


class TestNormalize:
    """Test normalize function."""
    
    def test_zscore_normalization(self):
        """Test z-score normalization."""
        data = np.array([1, 2, 3, 4, 5])
        normalized = signals.normalize(data, method='zscore')
        
        assert np.abs(np.mean(normalized)) < 1e-10
        assert np.abs(np.std(normalized) - 1.0) < 1e-10
    
    def test_minmax_normalization(self):
        """Test min-max normalization."""
        data = np.array([1, 2, 3, 4, 5])
        normalized = signals.normalize(data, method='minmax')
        
        assert np.isclose(normalized.min(), 0.0)
        assert np.isclose(normalized.max(), 1.0)
    
    def test_maxabs_normalization(self):
        """Test max absolute normalization."""
        data = np.array([-5, -2, 0, 2, 5])
        normalized = signals.normalize(data, method='maxabs')
        
        assert np.isclose(np.max(np.abs(normalized)), 1.0)
    
    def test_invalid_method(self):
        """Test invalid normalization method."""
        data = np.array([1, 2, 3, 4, 5])
        
        with pytest.raises(ValueError, match="Unknown method"):
            signals.normalize(data, method='invalid')


class TestResample:
    """Test resample function."""
    
    def test_downsample(self):
        """Test downsampling."""
        t, sig = signals.generate_test_signal('sine', sampling_rate=1000, seed=42)
        
        resampled = signals.resample(sig, original_rate=1000, target_rate=100)
        
        # Should have 1/10th the samples
        assert len(resampled) == len(sig) // 10
    
    def test_upsample(self):
        """Test upsampling."""
        t, sig = signals.generate_test_signal('sine', sampling_rate=100, seed=42)
        
        resampled = signals.resample(sig, original_rate=100, target_rate=1000)
        
        # Should have 10x the samples
        assert len(resampled) == len(sig) * 10


class TestAutocorrelation:
    """Test autocorrelation function."""
    
    def test_basic_functionality(self, sample_time_series):
        """Test autocorrelation basic functionality."""
        t, sig = sample_time_series
        lags, acf = signals.autocorrelation(sig, max_lag=100)
        
        assert len(lags) == 101
        assert len(acf) == 101
    
    def test_zero_lag_is_one(self, sample_time_series):
        """Test that autocorrelation at lag 0 is 1."""
        t, sig = sample_time_series
        lags, acf = signals.autocorrelation(sig)
        
        assert np.isclose(acf[0], 1.0)
    
    def test_periodic_signal(self):
        """Test autocorrelation of periodic signal."""
        t = np.linspace(0, 1, 1000)
        sig = np.sin(2*np.pi*10*t)  # 10 Hz sine wave
        
        lags, acf = signals.autocorrelation(sig, max_lag=200)
        
        # Should show periodicity
        assert acf[0] > 0.9  # High at zero lag


class TestCrosscorrelation:
    """Test crosscorrelation function."""
    
    def test_basic_functionality(self):
        """Test cross-correlation basic functionality."""
        t, x = signals.generate_test_signal('sine', seed=42)
        y = np.roll(x, 10)  # Shifted version
        
        lags, ccf = signals.crosscorrelation(x, y, max_lag=50)
        
        assert len(lags) > 0
        assert len(ccf) == len(lags)
    
    def test_identical_signals(self):
        """Test cross-correlation of identical signals."""
        t, x = signals.generate_test_signal('sine', seed=42)
        
        lags, ccf = signals.crosscorrelation(x, x, max_lag=10)
        
        # Peak should be at zero lag
        peak_idx = np.argmax(ccf)
        assert lags[peak_idx] == 0


class TestHilbertTransform:
    """Test hilbert_transform function."""
    
    def test_returns_three_arrays(self, sample_time_series):
        """Test that hilbert_transform returns three arrays."""
        t, sig = sample_time_series
        amplitude, phase, frequency = signals.hilbert_transform(sig)
        
        assert len(amplitude) == len(sig)
        assert len(phase) == len(sig)
        assert len(frequency) == len(sig)
    
    def test_amplitude_positive(self, sample_time_series):
        """Test that amplitude is positive."""
        t, sig = sample_time_series
        amplitude, phase, frequency = signals.hilbert_transform(sig)
        
        assert np.all(amplitude >= 0)


class TestEnvelope:
    """Test envelope function."""
    
    def test_envelope_bounds_signal(self):
        """Test that envelope bounds the signal."""
        t = np.linspace(0, 1, 1000)
        carrier = np.sin(2*np.pi*50*t)
        modulation = 1 + 0.5*np.sin(2*np.pi*5*t)
        sig = modulation * carrier
        
        env = signals.envelope(sig)
        
        # Envelope should be >= absolute value of signal
        assert np.all(env >= np.abs(sig) - 0.1)  # Small tolerance
