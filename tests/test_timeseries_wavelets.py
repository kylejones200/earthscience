"""
Tests for timeseries wavelet analysis module
"""

import pytest
import numpy as np
from earthsciences.timeseries import wavelets


class TestCWT:
    """Test continuous wavelet transform."""
    
    def test_basic_functionality(self):
        """Test basic CWT."""
        np.random.seed(42)
        signal = np.sin(2 * np.pi * 10 * np.linspace(0, 1, 256))
        
        result = wavelets.cwt(signal, wavelet='morlet')
        
        assert 'coefficients' in result
        assert 'scales' in result or 'frequencies' in result
        assert result['coefficients'].ndim == 2
    
    def test_different_wavelets(self):
        """Test different wavelet types."""
        signal = np.random.randn(128)
        
        # Only test wavelets that are available in pywavelets
        for wavelet_type in ['morl', 'mexh']:
            result = wavelets.cwt(signal, wavelet=wavelet_type)
            assert result['coefficients'].shape[1] == len(signal)


class TestDWT:
    """Test discrete wavelet transform."""
    
    def test_basic_functionality(self):
        """Test basic DWT."""
        signal = np.random.randn(128)
        
        result = wavelets.dwt(signal, wavelet='db4')
        
        assert 'approximation' in result or 'cA' in result
        assert 'detail' in result or 'cD' in result
    
    def test_reconstruction(self):
        """Test that IDWT reconstructs signal."""
        signal = np.random.randn(128)
        
        result = wavelets.dwt(signal, wavelet='db4')
        reconstructed = wavelets.idwt(result)
        
        np.testing.assert_array_almost_equal(signal, reconstructed, decimal=10)


class TestWaveletCoherence:
    """Test wavelet coherence."""
    
    def test_basic_functionality(self):
        """Test wavelet coherence between two signals."""
        np.random.seed(42)
        t = np.linspace(0, 10, 500)
        x = np.sin(2 * np.pi * 1 * t)
        y = np.sin(2 * np.pi * 1 * t + np.pi/4)
        
        result = wavelets.wavelet_coherence(x, y)
        
        assert 'coherence' in result
        assert 'phase' in result or 'angle' in result
        assert np.all((result['coherence'] >= 0) & (result['coherence'] <= 1))


class TestWaveletPower:
    """Test wavelet power spectrum."""
    
    def test_power_calculation(self):
        """Test wavelet power spectrum."""
        np.random.seed(42)
        signal = np.sin(2 * np.pi * 5 * np.linspace(0, 10, 500))
        
        result = wavelets.wavelet_power(signal)
        
        assert 'power' in result
        assert 'scales' in result or 'frequencies' in result
        assert result['power'].ndim == 2


class TestWaveletDenoising:
    """Test wavelet denoising."""
    
    def test_noise_reduction(self):
        """Test wavelet-based denoising."""
        np.random.seed(42)
        signal = np.sin(2 * np.pi * 2 * np.linspace(0, 5, 256))
        noisy = signal + np.random.randn(256) * 0.5
        
        denoised = wavelets.wavelet_denoise(noisy, wavelet='db4')
        
        assert len(denoised) == len(noisy)
        
        # Should be closer to original
        mse_noisy = np.mean((noisy - signal)**2)
        mse_denoised = np.mean((denoised - signal)**2)
        assert mse_denoised < mse_noisy
