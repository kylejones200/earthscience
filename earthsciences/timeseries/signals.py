"""
Signal generation and utilities

Helper functions for creating test signals and signal processing.
"""

import numpy as np
from scipy import signal
from typing import Tuple, Optional


def generate_test_signal(signal_type: str = 'mixed',
                        duration: float = 1.0,
                        sampling_rate: float = 1000.0,
                        noise_level: float = 0.0,
                        seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate test signals for demonstration and testing.
    
    Parameters
    ----------
    signal_type : str, optional
        Signal type: 'sine', 'mixed', 'chirp', 'pulse', 'noise' (default='mixed')
    duration : float, optional
        Signal duration in seconds (default=1.0)
    sampling_rate : float, optional
        Sampling rate in Hz (default=1000.0)
    noise_level : float, optional
        Noise standard deviation (default=0.0)
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    t : ndarray
        Time array
    signal : ndarray
        Generated signal
        
    Examples
    --------
    >>> t, sig = generate_test_signal('chirp', duration=2.0, 
    ...                               sampling_rate=1000, noise_level=0.1)
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(t, sig)
    >>> plt.show()
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_samples = int(duration * sampling_rate)
    t = np.linspace(0, duration, n_samples)
    
    match signal_type:
        case 'sine':
            sig = np.sin(2 * np.pi * 5 * t)
        
        case 'mixed':
            sig = (np.sin(2 * np.pi * 5 * t) +
                   0.5 * np.sin(2 * np.pi * 10 * t) +
                   0.3 * np.sin(2 * np.pi * 20 * t))
        
        case 'chirp':
            f0, f1 = 1, 50
            sig = signal.chirp(t, f0, duration, f1)
        
        case 'pulse':
            sig = signal.square(2 * np.pi * 5 * t)
        
        case 'noise':
            sig = np.random.randn(n_samples)
        
        case _:
            raise ValueError(f"Unknown signal type: {signal_type}")
    
    if noise_level > 0:
        sig = sig + np.random.randn(n_samples) * noise_level
    
    return t, sig


def detrend(data: np.ndarray, method: str = 'linear') -> np.ndarray:
    """
    Remove trend from signal.
    
    Parameters
    ----------
    data : array_like
        Input signal
    method : str, optional
        Detrending method: 'linear', 'constant' (default='linear')
        
    Returns
    -------
    ndarray
        Detrended signal
        
    Examples
    --------
    >>> t = np.linspace(0, 1, 100)
    >>> signal_with_trend = 2*t + np.sin(2*np.pi*5*t)
    >>> detrended = detrend(signal_with_trend, method='linear')
    """
    data = np.asarray(data)
    
    match method:
        case 'linear':
            return signal.detrend(data, type='linear')
        case 'constant':
            return signal.detrend(data, type='constant')
        case _:
            raise ValueError(f"Unknown method: {method}")


def normalize(data: np.ndarray, method: str = 'zscore') -> np.ndarray:
    """
    Normalize signal.
    
    Parameters
    ----------
    data : array_like
        Input signal
    method : str, optional
        Normalization: 'zscore', 'minmax', 'maxabs' (default='zscore')
        
    Returns
    -------
    ndarray
        Normalized signal
        
    Examples
    --------
    >>> data = np.array([1, 2, 3, 4, 5])
    >>> normalized = normalize(data, method='zscore')
    """
    data = np.asarray(data, dtype=float)
    
    match method:
        case 'zscore':
            return (data - np.mean(data)) / np.std(data)
        
        case 'minmax':
            data_min = np.min(data)
            data_max = np.max(data)
            return (data - data_min) / (data_max - data_min)
        
        case 'maxabs':
            return data / np.max(np.abs(data))
        
        case _:
            raise ValueError(f"Unknown method: {method}")


def resample(data: np.ndarray,
            original_rate: float,
            target_rate: float) -> np.ndarray:
    """
    Resample signal to different sampling rate.
    
    Parameters
    ----------
    data : array_like
        Input signal
    original_rate : float
        Original sampling rate
    target_rate : float
        Target sampling rate
        
    Returns
    -------
    ndarray
        Resampled signal
        
    Examples
    --------
    >>> # Downsample from 1000 Hz to 100 Hz
    >>> t, sig = generate_test_signal('sine', sampling_rate=1000)
    >>> resampled = resample(sig, original_rate=1000, target_rate=100)
    """
    data = np.asarray(data)
    
    # Calculate number of samples in resampled signal
    n_samples = int(len(data) * target_rate / original_rate)
    
    # Resample
    resampled = signal.resample(data, n_samples)
    
    return resampled


def autocorrelation(data: np.ndarray,
                   max_lag: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate autocorrelation function.
    
    Parameters
    ----------
    data : array_like
        Input signal
    max_lag : int, optional
        Maximum lag (default=len(data)-1)
        
    Returns
    -------
    lags : ndarray
        Lag values
    acf : ndarray
        Autocorrelation function
        
    Examples
    --------
    >>> t, sig = generate_test_signal('sine', duration=2.0)
    >>> lags, acf = autocorrelation(sig, max_lag=200)
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(lags, acf)
    >>> plt.xlabel('Lag')
    >>> plt.ylabel('Autocorrelation')
    >>> plt.show()
    """
    data = np.asarray(data)
    n = len(data)
    
    if max_lag is None:
        max_lag = n - 1
    
    # Normalize data
    data_normalized = (data - np.mean(data)) / np.std(data)
    
    # Calculate autocorrelation using FFT (fast)
    fft_data = np.fft.fft(data_normalized, n=2*n)
    acf = np.fft.ifft(fft_data * np.conj(fft_data)).real
    acf = acf[:n] / np.arange(n, 0, -1)
    
    # Normalize to make acf[0] = 1
    acf = acf / acf[0]
    
    # Return only up to max_lag
    lags = np.arange(max_lag + 1)
    acf = acf[:max_lag + 1]
    
    return lags, acf


def crosscorrelation(x: np.ndarray,
                    y: np.ndarray,
                    max_lag: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate cross-correlation between two signals.
    
    Parameters
    ----------
    x, y : array_like
        Input signals
    max_lag : int, optional
        Maximum lag
        
    Returns
    -------
    lags : ndarray
        Lag values
    ccf : ndarray
        Cross-correlation function
        
    Examples
    --------
    >>> t, x = generate_test_signal('sine', duration=1.0)
    >>> y = np.roll(x, 10)  # Shifted version
    >>> lags, ccf = crosscorrelation(x, y)
    """
    x = np.asarray(x)
    y = np.asarray(y)
    
    # Normalize
    x = (x - np.mean(x)) / np.std(x)
    y = (y - np.mean(y)) / np.std(y)
    
    # Calculate cross-correlation
    ccf = signal.correlate(x, y, mode='full') / len(x)
    
    # Create lag array
    lags = signal.correlation_lags(len(x), len(y), mode='full')
    
    # Limit to max_lag if specified
    if max_lag is not None:
        mask = np.abs(lags) <= max_lag
        lags = lags[mask]
        ccf = ccf[mask]
    
    return lags, ccf


def hilbert_transform(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate Hilbert transform for instantaneous amplitude and phase.
    
    Parameters
    ----------
    data : array_like
        Input signal
        
    Returns
    -------
    amplitude : ndarray
        Instantaneous amplitude (envelope)
    phase : ndarray
        Instantaneous phase
    frequency : ndarray
        Instantaneous frequency
        
    Examples
    --------
    >>> t, sig = generate_test_signal('chirp', duration=1.0, sampling_rate=1000)
    >>> amp, phase, freq = hilbert_transform(sig)
    """
    data = np.asarray(data)
    
    # Calculate analytic signal
    analytic_signal = signal.hilbert(data)
    
    # Instantaneous amplitude
    amplitude = np.abs(analytic_signal)
    
    # Instantaneous phase
    phase = np.unwrap(np.angle(analytic_signal))
    
    # Instantaneous frequency (derivative of phase)
    frequency = np.diff(phase) / (2.0 * np.pi)
    frequency = np.append(frequency, frequency[-1])  # Match length
    
    return amplitude, phase, frequency


def envelope(data: np.ndarray) -> np.ndarray:
    """
    Calculate signal envelope using Hilbert transform.
    
    Parameters
    ----------
    data : array_like
        Input signal
        
    Returns
    -------
    ndarray
        Signal envelope
        
    Examples
    --------
    >>> t = np.linspace(0, 1, 1000)
    >>> # Amplitude modulated signal
    >>> carrier = np.sin(2*np.pi*50*t)
    >>> modulation = (1 + 0.5*np.sin(2*np.pi*5*t))
    >>> signal_am = modulation * carrier
    >>> env = envelope(signal_am)
    """
    amplitude, _, _ = hilbert_transform(data)
    return amplitude
