"""
Digital filtering methods 

Butterworth, FIR, moving average, and other filters.
"""

import numpy as np
from scipy import signal
from typing import Tuple, Optional, Union


def butter_filter(data: np.ndarray,
                 cutoff: Union[float, Tuple[float, float]],
                 sampling_rate: float = 1.0,
                 filter_type: str = 'lowpass',
                 order: int = 4) -> np.ndarray:
    """
    Apply Butterworth filter to data.
    
    Parameters
    ----------
    data : array_like
        Input signal
    cutoff : float or tuple
        Cutoff frequency (or tuple of (low, high) for bandpass/bandstop)
    sampling_rate : float, optional
        Sampling rate (default=1.0)
    filter_type : str, optional
        Filter type: 'lowpass', 'highpass', 'bandpass', 'bandstop' (default='lowpass')
    order : int, optional
        Filter order (default=4)
        
    Returns
    -------
    ndarray
        Filtered signal
        
    Warnings
    --------
    **NON-CAUSAL FILTERING**
    
    This function uses zero-phase filtering (filtfilt) which processes data
    in BOTH forward and backward directions. This means:
    
    ✅ **Appropriate for**:
       - Retrospective data analysis
       - Signal visualization
       - Data cleaning and preprocessing
       - Scientific analysis of recorded data
    
    ❌ **Inappropriate for**:
       - Real-time/online signal processing
       - Causal analysis requiring strict time ordering
       - Forecasting and prediction
       - Live data streaming applications
    
    **For causal (online) filtering**, use signal.lfilter() instead:
    
    >>> from scipy import signal
    >>> b, a = signal.butter(4, cutoff/(sampling_rate/2), btype='lowpass')
    >>> filtered_causal = signal.lfilter(b, a, data)  # Forward pass only
    
    Note: Causal filtering introduces phase delay; use filtfilt only when
    analyzing historical data where all samples are available.
        
    Examples
    --------
    >>> # Remove high-frequency noise (retrospective analysis)
    >>> t = np.linspace(0, 1, 1000)
    >>> signal_clean = np.sin(2*np.pi*5*t)
    >>> noise = np.random.randn(1000) * 0.5
    >>> signal_noisy = signal_clean + noise
    >>> filtered = butter_filter(signal_noisy, cutoff=10, 
    ...                          sampling_rate=1000, filter_type='lowpass')
    """
    data = np.asarray(data)
    nyquist = sampling_rate / 2
    
    # Normalize cutoff frequency
    if isinstance(cutoff, (tuple, list)):
        normalized_cutoff = [c / nyquist for c in cutoff]
    else:
        normalized_cutoff = cutoff / nyquist
    
    # Design filter
    b, a = signal.butter(order, normalized_cutoff, btype=filter_type)
    
    # Apply filter (using filtfilt for zero-phase filtering)
    filtered_data = signal.filtfilt(b, a, data)
    
    return filtered_data


def fir_filter(data: np.ndarray,
              cutoff: Union[float, Tuple[float, float]],
              sampling_rate: float = 1.0,
              filter_type: str = 'lowpass',
              num_taps: int = 101,
              window: str = 'hamming') -> np.ndarray:
    """
    Apply FIR (Finite Impulse Response) filter to data.
    
    FIR filters are always stable and can have linear phase.
    
    Parameters
    ----------
    data : array_like
        Input signal
    cutoff : float or tuple
        Cutoff frequency (or tuple for bandpass/bandstop)
    sampling_rate : float, optional
        Sampling rate (default=1.0)
    filter_type : str, optional
        Filter type: 'lowpass', 'highpass', 'bandpass', 'bandstop'
    num_taps : int, optional
        Number of filter taps (default=101)
    window : str, optional
        Window function (default='hamming')
        
    Returns
    -------
    ndarray
        Filtered signal
        
    Examples
    --------
    >>> t = np.linspace(0, 1, 1000)
    >>> signal_data = np.sin(2*np.pi*5*t) + np.sin(2*np.pi*50*t)
    >>> filtered = fir_filter(signal_data, cutoff=20,
    ...                       sampling_rate=1000, filter_type='lowpass')
    """
    data = np.asarray(data)
    nyquist = sampling_rate / 2
    
    # Normalize cutoff frequency
    if isinstance(cutoff, tuple):
        normalized_cutoff = [c / nyquist for c in cutoff]
    else:
        normalized_cutoff = cutoff / nyquist
    
    # Design FIR filter
    if filter_type == 'bandpass':
        taps = signal.firwin(num_taps, normalized_cutoff, 
                           pass_zero=False, window=window)
    elif filter_type == 'bandstop':
        taps = signal.firwin(num_taps, normalized_cutoff,
                           pass_zero=True, window=window)
    else:
        pass_zero = (filter_type == 'lowpass')
        taps = signal.firwin(num_taps, normalized_cutoff,
                           pass_zero=pass_zero, window=window)
    
    # Apply filter
    filtered_data = signal.filtfilt(taps, [1.0], data)
    
    return filtered_data


def moving_average_filter(data: np.ndarray,
                          window_size: int = 5,
                          mode: str = 'same') -> np.ndarray:
    """
    Apply moving average (boxcar) filter.
    
    Simple smoothing filter that averages neighboring points.
    
    Parameters
    ----------
    data : array_like
        Input signal
    window_size : int, optional
        Window size for averaging (default=5)
    mode : str, optional
        Boundary handling: 'same', 'valid', 'full' (default='same')
        
    Returns
    -------
    ndarray
        Smoothed signal
        
    Examples
    --------
    >>> data = np.array([1, 2, 1, 2, 1, 2, 1, 2, 1, 2])
    >>> smoothed = moving_average_filter(data, window_size=3)
    """
    data = np.asarray(data)
    
    if window_size < 1:
        raise ValueError("Window size must be at least 1")
    
    # Create averaging kernel
    kernel = np.ones(window_size) / window_size
    
    # Convolve with data
    smoothed = np.convolve(data, kernel, mode=mode)
    
    return smoothed


def savitzky_golay(data: np.ndarray,
                  window_size: int = 11,
                  poly_order: int = 3,
                  deriv: int = 0,
                  window: Optional[int] = None,
                  order: Optional[int] = None) -> np.ndarray:
    """
    Apply Savitzky-Golay filter.
    
    Smooths data using polynomial fitting within a moving window.
    Can also calculate derivatives.
    
    Parameters
    ----------
    data : array_like
        Input signal
    window_size : int, optional
        Window size (must be odd, default=11)
    poly_order : int, optional
        Polynomial order (default=3)
    deriv : int, optional
        Derivative order (0=smooth, 1=first derivative, etc.) (default=0)
        
    Returns
    -------
    ndarray
        Filtered signal or derivative
        
    Examples
    --------
    >>> # Smooth noisy data
    >>> x = np.linspace(0, 2*np.pi, 100)
    >>> y = np.sin(x) + np.random.randn(100) * 0.1
    >>> smoothed = savitzky_golay(y, window_size=11, poly_order=3)
    >>> 
    >>> # Calculate first derivative
    >>> derivative = savitzky_golay(y, window_size=11, poly_order=3, deriv=1)
    """
    # Support legacy parameter names
    if window is not None:
        window_size = window
    if order is not None:
        poly_order = order
    
    data = np.asarray(data)
    
    if window_size % 2 == 0:
        window_size += 1  # Make odd
    
    if window_size < poly_order + 2:
        raise ValueError("Window size must be larger than polynomial order")
    
    filtered = signal.savgol_filter(data, window_size, poly_order, deriv=deriv)
    
    return filtered


def median_filter(data: np.ndarray,
                 kernel_size: int = 3,
                 window: Optional[int] = None) -> np.ndarray:
    """
    Apply median filter.
    
    Particularly effective at removing outliers and impulsive noise.
    
    Parameters
    ----------
    data : array_like
        Input signal
    kernel_size : int, optional
        Kernel size (default=3)
        
    Returns
    -------
    ndarray
        Filtered signal
        
    Examples
    --------
    >>> # Remove outliers
    >>> data = np.array([1, 2, 3, 100, 4, 5, 6, -50, 7, 8])
    >>> filtered = median_filter(data, kernel_size=3)
    """
    data = np.asarray(data)
    
    if window is not None:
        kernel_size = window
    
    filtered = signal.medfilt(data, kernel_size=kernel_size)
    
    return filtered


def notch_filter(data: np.ndarray,
                frequency: float,
                sampling_rate: float = 1.0,
                quality: float = 30.0) -> np.ndarray:
    """
    Apply notch filter to remove specific frequency.
    
    Useful for removing interference at known frequencies (e.g., 50/60 Hz power line).
    
    Parameters
    ----------
    data : array_like
        Input signal
    frequency : float
        Frequency to remove
    sampling_rate : float, optional
        Sampling rate (default=1.0)
    quality : float, optional
        Quality factor (higher = narrower notch) (default=30.0)
        
    Returns
    -------
    ndarray
        Filtered signal
        
    Examples
    --------
    >>> # Remove 60 Hz interference
    >>> t = np.linspace(0, 1, 1000)
    >>> signal_clean = np.sin(2*np.pi*5*t)
    >>> interference = 0.5 * np.sin(2*np.pi*60*t)
    >>> signal_noisy = signal_clean + interference
    >>> filtered = notch_filter(signal_noisy, frequency=60,
    ...                          sampling_rate=1000)
    """
    data = np.asarray(data)
    
    # Design notch filter
    b, a = signal.iirnotch(frequency, quality, sampling_rate)
    
    # Apply filter
    filtered = signal.filtfilt(b, a, data)
    
    return filtered


def bandpass_filter(data: np.ndarray,
                   lowcut: float,
                   highcut: float,
                   sampling_rate: float = 1.0,
                   order: int = 4) -> np.ndarray:
    """
    Apply bandpass filter to keep frequencies in a specific range.
    
    Parameters
    ----------
    data : array_like
        Input signal
    lowcut : float
        Lower cutoff frequency
    highcut : float
        Upper cutoff frequency
    sampling_rate : float, optional
        Sampling rate (default=1.0)
    order : int, optional
        Filter order (default=4)
        
    Returns
    -------
    ndarray
        Filtered signal
        
    Warnings
    --------
    Uses non-causal filtering (filtfilt). See butter_filter() for details
    on when this is appropriate vs. causal filtering.
        
    Examples
    --------
    >>> # Keep only frequencies between 5 and 15 Hz
    >>> t = np.linspace(0, 1, 1000)
    >>> signal_data = (np.sin(2*np.pi*2*t) + 
    ...                np.sin(2*np.pi*10*t) + 
    ...                np.sin(2*np.pi*50*t))
    >>> filtered = bandpass_filter(signal_data, lowcut=5, highcut=15,
    ...                            sampling_rate=1000)
    """
    return butter_filter(data, (lowcut, highcut), sampling_rate,
                        filter_type='bandpass', order=order)


def highpass_filter(data: np.ndarray,
                   cutoff: float,
                   sampling_rate: float = 1.0,
                   order: int = 4) -> np.ndarray:
    """
    Apply highpass filter to remove low frequencies.
    
    Parameters
    ----------
    data : array_like
        Input signal
    cutoff : float
        Cutoff frequency
    sampling_rate : float, optional
        Sampling rate (default=1.0)
    order : int, optional
        Filter order (default=4)
        
    Returns
    -------
    ndarray
        Filtered signal
        
    Warnings
    --------
    Uses non-causal filtering (filtfilt). See butter_filter() for details
    on when this is appropriate vs. causal filtering.
        
    Examples
    --------
    >>> # Remove trend and keep high-frequency variations
    >>> t = np.linspace(0, 1, 1000)
    >>> trend = 0.5 * t
    >>> signal_data = trend + np.sin(2*np.pi*50*t)
    >>> filtered = highpass_filter(signal_data, cutoff=10,
    ...                            sampling_rate=1000)
    """
    return butter_filter(data, cutoff, sampling_rate,
                        filter_type='highpass', order=order)


def lowpass_filter(data: np.ndarray,
                  cutoff: float,
                  sampling_rate: float = 1.0,
                  order: int = 4) -> np.ndarray:
    """
    Apply lowpass filter to remove high frequencies.
    
    Parameters
    ----------
    data : array_like
        Input signal
    cutoff : float
        Cutoff frequency
    sampling_rate : float, optional
        Sampling rate (default=1.0)
    order : int, optional
        Filter order (default=4)
        
    Returns
    -------
    ndarray
        Filtered signal
        
    Warnings
    --------
    Uses non-causal filtering (filtfilt). See butter_filter() for details
    on when this is appropriate vs. causal filtering.
        
    Examples
    --------
    >>> # Remove high-frequency noise
    >>> t = np.linspace(0, 1, 1000)
    >>> signal_clean = np.sin(2*np.pi*5*t)
    >>> noise = np.random.randn(1000) * 0.5
    >>> signal_noisy = signal_clean + noise
    >>> filtered = lowpass_filter(signal_noisy, cutoff=10,
    ...                           sampling_rate=1000)
    """
    return butter_filter(data, cutoff, sampling_rate,
                        filter_type='lowpass', order=order)


def butterworth_filter(data: np.ndarray,
                       cutoff: Union[float, list],
                       fs: float,
                       filter_type: str = 'lowpass',
                       order: int = 4) -> np.ndarray:
    """
    Apply Butterworth filter (alias for butter_filter).
    
    Parameters
    ----------
    data : array_like
        Input signal
    cutoff : float or list
        Cutoff frequency (or [low, high] for bandpass/bandstop)
    fs : float
        Sampling frequency
    filter_type : str
        'lowpass', 'highpass', 'bandpass', or 'bandstop'
    order : int
        Filter order
        
    Returns
    -------
    ndarray
        Filtered signal
    """
    return butter_filter(data, cutoff, fs, filter_type, order)


def moving_average(data: np.ndarray, window: int, mode: str = 'valid') -> np.ndarray:
    """
    Apply moving average filter (alias for moving_average_filter).
    
    Parameters
    ----------
    data : array_like
        Input signal
    window : int
        Window size
    mode : str
        'valid', 'same', or 'full'
        
    Returns
    -------
    ndarray
        Smoothed signal
    """
    return moving_average_filter(data, window)


def kalman_filter(measurements: np.ndarray,
                  process_variance: float = 1e-5,
                  measurement_variance: float = 0.1,
                  initial_estimate: Optional[float] = None,
                  initial_error: float = 1.0) -> np.ndarray:
    """
    Apply 1D Kalman filter for state estimation.
    
    Parameters
    ----------
    measurements : array_like
        Noisy measurements
    process_variance : float
        Process noise variance (Q)
    measurement_variance : float
        Measurement noise variance (R)
    initial_estimate : float, optional
        Initial state estimate (default=first measurement)
    initial_error : float
        Initial estimate error covariance (P)
        
    Returns
    -------
    ndarray
        Filtered estimates
    """
    measurements = np.asarray(measurements)
    n = len(measurements)
    
    if initial_estimate is None:
        initial_estimate = measurements[0]
    
    estimates = np.zeros(n)
    estimate = initial_estimate
    error_cov = initial_error
    
    for i in range(n):
        predicted_estimate = estimate
        predicted_error_cov = error_cov + process_variance
        
        kalman_gain = predicted_error_cov / (predicted_error_cov + measurement_variance)
        estimate = predicted_estimate + kalman_gain * (measurements[i] - predicted_estimate)
        error_cov = (1 - kalman_gain) * predicted_error_cov
        
        estimates[i] = estimate
    
    return estimates


def wiener_filter(data: np.ndarray,
                  noise_power: Optional[float] = None) -> np.ndarray:
    """
    Apply Wiener filter for noise reduction.
    
    Parameters
    ----------
    data : array_like
        Noisy signal
    noise_power : float, optional
        Noise power estimate (default=auto-estimate)
        
    Returns
    -------
    ndarray
        Filtered signal
    """
    from scipy.signal import wiener as scipy_wiener
    
    data = np.asarray(data)
    filtered = scipy_wiener(data)
    
    return filtered
