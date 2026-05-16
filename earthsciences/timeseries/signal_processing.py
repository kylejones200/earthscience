"""
Advanced signal processing

Convolution, deconvolution, filter design, and system analysis.
"""

import numpy as np
from scipy import signal


def convolve(
    signal1: np.ndarray, signal2: np.ndarray, mode: str = "full", method: str = "auto"
) -> np.ndarray:
    """
    Convolve two signals.

    Convolution is fundamental for filtering, system response, and smoothing.

    Parameters
    ----------
    signal1 : array_like
        First signal
    signal2 : array_like
        Second signal (often a filter kernel)
    mode : str, optional
        'full', 'valid', or 'same' (default='full')
        - 'full': Full convolution (length = len(s1) + len(s2) - 1)
        - 'valid': Only where signals fully overlap
        - 'same': Same length as signal1
    method : str, optional
        'auto', 'direct', or 'fft' (default='auto')

    Returns
    -------
    ndarray
        Convolved signal

    Notes
    -----
    Convolution: (f * g)(t) = ∫ f(τ)g(t-τ)dτ

    Applications:
    - Filtering: convolve signal with filter kernel
    - Smoothing: convolve with averaging window
    - System response: convolve input with impulse response

    Examples
    --------
    >>> # Smooth a signal with moving average
    >>> signal_data = np.random.randn(100)
    >>> kernel = np.ones(5) / 5  # 5-point moving average
    >>> smoothed = convolve(signal_data, kernel, mode='same')
    >>>
    >>> # Apply a Gaussian filter
    >>> from scipy.signal import gaussian
    >>> kernel = gaussian(11, std=2)
    >>> kernel = kernel / np.sum(kernel)  # Normalize
    >>> filtered = convolve(signal_data, kernel, mode='same')
    """
    signal1 = np.asarray(signal1)
    signal2 = np.asarray(signal2)

    return signal.convolve(signal1, signal2, mode=mode, method=method)


def deconvolve(
    signal_data: np.ndarray,
    impulse_response: np.ndarray,
    method: str = "wiener",
    noise_power: float = 0.1,
) -> np.ndarray:
    """
    Deconvolve a signal (inverse filtering).

    Attempts to recover original signal from convolved/filtered version.

    Parameters
    ----------
    signal_data : array_like
        Observed signal (result of convolution)
    impulse_response : array_like
        Known impulse response (filter that was applied)
    method : str, optional
        'wiener' or 'direct' (default='wiener')
    noise_power : float, optional
        Estimated noise power for Wiener deconvolution (default=0.1)

    Returns
    -------
    ndarray
        Deconvolved signal

    Notes
    -----
    Deconvolution is ill-posed and sensitive to noise.
    Wiener deconvolution provides regularization.

    Examples
    --------
    >>> # Original signal
    >>> original = np.sin(2*np.pi*5*np.linspace(0, 1, 100))
    >>>
    >>> # Apply impulse response (blur)
    >>> impulse = np.array([0.2, 0.6, 0.2])
    >>> blurred = convolve(original, impulse, mode='same')
    >>>
    >>> # Deconvolve
    >>> recovered = deconvolve(blurred, impulse, method='wiener')
    """
    signal_data = np.asarray(signal_data)
    impulse_response = np.asarray(impulse_response)

    if method == "direct":
        # Direct deconvolution (unstable!)
        quotient, remainder = signal.deconvolve(signal_data, impulse_response)
        return quotient

    elif method == "wiener":
        # Wiener deconvolution (more stable)
        # FFT-based approach
        S = np.fft.fft(signal_data)
        H = np.fft.fft(impulse_response, n=len(signal_data))

        # Wiener filter
        H_conj = np.conj(H)
        H_mag_sq = np.abs(H) ** 2

        # Regularized inverse filter
        W = H_conj / (H_mag_sq + noise_power)

        # Apply filter
        X_est = S * W

        # Inverse FFT
        x_est = np.fft.ifft(X_est).real

        return x_est

    else:
        raise ValueError(f"Unknown method: {method}")


def impulse_response(b: np.ndarray, a: np.ndarray, n_samples: int = 100) -> np.ndarray:
    """
    Calculate impulse response of a digital filter.

    Parameters
    ----------
    b : array_like
        Numerator coefficients
    a : array_like
        Denominator coefficients
    n_samples : int, optional
        Number of samples (default=100)

    Returns
    -------
    ndarray
        Impulse response

    Notes
    -----
    The impulse response characterizes a linear time-invariant system.

    Examples
    --------
    >>> # Butterworth lowpass filter
    >>> from scipy.signal import butter
    >>> b, a = butter(4, 0.2, btype='low')
    >>>
    >>> # Get impulse response
    >>> h = impulse_response(b, a, n_samples=50)
    >>>
    >>> # Plot to visualize filter characteristics
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(h)
    >>> plt.title('Impulse Response')
    """
    b = np.asarray(b)
    a = np.asarray(a)

    # Create impulse signal
    impulse = np.zeros(n_samples)
    impulse[0] = 1.0

    # Filter impulse
    h = signal.lfilter(b, a, impulse)

    return h


def frequency_response(
    b: np.ndarray, a: np.ndarray, sampling_rate: float = 1.0, n_points: int = 512
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate frequency response of a digital filter.

    Parameters
    ----------
    b : array_like
        Numerator coefficients
    a : array_like
        Denominator coefficients
    sampling_rate : float, optional
        Sampling rate (default=1.0)
    n_points : int, optional
        Number of frequency points (default=512)

    Returns
    -------
    frequencies : ndarray
        Frequency array
    magnitude : ndarray
        Magnitude response (linear scale)
    phase : ndarray
        Phase response (radians)

    Examples
    --------
    >>> from scipy.signal import butter
    >>> b, a = butter(4, 0.2, btype='low')
    >>>
    >>> freqs, mag, phase = frequency_response(b, a, sampling_rate=1000)
    >>>
    >>> # Plot magnitude response
    >>> import matplotlib.pyplot as plt
    >>> plt.subplot(2, 1, 1)
    >>> plt.plot(freqs, 20*np.log10(mag))  # dB scale
    >>> plt.ylabel('Magnitude (dB)')
    >>>
    >>> plt.subplot(2, 1, 2)
    >>> plt.plot(freqs, np.degrees(phase))
    >>> plt.ylabel('Phase (degrees)')
    >>> plt.xlabel('Frequency (Hz)')
    """
    b = np.asarray(b)
    a = np.asarray(a)

    # Calculate frequency response
    w, h = signal.freqz(b, a, worN=n_points, fs=sampling_rate)

    # Magnitude and phase
    magnitude = np.abs(h)
    phase = np.angle(h)

    return w, magnitude, phase


def design_fir_filter(
    cutoff: float | tuple[float, float],
    sampling_rate: float,
    filter_type: str = "lowpass",
    num_taps: int = 101,
    window: str = "hamming",
) -> np.ndarray:
    """
    Design FIR (Finite Impulse Response) filter.

    FIR filters are always stable and have linear phase.

    Parameters
    ----------
    cutoff : float or tuple
        Cutoff frequency (Hz) or (low, high) for bandpass/bandstop
    sampling_rate : float
        Sampling rate (Hz)
    filter_type : str, optional
        'lowpass', 'highpass', 'bandpass', or 'bandstop' (default='lowpass')
    num_taps : int, optional
        Filter length (odd number recommended) (default=101)
    window : str, optional
        Window function: 'hamming', 'hann', 'blackman', etc. (default='hamming')

    Returns
    -------
    ndarray
        FIR filter coefficients

    Examples
    --------
    >>> # Design lowpass filter
    >>> fs = 1000  # 1000 Hz sampling rate
    >>> cutoff = 100  # 100 Hz cutoff
    >>>
    >>> fir_coef = design_fir_filter(cutoff, fs, filter_type='lowpass', num_taps=51)
    >>>
    >>> # Apply filter to signal
    >>> filtered_signal = convolve(signal_data, fir_coef, mode='same')
    >>>
    >>> # Design bandpass filter
    >>> fir_bp = design_fir_filter((50, 150), fs, filter_type='bandpass')
    """
    nyquist = sampling_rate / 2.0

    # Normalize cutoff frequencies
    if isinstance(cutoff, (tuple, list)):
        cutoff_norm = [c / nyquist for c in cutoff]
    else:
        cutoff_norm = cutoff / nyquist

    # Design FIR filter using window method
    if filter_type == "lowpass":
        fir_coef = signal.firwin(num_taps, cutoff_norm, window=window, pass_zero="lowpass")
    elif filter_type == "highpass":
        fir_coef = signal.firwin(num_taps, cutoff_norm, window=window, pass_zero="highpass")
    elif filter_type == "bandpass":
        fir_coef = signal.firwin(num_taps, cutoff_norm, window=window, pass_zero="bandpass")
    elif filter_type == "bandstop":
        fir_coef = signal.firwin(num_taps, cutoff_norm, window=window, pass_zero="bandstop")
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")

    return fir_coef


def design_iir_filter(
    cutoff: float | tuple[float, float],
    sampling_rate: float,
    filter_type: str = "lowpass",
    filter_design: str = "butter",
    order: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Design IIR (Infinite Impulse Response) filter.

    IIR filters are more efficient than FIR but may have nonlinear phase.

    Parameters
    ----------
    cutoff : float or tuple
        Cutoff frequency (Hz) or (low, high) for bandpass/bandstop
    sampling_rate : float
        Sampling rate (Hz)
    filter_type : str, optional
        'lowpass', 'highpass', 'bandpass', or 'bandstop' (default='lowpass')
    filter_design : str, optional
        'butter', 'cheby1', 'cheby2', 'ellip', or 'bessel' (default='butter')
    order : int, optional
        Filter order (default=4)

    Returns
    -------
    b : ndarray
        Numerator coefficients
    a : ndarray
        Denominator coefficients

    Examples
    --------
    >>> # Design Butterworth lowpass
    >>> fs = 1000
    >>> b, a = design_iir_filter(100, fs, filter_type='lowpass',
    ...                          filter_design='butter', order=4)
    >>>
    >>> # Apply filter
    >>> filtered = signal.filtfilt(b, a, signal_data)
    >>>
    >>> # Design Chebyshev Type I bandpass
    >>> b, a = design_iir_filter((50, 150), fs, filter_type='bandpass',
    ...                          filter_design='cheby1', order=6)
    """
    nyquist = sampling_rate / 2.0

    # Normalize cutoff
    if isinstance(cutoff, (tuple, list)):
        cutoff_norm = [c / nyquist for c in cutoff]
    else:
        cutoff_norm = cutoff / nyquist

    # Design IIR filter using dispatch dictionary
    FILTER_DESIGNS = {
        "butter": lambda: signal.butter(order, cutoff_norm, btype=filter_type),
        "cheby1": lambda: signal.cheby1(order, 0.5, cutoff_norm, btype=filter_type),
        "cheby2": lambda: signal.cheby2(order, 40, cutoff_norm, btype=filter_type),
        "ellip": lambda: signal.ellip(order, 0.5, 40, cutoff_norm, btype=filter_type),
        "bessel": lambda: signal.bessel(order, cutoff_norm, btype=filter_type),
    }

    if filter_design not in FILTER_DESIGNS:
        valid_designs = ", ".join(f"'{d}'" for d in FILTER_DESIGNS.keys())
        raise ValueError(f"Unknown filter design '{filter_design}'. Valid options: {valid_designs}")

    b, a = FILTER_DESIGNS[filter_design]()

    return b, a


def transfer_function(b: np.ndarray, a: np.ndarray, sampling_rate: float = 1.0) -> dict:
    """
    Analyze transfer function of a digital filter.

    Parameters
    ----------
    b : array_like
        Numerator coefficients
    a : array_like
        Denominator coefficients
    sampling_rate : float, optional
        Sampling rate (default=1.0)

    Returns
    -------
    dict
        Transfer function analysis including poles, zeros, and gain

    Examples
    --------
    >>> from scipy.signal import butter
    >>> b, a = butter(4, 0.2, btype='low')
    >>>
    >>> tf = transfer_function(b, a, sampling_rate=1000)
    >>> print(f"Zeros: {tf['zeros']}")
    >>> print(f"Poles: {tf['poles']}")
    >>> print(f"Gain: {tf['gain']}")
    """
    b = np.asarray(b)
    a = np.asarray(a)

    # Get zeros, poles, and gain
    zeros, poles, gain = signal.tf2zpk(b, a)

    # Stability check (all poles must be inside unit circle)
    stable = np.all(np.abs(poles) < 1.0)

    # Frequency response
    freqs, mag, phase = frequency_response(b, a, sampling_rate)

    # -3dB cutoff frequency
    mag_db = 20 * np.log10(mag + 1e-10)
    cutoff_idx = np.where(mag_db < -3)[0]
    cutoff_freq = freqs[cutoff_idx[0]] if len(cutoff_idx) > 0 else None

    return {
        "zeros": zeros,
        "poles": poles,
        "gain": gain,
        "stable": stable,
        "cutoff_freq_3db": cutoff_freq,
        "frequencies": freqs,
        "magnitude": mag,
        "phase": phase,
    }


def group_delay(
    b: np.ndarray, a: np.ndarray, sampling_rate: float = 1.0, n_points: int = 512
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate group delay of a filter.

    Group delay measures the delay of different frequency components.
    Linear phase filters have constant group delay.

    Parameters
    ----------
    b : array_like
        Numerator coefficients
    a : array_like
        Denominator coefficients
    sampling_rate : float, optional
        Sampling rate (default=1.0)
    n_points : int, optional
        Number of frequency points (default=512)

    Returns
    -------
    frequencies : ndarray
        Frequency array
    delay : ndarray
        Group delay (samples)

    Examples
    --------
    >>> from scipy.signal import butter
    >>> b, a = butter(4, 0.2)
    >>>
    >>> freqs, delay = group_delay(b, a, sampling_rate=1000)
    >>>
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(freqs, delay)
    >>> plt.xlabel('Frequency (Hz)')
    >>> plt.ylabel('Group Delay (samples)')
    """
    b = np.asarray(b)
    a = np.asarray(a)

    # Calculate group delay
    w, gd = signal.group_delay((b, a), w=n_points, fs=sampling_rate)

    return w, gd


def zero_phase_filter(data: np.ndarray, b: np.ndarray, a: np.ndarray) -> np.ndarray:
    """
    Apply zero-phase filtering (forward-backward filtering).

    Eliminates phase distortion by filtering twice (forward and backward).

    Parameters
    ----------
    data : array_like
        Input signal
    b : array_like
        Numerator coefficients
    a : array_like
        Denominator coefficients

    Returns
    -------
    ndarray
        Zero-phase filtered signal

    Notes
    -----
    Effectively squares the magnitude response.
    No phase distortion but doubles the filter order.

    Examples
    --------
    >>> from scipy.signal import butter
    >>> b, a = butter(4, 0.2)
    >>>
    >>> # Regular filtering (has phase shift)
    >>> filtered_regular = signal.lfilter(b, a, data)
    >>>
    >>> # Zero-phase filtering (no phase shift)
    >>> filtered_zero_phase = zero_phase_filter(data, b, a)
    """
    data = np.asarray(data)
    b = np.asarray(b)
    a = np.asarray(a)

    return signal.filtfilt(b, a, data)


def minimum_phase_filter(h: np.ndarray) -> np.ndarray:
    """
    Convert FIR filter to minimum phase.

    Minimum phase filters have the shortest group delay.

    Parameters
    ----------
    h : array_like
        FIR filter coefficients

    Returns
    -------
    ndarray
        Minimum phase filter coefficients

    Examples
    --------
    >>> # Design linear phase FIR filter
    >>> h_linear = design_fir_filter(100, 1000, num_taps=51)
    >>>
    >>> # Convert to minimum phase
    >>> h_minphase = minimum_phase_filter(h_linear)
    """
    h = np.asarray(h)

    # FFT
    H = np.fft.fft(h)

    # Magnitude and phase
    mag = np.abs(H)

    # Minimum phase via Hilbert transform of log magnitude
    log_mag = np.log(mag + 1e-10)
    min_phase = -signal.hilbert(log_mag).imag

    # Reconstruct
    H_min = mag * np.exp(1j * min_phase)

    # IFFT
    h_min = np.fft.ifft(H_min).real

    return h_min
