"""
Spectral analysis methods

Power spectrum, periodogram, and spectral density estimation.
"""

import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq, ifft


def power_spectrum(
    data: np.ndarray,
    dt: float = 1.0,
    method: str = "periodogram",
    window: str | None = None,
    detrend_type: str = "linear",
    sampling_rate: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate power spectrum of a time series.

    Parameters
    ----------
    data : array_like
        Input time series
    dt : float, optional
        Time step / sampling interval (default=1.0)
    method : str, optional
        Method: 'periodogram', 'welch', 'fft', or 'multitaper' (default='periodogram')
    window : str, optional
        Window function: 'hann', 'hamming', 'blackman', etc.
    detrend_type : str, optional
        Detrending: 'linear', 'constant', or None (default='linear')
    sampling_rate : float, optional
        Sampling rate (1/dt). If provided, overrides dt.

    Returns
    -------
    frequencies : ndarray
        Frequency array
    power : ndarray
        Power spectral density

    Examples
    --------
    >>> t = np.linspace(0, 10, 1000)
    >>> signal = np.sin(2*np.pi*5*t) + np.sin(2*np.pi*10*t)
    >>> freqs, psd = power_spectrum(signal, dt=0.01)
    """
    data = np.asarray(data)

    if sampling_rate is not None:
        fs = sampling_rate
    else:
        fs = 1.0 / dt

    if detrend_type:
        data = signal.detrend(data, type=detrend_type)

    match method:
        case "fft":
            n = len(data)
            if window:
                window_func = signal.get_window(window, n)
                data = data * window_func

            fft_vals = fft(data)
            power = np.abs(fft_vals) ** 2 / n
            frequencies = fftfreq(n, dt)

            pos_mask = frequencies >= 0
            frequencies = frequencies[pos_mask]
            power = power[pos_mask]

        case "periodogram":
            frequencies, power = signal.periodogram(
                data, fs=fs, window=window if window else "boxcar", detrend=False
            )

        case "welch":
            frequencies, power = signal.welch(
                data, fs=fs, window=window if window else "hann", detrend=False
            )

        case "multitaper":
            from scipy.signal import windows

            n = len(data)
            nw = 4
            k = int(2 * nw - 1)

            tapers = windows.dpss(n, nw, k)

            fft_vals = np.zeros((k, n), dtype=complex)
            for i in range(k):
                fft_vals[i] = fft(data * tapers[i])

            power = np.mean(np.abs(fft_vals) ** 2, axis=0) / n
            frequencies = fftfreq(n, dt)

            pos_mask = frequencies >= 0
            frequencies = frequencies[pos_mask]
            power = power[pos_mask]

        case _:
            raise ValueError(
                f"Unknown method: {method}. Use 'fft', 'periodogram', 'welch', or 'multitaper'"
            )

    return frequencies, power


def blackman_tukey(
    data: np.ndarray, sampling_rate: float = 1.0, max_lag: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Blackman-Tukey spectral estimation using autocorrelation.

    Parameters
    ----------
    data : array_like
        Input time series
    sampling_rate : float, optional
        Sampling rate (default=1.0)
    max_lag : int, optional
        Maximum lag for autocorrelation (default=N/2)

    Returns
    -------
    frequencies : ndarray
        Frequency array
    power : ndarray
        Power spectral density
    """
    data = np.asarray(data)
    n = len(data)

    if max_lag is None:
        max_lag = n // 2

    data_normalized = (data - np.mean(data)) / np.std(data)
    autocorr = np.correlate(data_normalized, data_normalized, mode="full")
    autocorr = autocorr[n - 1 :]
    autocorr = autocorr[:max_lag] / autocorr[0]

    window = signal.windows.blackman(len(autocorr))
    autocorr_windowed = autocorr * window

    power = np.abs(fft(autocorr_windowed))
    frequencies = fftfreq(len(autocorr_windowed), 1 / sampling_rate)

    pos_mask = frequencies >= 0
    frequencies = frequencies[pos_mask]
    power = power[pos_mask]

    return frequencies, power


def lomb_scargle(
    t: np.ndarray, y: np.ndarray, frequencies: np.ndarray | None = None, normalize: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """
    Lomb-Scargle periodogram for unevenly sampled data.

    Parameters
    ----------
    t : array_like
        Time values (can be unevenly spaced)
    y : array_like
        Signal values
    frequencies : array_like, optional
        Frequencies to evaluate (default=auto)
    normalize : bool, optional
        Normalize periodogram (default=True)

    Returns
    -------
    frequencies : ndarray
        Frequency array
    power : ndarray
        Lomb-Scargle power
    """
    from scipy.signal import lombscargle

    t = np.asarray(t)
    y = np.asarray(y)

    if frequencies is None:
        dt_min = np.min(np.diff(np.sort(t)))
        f_max = 0.5 / dt_min
        frequencies = np.linspace(0.01 * f_max, f_max, 1000)

    angular_freqs = 2 * np.pi * frequencies

    power = lombscargle(t, y - np.mean(y), angular_freqs, normalize=normalize)

    return frequencies, power


def spectrogram(
    data: np.ndarray,
    dt: float = 1.0,
    window_size: int = 256,
    overlap: int = 128,
    sampling_rate: float | None = None,
) -> dict:
    """
    Calculate spectrogram (time-frequency representation).

    Parameters
    ----------
    data : array_like
        Input time series
    dt : float, optional
        Time step (default=1.0)
    window_size : int, optional
        Window size (default=256)
    overlap : int, optional
        Overlap between windows in samples (default=128)
    sampling_rate : float, optional
        Sampling rate (1/dt). If provided, overrides dt.

    Returns
    -------
    dict
        Dictionary containing:
        - time: time array
        - freq: frequency array
        - power: 2D power array (freq x time)
    """
    data = np.asarray(data)

    if sampling_rate is not None:
        fs = sampling_rate
    else:
        fs = 1.0 / dt

    frequencies, times, Sxx = signal.spectrogram(
        data, fs=fs, window="hann", nperseg=window_size, noverlap=overlap
    )

    return {"time": times, "freq": frequencies, "power": Sxx}


def coherence(
    x: np.ndarray, y: np.ndarray, dt: float = 1.0, window_size: int = 256, overlap: int = 128
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate coherence between two signals.

    Parameters
    ----------
    x, y : array_like
        Input signals
    dt : float, optional
        Time step (default=1.0)
    window_size : int, optional
        Window size (default=256)
    overlap : int, optional
        Overlap between windows (default=128)

    Returns
    -------
    frequencies : ndarray
        Frequency array
    coherence : ndarray
        Coherence values (0 to 1)
    """
    x = np.asarray(x)
    y = np.asarray(y)

    fs = 1.0 / dt

    frequencies, coh = signal.coherence(
        x, y, fs=fs, window="hann", nperseg=window_size, noverlap=overlap
    )

    return frequencies, coh


def cross_spectrum(
    x: np.ndarray, y: np.ndarray, dt: float = 1.0, window_size: int = 256, overlap: int = 128
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate cross-spectrum between two signals.

    Parameters
    ----------
    x, y : array_like
        Input signals
    dt : float, optional
        Time step (default=1.0)
    window_size : int, optional
        Window size (default=256)
    overlap : int, optional
        Overlap between windows (default=128)

    Returns
    -------
    frequencies : ndarray
        Frequency array
    cross_spectrum : ndarray
        Complex cross-spectrum
    """
    x = np.asarray(x)
    y = np.asarray(y)

    fs = 1.0 / dt

    frequencies, Pxy = signal.csd(x, y, fs=fs, window="hann", nperseg=window_size, noverlap=overlap)

    return frequencies, Pxy


def evolutionary_spectrum(
    data: np.ndarray,
    sampling_rate: float = 1.0,
    window_size: int = 256,
    hop_size: int | None = None,
) -> dict:
    """
    Calculate evolutionary (time-varying) power spectrum.

    Similar to spectrogram but specifically for analyzing how spectral
    properties evolve over time.

    Parameters
    ----------
    data : array_like
        Input time series
    sampling_rate : float, optional
        Sampling rate (default=1.0)
    window_size : int, optional
        Window size for STFT (default=256)
    hop_size : int, optional
        Hop size between windows (default=window_size//4)

    Returns
    -------
    dict
        Dictionary containing times, frequencies, and power
    """
    data = np.asarray(data)

    if hop_size is None:
        hop_size = window_size // 4

    result = spectrogram(
        data, sampling_rate=sampling_rate, window_size=window_size, overlap=window_size - hop_size
    )

    return {
        "times": result["time"],
        "frequencies": result["freq"],
        "power": result["power"],
    }


def autocorrelation(data: np.ndarray, max_lag: int | None = None) -> np.ndarray:
    """
    Calculate autocorrelation function.

    Parameters
    ----------
    data : array_like
        Input time series
    max_lag : int, optional
        Maximum lag (default=len(data)-1)

    Returns
    -------
    acf : ndarray
        Autocorrelation function, normalized so acf[0]=1
    """
    data = np.asarray(data)
    n = len(data)

    if max_lag is None:
        max_lag = n - 1

    data_mean = np.mean(data)
    data_centered = data - data_mean

    fft_data = fft(data_centered, n=2 * n)
    power = np.abs(fft_data) ** 2
    acf_full = ifft(power).real[:n]

    acf = acf_full / acf_full[0]

    return acf[: max_lag + 1]


def multitaper_spectrum(
    data: np.ndarray,
    dt: float = 1.0,
    NW: float = 4.0,
    k: int | None = None,
    adaptive: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Thomson's multitaper power spectral density estimate.

    Uses multiple orthogonal tapers (Slepian sequences) to reduce variance
    in spectral estimates while maintaining good frequency resolution.
    Superior to single-taper methods for detecting weak signals.

    Parameters
    ----------
    data : array_like
        Input time series
    dt : float, optional
        Time step / sampling interval (default=1.0)
    NW : float, optional
        Time-bandwidth product (default=4.0)
        Controls frequency resolution: Δf ≈ NW / (N * dt)
        Common values: 2.5, 3, 4, 5
    k : int, optional
        Number of tapers to use (default=2*NW-1)
        More tapers = lower variance but lower resolution
    adaptive : bool, optional
        Use adaptive weighting (default=True)
        Weights tapers based on eigenvalues for better performance

    Returns
    -------
    frequencies : ndarray
        Frequency array
    power : ndarray
        Multitaper power spectral density

    Notes
    -----
    The multitaper method was developed by David J. Thomson (1982).
    It provides optimal bias-variance tradeoff for spectral estimation.

    Time-bandwidth product NW controls the tradeoff:
    - Larger NW: better frequency resolution, higher variance
    - Smaller NW: lower variance, poorer frequency resolution

    References
    ----------
    Thomson, D. J. (1982). Spectrum estimation and harmonic analysis.
    Proceedings of the IEEE, 70(9), 1055-1096.

    Examples
    --------
    >>> # Signal with two close frequencies
    >>> t = np.linspace(0, 10, 1000)
    >>> signal = np.sin(2*np.pi*5*t) + np.sin(2*np.pi*5.5*t)
    >>> signal += np.random.randn(1000) * 0.5  # Add noise
    >>>
    >>> # Multitaper spectrum can resolve close frequencies
    >>> freqs, psd = multitaper_spectrum(signal, dt=0.01, NW=4.0)
    >>>
    >>> # Compare with standard periodogram
    >>> freqs_p, psd_p = power_spectrum(signal, dt=0.01, method='periodogram')
    """
    data = np.asarray(data, dtype=float)
    N = len(data)

    if k is None:
        k = int(2 * NW - 1)

    # Remove mean
    data = data - np.mean(data)

    # Generate Slepian sequences (DPSS - Discrete Prolate Spheroidal Sequences)
    tapers, eigenvalues = _dpss_windows(N, NW, k)

    # Sampling frequency
    fs = 1.0 / dt

    # Compute tapered FFTs
    n_fft = N
    freqs = np.fft.rfftfreq(n_fft, dt)

    # Calculate spectrum for each taper
    S_k = np.zeros((k, len(freqs)), dtype=complex)
    for i in range(k):
        tapered_data = data * tapers[i, :]
        S_k[i, :] = np.fft.rfft(tapered_data, n=n_fft)

    # Power for each taper
    P_k = np.abs(S_k) ** 2

    if adaptive:
        # Adaptive weighting (Thomson, 1982)
        # Iteratively compute weights based on eigenvalues
        weights = np.zeros((k, len(freqs)))

        # Initial estimate: average of all tapers
        S_est = np.mean(P_k, axis=0)

        # Iterate to convergence
        for _ in range(5):  # Usually converges in 2-3 iterations
            for i in range(k):
                # Adaptive weights
                weights[i, :] = (eigenvalues[i] * S_est) / (
                    eigenvalues[i] * S_est + (1 - eigenvalues[i]) * np.mean(S_est)
                )

            # Weighted average
            S_est = np.sum(weights * P_k, axis=0) / np.sum(weights, axis=0)

        power = S_est
    else:
        # Simple eigenvalue weighting
        weights = eigenvalues[:, np.newaxis]
        power = np.sum(weights * P_k, axis=0) / np.sum(weights)

    # Normalize (one-sided spectrum)
    power = power / (fs * N)
    power[1:-1] *= 2  # Double for one-sided spectrum (except DC and Nyquist)

    return freqs, power


def _dpss_windows(N: int, NW: float, k: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate Discrete Prolate Spheroidal Sequences (DPSS) / Slepian sequences.

    These are the optimal tapers for multitaper spectral estimation.

    Parameters
    ----------
    N : int
        Length of sequences
    NW : float
        Time-bandwidth product
    k : int
        Number of sequences to generate

    Returns
    -------
    tapers : ndarray
        DPSS sequences, shape (k, N)
    eigenvalues : ndarray
        Concentration eigenvalues, shape (k,)
    """
    from scipy import linalg

    # Frequency bandwidth
    W = NW / N

    # Create tridiagonal matrix for eigenvalue problem
    # This is the Toeplitz matrix approach
    n = np.arange(N)

    # Main diagonal
    main_diag = ((N - 1 - 2 * n) / 2) ** 2 * np.cos(2 * np.pi * W)

    # Off-diagonal
    off_diag = n[1:] * (N - n[1:]) / 2

    # Construct tridiagonal matrix
    diagonals = [main_diag, off_diag, off_diag]
    A = linalg.toeplitz(main_diag) + np.diag(off_diag, 1) + np.diag(off_diag, -1)

    # Solve eigenvalue problem
    eigenvalues, eigenvectors = linalg.eigh(A, subset_by_index=[N - k, N - 1])

    # Sort in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Normalize tapers
    tapers = eigenvectors.T
    for i in range(k):
        tapers[i, :] /= np.sqrt(np.sum(tapers[i, :] ** 2))

    # Ensure first element is positive (convention)
    for i in range(k):
        if tapers[i, 0] < 0:
            tapers[i, :] *= -1

    return tapers, eigenvalues


def multitaper_coherence(
    x: np.ndarray, y: np.ndarray, dt: float = 1.0, NW: float = 4.0, k: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Multitaper coherence between two signals.

    Coherence measures the linear relationship between two signals
    as a function of frequency.

    Parameters
    ----------
    x, y : array_like
        Input time series (must be same length)
    dt : float, optional
        Time step (default=1.0)
    NW : float, optional
        Time-bandwidth product (default=4.0)
    k : int, optional
        Number of tapers (default=2*NW-1)

    Returns
    -------
    frequencies : ndarray
        Frequency array
    coherence : ndarray
        Coherence values (0 to 1)

    Examples
    --------
    >>> t = np.linspace(0, 10, 1000)
    >>> x = np.sin(2*np.pi*5*t) + np.random.randn(1000)*0.5
    >>> y = np.sin(2*np.pi*5*t + 0.5) + np.random.randn(1000)*0.5
    >>> freqs, coh = multitaper_coherence(x, y, dt=0.01)
    >>> print(f"Peak coherence at {freqs[np.argmax(coh)]:.1f} Hz")
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) != len(y):
        raise ValueError("x and y must have the same length")

    N = len(x)

    if k is None:
        k = int(2 * NW - 1)

    # Remove means
    x = x - np.mean(x)
    y = y - np.mean(y)

    # Generate tapers
    tapers, eigenvalues = _dpss_windows(N, NW, k)

    # Frequency array
    freqs = np.fft.rfftfreq(N, dt)

    # Compute tapered FFTs for both signals
    X_k = np.zeros((k, len(freqs)), dtype=complex)
    Y_k = np.zeros((k, len(freqs)), dtype=complex)

    for i in range(k):
        X_k[i, :] = np.fft.rfft(x * tapers[i, :])
        Y_k[i, :] = np.fft.rfft(y * tapers[i, :])

    # Cross-spectrum and auto-spectra
    S_xy = np.mean(X_k * np.conj(Y_k), axis=0)
    S_xx = np.mean(np.abs(X_k) ** 2, axis=0)
    S_yy = np.mean(np.abs(Y_k) ** 2, axis=0)

    # Coherence
    coherence = np.abs(S_xy) ** 2 / (S_xx * S_yy)

    return freqs, coherence


def multitaper_spectrogram(
    data: np.ndarray,
    dt: float = 1.0,
    window_length: int = 256,
    overlap: int = 128,
    NW: float = 4.0,
    k: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Time-frequency representation using multitaper method.

    Combines multitaper spectral estimation with sliding windows
    for time-varying spectral analysis.

    Parameters
    ----------
    data : array_like
        Input time series
    dt : float, optional
        Time step (default=1.0)
    window_length : int, optional
        Length of each window (default=256)
    overlap : int, optional
        Number of overlapping samples (default=128)
    NW : float, optional
        Time-bandwidth product (default=4.0)
    k : int, optional
        Number of tapers (default=2*NW-1)

    Returns
    -------
    times : ndarray
        Time array for windows
    frequencies : ndarray
        Frequency array
    spectrogram : ndarray
        Time-frequency power, shape (n_times, n_freqs)

    Examples
    --------
    >>> # Chirp signal (frequency increases with time)
    >>> t = np.linspace(0, 10, 1000)
    >>> signal = np.sin(2*np.pi*(5 + 2*t)*t)
    >>>
    >>> times, freqs, spec = multitaper_spectrogram(
    ...     signal, dt=0.01, window_length=128, overlap=64
    ... )
    """
    data = np.asarray(data, dtype=float)
    N = len(data)

    # Calculate number of windows
    step = window_length - overlap
    n_windows = (N - window_length) // step + 1

    # Initialize output
    times = np.zeros(n_windows)
    spectrogram_data = []

    for i in range(n_windows):
        start = i * step
        end = start + window_length

        if end > N:
            break

        # Extract window
        window_data = data[start:end]

        # Compute multitaper spectrum for this window
        freqs, power = multitaper_spectrum(window_data, dt=dt, NW=NW, k=k)

        times[i] = (start + end) / 2 * dt
        spectrogram_data.append(power)

    spectrogram = np.array(spectrogram_data)
    times = times[: len(spectrogram_data)]

    return times, freqs, spectrogram
