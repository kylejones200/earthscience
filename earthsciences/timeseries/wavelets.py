"""
Wavelet analysis methods

Continuous and discrete wavelet transforms.
"""

import numpy as np
import pywt


def cwt(
    data: np.ndarray,
    wavelet: str = "morl",
    scales: np.ndarray | None = None,
    sampling_period: float = 1.0,
) -> dict:
    """
    Continuous Wavelet Transform.

    Parameters
    ----------
    data : array_like
        Input signal
    wavelet : str
        Wavelet name: 'morl' (Morlet), 'mexh' (Mexican hat), 'paul', etc.
    scales : array_like, optional
        Scales to use (default=auto)
    sampling_period : float
        Sampling period

    Returns
    -------
    dict
        Dictionary with coefficients, scales, frequencies
    """
    data = np.asarray(data)

    # Map common names
    wavelet_map = {"morlet": "morl", "mexican_hat": "mexh", "paul": "paul"}
    wavelet = wavelet_map.get(wavelet, wavelet)

    if scales is None:
        scales = np.arange(1, min(len(data) // 2, 128))

    coefficients, frequencies = pywt.cwt(data, scales, wavelet, sampling_period=sampling_period)

    return {"coefficients": coefficients, "scales": scales, "frequencies": frequencies}


def dwt(
    data: np.ndarray, wavelet: str = "db4", level: int | None = None, mode: str = "symmetric"
) -> dict:
    """
    Discrete Wavelet Transform.

    Parameters
    ----------
    data : array_like
        Input signal
    wavelet : str
        Wavelet name (default='db4')
    level : int, optional
        Decomposition level (default=auto)
    mode : str
        Signal extension mode

    Returns
    -------
    dict
        Dictionary with approximation (cA) and detail (cD) coefficients
    """
    data = np.asarray(data)

    if level is None:
        level = pywt.dwt_max_level(len(data), wavelet)

    coeffs = pywt.wavedec(data, wavelet, level=level, mode=mode)

    return {
        "approximation": coeffs[0],
        "cA": coeffs[0],
        "details": coeffs[1:],
        "cD": coeffs[1:],
        "detail": coeffs[1],  # First detail level
        "all_coeffs": coeffs,
    }


def idwt(coeffs: dict | list, wavelet: str = "db4", mode: str = "symmetric") -> np.ndarray:
    """
    Inverse Discrete Wavelet Transform.

    Parameters
    ----------
    coeffs : dict or list
        Wavelet coefficients from dwt()
    wavelet : str
        Wavelet name
    mode : str
        Signal extension mode

    Returns
    -------
    ndarray
        Reconstructed signal
    """
    if isinstance(coeffs, dict):
        if "all_coeffs" in coeffs:
            coeffs_list = coeffs["all_coeffs"]
        else:
            coeffs_list = [coeffs["cA"]] + list(coeffs["cD"])
    else:
        coeffs_list = coeffs

    reconstructed = pywt.waverec(coeffs_list, wavelet, mode=mode)

    return reconstructed


def wavelet_coherence(
    x: np.ndarray,
    y: np.ndarray,
    wavelet: str = "morl",
    scales: np.ndarray | None = None,
    sampling_period: float = 1.0,
) -> dict:
    """
    Wavelet coherence between two signals.

    Parameters
    ----------
    x, y : array_like
        Input signals
    wavelet : str
        Wavelet name
    scales : array_like, optional
        Scales to use
    sampling_period : float
        Sampling period

    Returns
    -------
    dict
        Dictionary with coherence, phase, scales
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if scales is None:
        scales = np.arange(1, min(len(x) // 2, 128))

    # CWT of both signals
    cwt_x = cwt(x, wavelet, scales, sampling_period)
    cwt_y = cwt(y, wavelet, scales, sampling_period)

    coeff_x = cwt_x["coefficients"]
    coeff_y = cwt_y["coefficients"]

    # Cross-wavelet spectrum
    cross_spectrum = coeff_x * np.conj(coeff_y)

    # Auto-spectra
    power_x = np.abs(coeff_x) ** 2
    power_y = np.abs(coeff_y) ** 2

    # Coherence
    coherence = np.abs(cross_spectrum) ** 2 / (power_x * power_y + 1e-10)

    # Phase
    phase = np.angle(cross_spectrum)

    return {"coherence": coherence, "phase": phase, "angle": phase, "scales": scales}


def wavelet_power(
    data: np.ndarray,
    wavelet: str = "morl",
    scales: np.ndarray | None = None,
    sampling_period: float = 1.0,
) -> dict:
    """
    Wavelet power spectrum.

    Parameters
    ----------
    data : array_like
        Input signal
    wavelet : str
        Wavelet name
    scales : array_like, optional
        Scales to use
    sampling_period : float
        Sampling period

    Returns
    -------
    dict
        Dictionary with power, scales, frequencies
    """
    result = cwt(data, wavelet, scales, sampling_period)

    power = np.abs(result["coefficients"]) ** 2

    return {"power": power, "scales": result["scales"], "frequencies": result["frequencies"]}


def wavelet_denoise(
    data: np.ndarray,
    wavelet: str = "db4",
    level: int | None = None,
    threshold_mode: str = "soft",
    threshold_scale: float = 1.0,
) -> np.ndarray:
    """
    Wavelet-based denoising.

    Parameters
    ----------
    data : array_like
        Noisy signal
    wavelet : str
        Wavelet name
    level : int, optional
        Decomposition level
    threshold_mode : str
        'soft' or 'hard' thresholding
    threshold_scale : float
        Threshold scaling factor

    Returns
    -------
    ndarray
        Denoised signal
    """
    data = np.asarray(data)

    if level is None:
        level = pywt.dwt_max_level(len(data), wavelet)

    # Decompose
    coeffs = pywt.wavedec(data, wavelet, level=level)

    # Estimate noise level from finest detail coefficients
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745

    # Universal threshold
    threshold = threshold_scale * sigma * np.sqrt(2 * np.log(len(data)))

    # Threshold detail coefficients
    coeffs_thresh = [coeffs[0]]  # Keep approximation
    for detail in coeffs[1:]:
        if threshold_mode == "soft":
            thresh_detail = pywt.threshold(detail, threshold, mode="soft")
        else:
            thresh_detail = pywt.threshold(detail, threshold, mode="hard")
        coeffs_thresh.append(thresh_detail)

    # Reconstruct
    denoised = pywt.waverec(coeffs_thresh, wavelet)

    # Handle length mismatch
    if len(denoised) > len(data):
        denoised = denoised[: len(data)]

    return denoised
