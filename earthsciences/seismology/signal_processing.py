"""
Seismic signal processing

Phase picking, receiver functions, and waveform analysis.
"""

import numpy as np
from scipy.signal import find_peaks
from typing import Dict
import warnings


def phase_picking(seismogram: np.ndarray, dt: float, method: str = 'sta_lta',
                 sta_window: float = 0.5, lta_window: float = 5.0,
                 threshold: float = 3.0) -> Dict:
    """
    Automatic phase picking using STA/LTA or other methods.
    """
    seismogram = np.asarray(seismogram)
    
    if method == 'sta_lta':
        from scipy.ndimage import uniform_filter1d
        
        # Calculate characteristic function
        cf = seismogram ** 2
        
        # STA and LTA
        sta_samples = int(sta_window / dt)
        lta_samples = int(lta_window / dt)
        
        sta = uniform_filter1d(cf, sta_samples, mode='constant')
        lta = uniform_filter1d(cf, lta_samples, mode='constant')
        
        # STA/LTA ratio
        sta_lta = sta / (lta + 1e-10)
        
        # Find peaks above threshold
        peaks, properties = find_peaks(sta_lta, height=threshold)
        
        if len(peaks) > 0:
            pick_sample = peaks[0]
            pick_time = pick_sample * dt
            sta_lta_max = sta_lta[pick_sample]
        else:
            pick_time = np.nan
            sta_lta_max = np.max(sta_lta)
        
        return {
            'pick_time': pick_time,
            'sta_lta': sta_lta,
            'sta_lta_max': sta_lta_max,
            'method': 'sta_lta',
            'threshold': threshold,
        }
        
    elif method == 'ar_aic':
        # AIC picker
        n = len(seismogram)
        aic = np.zeros(n)
        
        for k in range(10, n-10):
            var1 = np.var(seismogram[:k])
            var2 = np.var(seismogram[k:])
            aic[k] = k * np.log(var1) + (n-k) * np.log(var2)
        
        pick_sample = np.argmin(aic[10:-10]) + 10
        pick_time = pick_sample * dt
        
        return {
            'pick_time': pick_time,
            'aic': aic,
            'method': 'ar_aic',
        }
        
    elif method == 'kurtosis':
        from scipy.stats import kurtosis as kurt
        
        window_samples = int(1.0 / dt)
        
        kurtosis = np.zeros(len(seismogram))
        for i in range(window_samples, len(seismogram)):
            window = seismogram[i-window_samples:i]
            kurtosis[i] = kurt(window)
        
        kurtosis_diff = np.diff(kurtosis)
        pick_sample = np.argmax(np.abs(kurtosis_diff)) + window_samples
        pick_time = pick_sample * dt
        
        return {
            'pick_time': pick_time,
            'kurtosis': kurtosis,
            'method': 'kurtosis',
        }
        
    else:
        raise ValueError(f"Unknown method: {method}")


def receiver_function(seismogram: np.ndarray, dt: float,
                     method: str = 'frequency', gaussian_width: float = 2.5) -> Dict:
    """
    Calculate receiver function from teleseismic P waves.
    
    Isolates P-to-S conversions at discontinuities beneath station.
    """
    seismogram = np.asarray(seismogram)
    
    if seismogram.ndim == 1:
        raise ValueError("Seismogram must have at least 2 components (Z, R)")
    
    Z = seismogram[:, 0]  # Vertical
    R = seismogram[:, 1]  # Radial
    
    if method == 'frequency':
        from scipy.fft import fft, ifft, fftfreq
        
        # FFT
        Z_fft = fft(Z)
        R_fft = fft(R)
        
        # Water level deconvolution
        water_level = 0.01 * np.max(np.abs(Z_fft))
        denominator = Z_fft * np.conj(Z_fft)
        denominator = np.maximum(denominator, water_level)
        
        RF_fft = R_fft * np.conj(Z_fft) / denominator
        
        # Gaussian filter
        freqs = fftfreq(len(Z), dt)
        gaussian_filter = np.exp(-(freqs * gaussian_width)**2)
        RF_fft *= gaussian_filter
        
        # IFFT
        RF = np.real(ifft(RF_fft))
        
    elif method == 'time':
        from scipy.signal import correlate
        
        # Normalize
        Z_norm = Z / np.max(np.abs(Z))
        R_norm = R / np.max(np.abs(R))
        
        # Cross-correlation as approximation
        RF = correlate(R_norm, Z_norm, mode='same')
        RF = RF / np.max(np.abs(RF))
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Time axis
    time = np.arange(len(RF)) * dt - len(RF) * dt / 2
    
    return {
        'receiver_function': RF,
        'time': time,
        'method': method,
        'dt': dt,
    }
