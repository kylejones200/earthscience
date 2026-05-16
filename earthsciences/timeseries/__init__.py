"""
Time-series analysis and signal processing (Chapters 5-6)

Spectral analysis, filtering, and time-series methods for earth sciences.
"""

from .changepoint import *
from .filtering import *
from .nonlinear import *
from .signal_processing import *
from .signals import *
from .spectral import autocorrelation as spectral_autocorrelation
from .spectral import (
    blackman_tukey,
    coherence,
    cross_spectrum,
    evolutionary_spectrum,
    lomb_scargle,
    multitaper_coherence,
    multitaper_spectrogram,
    multitaper_spectrum,
    power_spectrum,
    spectrogram,
)
from .wavelets import *

__all__ = [
    # Spectral analysis
    "power_spectrum",
    "blackman_tukey",
    "lomb_scargle",
    "spectrogram",
    "coherence",
    "cross_spectrum",
    "evolutionary_spectrum",
    "multitaper_coherence",
    "multitaper_spectrogram",
    "multitaper_spectrum",
    # Filtering
    "butter_filter",
    "fir_filter",
    "moving_average_filter",
    "savitzky_golay",
    # Wavelets
    "wavelet_transform",
    "wavelet_coherence",
    # Signal generation and utilities
    "generate_test_signal",
    "detrend",
    "normalize",
    # Nonlinear analysis
    "time_delay_embedding",
    "recurrence_plot",
    "lyapunov_exponent",
    # Change point detection
    "cusum",
    "pettitt_test",
    "mann_kendall_test",
    "spectral_autocorrelation",
]
