#!/usr/bin/env python3
"""
Example 7: Time-Series Analysis - Seismic, Climate, and Signals

Demonstrates:
- Signal generation and filtering
- Spectral analysis (FFT, Welch, multitaper)
- Wavelet transforms
- Trend analysis and detrending
- Cross-correlation and coherence
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
from bootstrap import TUTORIAL_DIR
from plot_utils import clean_plot_style

from earthsciences.timeseries import (
    bandpass_filter,
    cwt,
    highpass_filter,
    lomb_scargle,
    lowpass_filter,
    multitaper_spectrum,
    power_spectrum,
)
from earthsciences.utils.logging_config import log_section, setup_logging

setup_logging()
logger = logging.getLogger(__name__)


logger.info("TIME-SERIES ANALYSIS EXAMPLES")


# =============================================================================
# Example 1: Signal Filtering - Seismic Data
# =============================================================================
log_section("Example 1: Seismic Signal Filtering")

# Generate synthetic seismic signal
# Low frequency surface waves + high frequency body waves + noise
fs = 100  # Sampling rate (Hz)
duration = 10  # seconds
t = np.linspace(0, duration, int(fs * duration))

# Surface waves (0.5-2 Hz)
surface_waves = 2 * np.sin(2 * np.pi * 0.8 * t) * np.exp(-0.3 * t)

# Body waves (5-15 Hz)
body_waves = np.sin(2 * np.pi * 10 * t) * np.exp(-1.5 * (t - 2) ** 2)

# Ambient noise
noise = np.random.normal(0, 0.3, len(t))

# Combined signal
seismic_signal = surface_waves + body_waves + noise

logger.info("\nSeismic signal characteristics:")
logger.info(f"  Duration: {duration} seconds")
logger.info(f"  Sampling rate: {fs} Hz")
logger.info(f"  Data points: {len(t)}")

# Apply different filters
lowpass = lowpass_filter(seismic_signal, cutoff=3, sampling_rate=fs)
highpass = highpass_filter(seismic_signal, cutoff=5, sampling_rate=fs)
bandpass = bandpass_filter(seismic_signal, lowcut=0.5, highcut=2, sampling_rate=fs)

logger.info("\nFilters applied:")
logger.info("  Lowpass (< 3 Hz): Isolates surface waves")
logger.info("  Highpass (> 5 Hz): Isolates body waves")
logger.info("  Bandpass (0.5-2 Hz): Targets Love/Rayleigh waves")

# Plot results
fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

# Original signal
axes[0].plot(t, seismic_signal, "#424242", linewidth=1, alpha=0.8)
axes[0].set_title(
    "Synthetic Seismic Signal: Surface + Body Waves + Noise", fontsize=12, fontweight="bold", pad=12
)
clean_plot_style(axes[0])

# Lowpass filtered
axes[1].plot(t, lowpass, "#1976D2", linewidth=1.5)
axes[1].set_title(
    "Lowpass Filtered < 3 Hz (Surface Waves Isolated)", fontsize=12, fontweight="bold", pad=12
)
clean_plot_style(axes[1])

# Highpass filtered
axes[2].plot(t, highpass, "#D32F2F", linewidth=1.5)
axes[2].set_title(
    "Highpass Filtered > 5 Hz (Body Waves Isolated)", fontsize=12, fontweight="bold", pad=12
)
clean_plot_style(axes[2])

# Bandpass filtered
axes[3].plot(t, bandpass, "#388E3C", linewidth=1.5)
axes[3].set_title(
    "Bandpass Filtered 0.5-2 Hz (Love/Rayleigh Waves)", fontsize=12, fontweight="bold", pad=12
)
clean_plot_style(axes[3])


plt.tight_layout()
plt.savefig(TUTORIAL_DIR / "07_timeseries_filtering.png", dpi=300, bbox_inches="tight")
logger.info("\n✓ Saved filtering example")
plt.close()

# =============================================================================
# Example 2: Spectral Analysis - Climate Oscillations
# =============================================================================
log_section("Example 2: Spectral Analysis of Climate Data")

# Simulate climate time series with multiple periodicities
# Annual cycle, ENSO (3-7 years), decadal variability
years = 100
t_climate = np.linspace(0, years, years * 12)  # Monthly data

# Annual cycle
annual = 2 * np.sin(2 * np.pi * t_climate)

# ENSO (~4 year period)
enso = 1.5 * np.sin(2 * np.pi * t_climate / 4 + np.random.uniform(0, 2 * np.pi))

# Decadal variability (~20 year period)
decadal = 1.0 * np.sin(2 * np.pi * t_climate / 20)

# Trend
trend = 0.01 * t_climate

# Noise
climate_noise = np.random.normal(0, 0.5, len(t_climate))

# Combined climate signal
climate_signal = annual + enso + decadal + trend + climate_noise

logger.info("\nClimate time series:")
logger.info(f"  Duration: {years} years")
logger.info("  Resolution: monthly")
logger.info(f"  Data points: {len(t_climate)}")

# Compute power spectra using different methods
# Welch method
freqs_welch, power_welch = power_spectrum(climate_signal, dt=1 / 12, method="welch")

# Multitaper method (better for identifying peaks)
freqs_mt, power_mt = multitaper_spectrum(climate_signal, dt=1 / 12, NW=4, k=7)

# Convert frequencies to periods
periods_welch = 1 / freqs_welch[1:]  # Exclude DC component
periods_mt = 1 / freqs_mt[1:]

logger.info("\nSpectral analysis:")
logger.info("  Method 1: Welch periodogram")
logger.info("  Method 2: Multitaper spectrum (optimal for line spectra)")

# Find peaks
peak_idx_mt = np.argsort(power_mt[1:])[-3:]  # Top 3 peaks (excluding DC)
peak_periods = periods_mt[peak_idx_mt]
logger.info("\nDetected periodicities:")
for i, period in enumerate(sorted(peak_periods, reverse=True)):
    logger.info(f"  {i+1}. {period:.1f} years")

# Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# Time series
ax1.plot(t_climate, climate_signal, "#424242", linewidth=0.8, alpha=0.7)
ax1.plot(
    t_climate, trend, "#D32F2F", linestyle="--", linewidth=2.5, label="Linear trend", alpha=0.8
)
ax1.set_title(
    "Simulated Climate Anomaly (°C) Over 100 Years (Monthly Data)",
    fontsize=13,
    fontweight="bold",
    pad=15,
)
ax1.legend(frameon=False, fontsize=11)
ax1.set_xlim(0, years)
clean_plot_style(ax1)

# Power spectrum
ax2.loglog(periods_welch, power_welch[1:], "#1976D2", alpha=0.6, linewidth=2, label="Welch method")
ax2.loglog(periods_mt, power_mt[1:], "#D32F2F", linewidth=2.5, label="Multitaper method")

# Mark expected peaks
for period, label in [(1, "Annual"), (4, "ENSO"), (20, "Decadal")]:
    ax2.axvline(period, color="#388E3C", linestyle="--", alpha=0.7, linewidth=2)
    ax2.text(
        period * 1.1,
        ax2.get_ylim()[1] * 0.7,
        label,
        rotation=0,
        va="center",
        ha="left",
        fontsize=10,
        color="#388E3C",
    )

ax2.set_title("Power Spectrum: Power vs Period (years)", fontsize=13, fontweight="bold", pad=15)
ax2.legend(frameon=False, fontsize=11)
ax2.set_xlim(0.5, 50)
clean_plot_style(ax2, remove_grid=False)

plt.tight_layout()
plt.savefig(TUTORIAL_DIR / "07_timeseries_spectral.png", dpi=300, bbox_inches="tight")
logger.info("\n✓ Saved spectral analysis example")
plt.close()

# =============================================================================
# Example 3: Wavelet Analysis - Non-Stationary Signal
# =============================================================================
log_section("Example 3: Wavelet Analysis - Time-Frequency Decomposition")

# Create non-stationary signal (frequency changes over time)
t_wavelet = np.linspace(0, 10, 1000)
fs_wavelet = 100

# Frequency chirp: starts at 5 Hz, increases to 20 Hz
instantaneous_freq = 5 + 1.5 * t_wavelet
phase = 2 * np.pi * np.cumsum(instantaneous_freq) / fs_wavelet
chirp_signal = np.sin(phase)

# Add a transient event
transient = 2 * np.exp(-((t_wavelet - 5) ** 2) / 0.1) * np.sin(2 * np.pi * 15 * t_wavelet)

# Combined signal
wavelet_signal = chirp_signal + transient + np.random.normal(0, 0.1, len(t_wavelet))

logger.info("\nNon-stationary signal:")
logger.info("  Chirp: 5 → 20 Hz over 10 seconds")
logger.info("  Transient: 15 Hz burst at t=5s")
logger.info(f"  Sampling rate: {fs_wavelet} Hz")

# Continuous Wavelet Transform
scales = np.arange(1, 128)
cwt_result = cwt(wavelet_signal, wavelet="morl", scales=scales, sampling_period=1 / fs_wavelet)
coefficients = cwt_result["coefficients"]
frequencies = cwt_result["frequencies"]

logger.info("\nCWT analysis:")
logger.info("  Wavelet: Morlet")
logger.info(f"  Scales: {len(scales)}")
logger.info(f"  Frequency range: {frequencies.min():.1f} - {frequencies.max():.1f} Hz")

# Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# Time series
ax1.plot(t_wavelet, wavelet_signal, "#424242", linewidth=1.2)
ax1.set_title(
    "Non-Stationary Signal: Chirp (Increasing Frequency) + Transient Pulse",
    fontsize=13,
    fontweight="bold",
    pad=15,
)
ax1.set_xlim(0, 10)
clean_plot_style(ax1)

# Scalogram (time-frequency representation)
T, F = np.meshgrid(t_wavelet, frequencies)
im = ax2.contourf(T, F, np.abs(coefficients), levels=50, cmap="viridis")
ax2.set_title(
    "Wavelet Scalogram: Frequency (Hz) vs Time (seconds)", fontsize=13, fontweight="bold", pad=15
)
ax2.set_ylim(0, 25)
cbar = plt.colorbar(im, ax=ax2)
cbar.set_label("Magnitude", fontsize=10)
cbar.outline.set_linewidth(0.8)
clean_plot_style(ax2)

plt.tight_layout()
plt.savefig(TUTORIAL_DIR / "07_timeseries_wavelet.png", dpi=300, bbox_inches="tight")
logger.info("\n✓ Saved wavelet analysis example")
plt.close()

# =============================================================================
# Example 4: Lomb-Scargle - Irregularly Sampled Data
# =============================================================================
log_section("Example 4: Lomb-Scargle for Irregular Sampling")

# Simulate irregularly sampled data (e.g., astronomical observations)
t_regular = np.linspace(0, 100, 500)

# Create irregular sampling (remove random points)
np.random.seed(42)
keep_fraction = 0.6
keep_indices = np.random.choice(len(t_regular), int(len(t_regular) * keep_fraction), replace=False)
t_irregular = np.sort(t_regular[keep_indices])

# Signal with two periodicities
true_periods = [10, 25]  # days
signal_irregular = (
    np.sin(2 * np.pi * t_irregular / true_periods[0])
    + 0.7 * np.sin(2 * np.pi * t_irregular / true_periods[1])
    + np.random.normal(0, 0.3, len(t_irregular))
)

logger.info("\nIrregularly sampled data:")
logger.info("  Total timespan: 100 days")
logger.info("  Regular grid: 500 points")
logger.info(f"  Irregular sample: {len(t_irregular)} points ({keep_fraction*100:.0f}%)")
logger.info(f"  True periods: {true_periods} days")

# Lomb-Scargle periodogram
frequencies = np.linspace(0.01, 0.5, 1000)  # 0.01 to 0.5 cycles/day
freqs_ls, power_ls = lomb_scargle(t_irregular, signal_irregular, frequencies=frequencies)

periods_ls = 1 / freqs_ls

# Find peaks
from scipy.signal import find_peaks

peaks, properties = find_peaks(power_ls, height=0.3, distance=20)
detected_periods = periods_ls[peaks]

logger.info("\nLomb-Scargle analysis:")
logger.info(f"  Detected {len(peaks)} significant peaks")
for i, period in enumerate(sorted(detected_periods, reverse=True)):
    logger.info(f"  Peak {i+1}: {period:.1f} days")

# Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# Time series
ax1.plot(
    t_irregular, signal_irregular, "o", color="#1976D2", markersize=5, alpha=0.7, markeredgewidth=0
)
ax1.set_title(
    "Irregularly Sampled Time Series (60% of Regular Grid)", fontsize=13, fontweight="bold", pad=15
)
clean_plot_style(ax1)

# Lomb-Scargle periodogram
ax2.plot(periods_ls, power_ls, "#1976D2", linewidth=2)
ax2.plot(
    periods_ls[peaks],
    power_ls[peaks],
    "o",
    color="#D32F2F",
    markersize=12,
    label="Detected peaks",
    zorder=5,
    markeredgecolor="black",
    markeredgewidth=1,
)

# Mark true periods
for period in true_periods:
    ax2.axvline(period, color="#388E3C", linestyle="--", linewidth=2.5, alpha=0.7)
    ax2.text(
        period,
        ax2.get_ylim()[1] * 0.9,
        f"{period}d true",
        ha="center",
        fontsize=10,
        fontweight="bold",
        color="#388E3C",
    )

ax2.set_title(
    "Lomb-Scargle Periodogram: Power vs Period (days)", fontsize=13, fontweight="bold", pad=15
)
ax2.legend(frameon=False, fontsize=11)
ax2.set_xlim(2, 50)
clean_plot_style(ax2)

plt.tight_layout()
plt.savefig(TUTORIAL_DIR / "07_timeseries_lombscargle.png", dpi=300, bbox_inches="tight")
logger.info("\n✓ Saved Lomb-Scargle example")
plt.close()

# =============================================================================
# Summary
# =============================================================================
log_section("SUMMARY")

logger.info(
    """
Generated plots:
  1. 07_timeseries_filtering.png - Seismic signal filtering
  2. 07_timeseries_spectral.png - Climate spectral analysis
  3. 07_timeseries_wavelet.png - Time-frequency analysis
  4. 07_timeseries_lombscargle.png - Irregular sampling

Key concepts demonstrated:
  • Digital filtering (lowpass, highpass, bandpass)
  • Spectral analysis (Welch, multitaper methods)
  • Wavelet transforms for non-stationary signals
  • Lomb-Scargle for irregularly sampled data
  • Time-frequency representation

Applications:
  - Seismology (earthquake signals, surface waves)
  - Climatology (ENSO, PDO, orbital forcing)
  - Paleoclimatology (ice cores, sediment records)
  - Astronomy (variable stars, exoplanet detection)
  - Hydrology (stream flow, groundwater levels)
  - Geophysics (magnetic variations, tides)
"""
)

logger.info("✓ All time-series examples complete!")
