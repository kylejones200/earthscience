"""
Geochemistry Example 1: Exploratory Data Analysis

This example demonstrates basic exploratory analysis of geochemical data
from the Alaska Geochemical Database (AGDB4).

Topics covered:
- Loading geochemical data
- Summary statistics
- Univariate statistics
- Distribution analysis
- Outlier detection
"""

import logging

import matplotlib.pyplot as plt
import numpy as np

from earthsciences.data import get_element_stats, load_stream_sediments
from earthsciences.statistics import descriptive_stats, detect_outliers, test_normality
from earthsciences.utils.logging_config import log_section, setup_logging
from earthsciences.utils.plot_style import use_earthsciences_style

setup_logging()
use_earthsciences_style()
logger = logging.getLogger(__name__)

np.random.seed(42)

log_section("GEOCHEMISTRY EXAMPLE 1: Exploratory Data Analysis")

# %% Load Data
logger.info("\n1. Loading stream sediment geochemical data...")
# Load copper, zinc, and lead data from stream sediments
elements = ["Cu", "Zn", "Pb"]
data = load_stream_sediments(elements)

logger.info(f"Loaded {len(data)} stream sediment samples")
logger.info(f"Elements: {elements}")
logger.info(f"\nData shape: {data.shape}")
logger.info(f"Columns: {list(data.columns)}")

# %% Summary Statistics
logger.info("\n\n2. Summary Statistics")
for element in elements:
    logger.info(f"\n{element} Statistics:")
    stats = get_element_stats(element)
    logger.info(f"  Count:   {stats['count']:,}")
    logger.info(f"  Mean:    {stats['mean']:.2f} {stats['units']}")
    logger.info(f"  Median:  {stats['median']:.2f} {stats['units']}")
    logger.info(f"  Std Dev: {stats['std']:.2f} {stats['units']}")
    logger.info(f"  Range:   {stats['min']:.2f} - {stats['max']:.2f} {stats['units']}")
    logger.info(f"  P95:     {stats['p95']:.2f} {stats['units']}")

# %% Detailed Statistics per Element
logger.info("\n\n3. Detailed Univariate Statistics")
for element in elements:
    if element not in data.columns:
        continue

    values = data[element].dropna().values

    if len(values) < 10:
        logger.info(f"\n{element}: Insufficient data")
        continue

    logger.info(f"\n{element}:")

    # Calculate descriptive statistics
    desc_stats = descriptive_stats(values)
    logger.info(f"  Mean ± SD: {desc_stats['mean']:.2f} ± {desc_stats['std']:.2f}")
    logger.info(f"  Median: {desc_stats['median']:.2f}")
    logger.info(f"  IQR: {desc_stats['iqr']:.2f}")
    logger.info(f"  Skewness: {desc_stats['skewness']:.3f}")
    logger.info(f"  Kurtosis: {desc_stats['kurtosis']:.3f}")

    # Test for normality
    normality = test_normality(values)
    logger.info("  Normality test (Shapiro-Wilk):")
    logger.info(f"    p-value: {normality['p_value']:.4f}")
    logger.info(f"    Normal? {'Yes' if normality['is_normal'] else 'No (use log-transform)'}")

    # Detect outliers
    outliers = detect_outliers(values, method="iqr")
    n_outliers = np.sum(outliers)
    pct_outliers = 100 * n_outliers / len(values)
    logger.info(f"  Outliers (IQR method): {n_outliers} ({pct_outliers:.1f}%)")

# %% Log-Transform for Lognormal Data
logger.info("\n\n4. Log-Transformed Statistics (for geochemical data)")
for element in elements:
    if element not in data.columns:
        continue

    values = data[element].dropna().values

    # Remove zeros and negatives
    values_pos = values[values > 0]

    if len(values_pos) < 10:
        continue

    # Log-transform
    log_values = np.log10(values_pos)

    logger.info(f"\nlog10({element}):")
    log_stats = descriptive_stats(log_values)
    logger.info(f"  Mean ± SD: {log_stats['mean']:.3f} ± {log_stats['std']:.3f}")
    logger.info(f"  Geometric mean: {10**log_stats['mean']:.2f}")

    # Test normality on log-transformed data
    log_normality = test_normality(log_values)
    logger.info("  Normality of log-data:")
    logger.info(f"    p-value: {log_normality['p_value']:.4f}")
    logger.info(
        f"    Normal? {'Yes (lognormal distribution)' if log_normality['is_normal'] else 'No'}"
    )

# %% Visualization
logger.info("\n\n5. Creating Visualizations...")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle("Geochemical Data: Exploratory Analysis", fontsize=16, fontweight="bold")

for idx, element in enumerate(elements):
    if element not in data.columns:
        continue

    values = data[element].dropna().values
    values_pos = values[values > 0]

    if len(values_pos) < 10:
        continue

    # Histogram (linear scale)
    ax1 = axes[0, idx]
    ax1.hist(values_pos, bins=50, edgecolor="black", alpha=0.7)
    ax1.set_xlabel(f"{element} (ppm)", fontsize=10)
    ax1.set_ylabel("Frequency", fontsize=10)
    ax1.set_title(f"{element} Distribution", fontsize=11, fontweight="bold")

    # Histogram (log scale)
    ax2 = axes[1, idx]
    ax2.hist(values_pos, bins=50, edgecolor="black", alpha=0.7)
    ax2.set_xlabel(f"{element} (ppm)", fontsize=10)
    ax2.set_ylabel("Frequency", fontsize=10)
    ax2.set_title(f"{element} Distribution (log scale)", fontsize=11, fontweight="bold")
    ax2.set_xscale("log")

    # Add statistics text
    stats = get_element_stats(element)
    stats_text = f"n={stats['count']:,}\nMedian={stats['median']:.1f}\nP95={stats['p95']:.1f}"
    ax2.text(
        0.98,
        0.97,
        stats_text,
        transform=ax2.transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        fontsize=8,
    )

plt.tight_layout()
plt.savefig("examples/geochem_01_exploratory.png", dpi=150, bbox_inches="tight")
logger.info("✓ Saved: examples/geochem_01_exploratory.png")

# %% Summary
log_section("SUMMARY")
logger.info(f"\nAnalyzed {len(data)} stream sediment samples for {len(elements)} elements")
logger.info("\nKey Findings:")
logger.info("• Geochemical data typically follows lognormal distribution")
logger.info("• Log-transformation often improves normality")
logger.info("• Outliers may represent mineralization or contamination")
logger.info("• P95 values useful for identifying anomalies")
logger.info("\nNext steps:")
logger.info("• Example 2: Bivariate analysis and element correlations")
logger.info("• Example 3: Spatial analysis and mapping")
logger.info("• Example 4: Multivariate analysis (PCA, clustering)")

plt.show()
