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

import numpy as np
import matplotlib.pyplot as plt
from earthsciences.data import load_stream_sediments, get_element_stats
from earthsciences.statistics import (
    descriptive_stats,
    detect_outliers,
    test_normality
)

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 70)
print("GEOCHEMISTRY EXAMPLE 1: Exploratory Data Analysis")
print("=" * 70)

# %% Load Data
print("\n1. Loading stream sediment geochemical data...")
print("-" * 70)

# Load copper, zinc, and lead data from stream sediments
elements = ['Cu', 'Zn', 'Pb']
data = load_stream_sediments(elements)

print(f"Loaded {len(data)} stream sediment samples")
print(f"Elements: {elements}")
print(f"\nData shape: {data.shape}")
print(f"Columns: {list(data.columns)}")

# %% Summary Statistics
print("\n\n2. Summary Statistics")
print("-" * 70)

for element in elements:
    print(f"\n{element} Statistics:")
    stats = get_element_stats(element)
    print(f"  Count:   {stats['count']:,}")
    print(f"  Mean:    {stats['mean']:.2f} {stats['units']}")
    print(f"  Median:  {stats['median']:.2f} {stats['units']}")
    print(f"  Std Dev: {stats['std']:.2f} {stats['units']}")
    print(f"  Range:   {stats['min']:.2f} - {stats['max']:.2f} {stats['units']}")
    print(f"  P95:     {stats['p95']:.2f} {stats['units']}")

# %% Detailed Statistics per Element
print("\n\n3. Detailed Univariate Statistics")
print("-" * 70)

for element in elements:
    if element not in data.columns:
        continue
    
    values = data[element].dropna().values
    
    if len(values) < 10:
        print(f"\n{element}: Insufficient data")
        continue
    
    print(f"\n{element}:")
    
    # Calculate descriptive statistics
    desc_stats = descriptive_stats(values)
    print(f"  Mean ± SD: {desc_stats['mean']:.2f} ± {desc_stats['std']:.2f}")
    print(f"  Median: {desc_stats['median']:.2f}")
    print(f"  IQR: {desc_stats['iqr']:.2f}")
    print(f"  Skewness: {desc_stats['skewness']:.3f}")
    print(f"  Kurtosis: {desc_stats['kurtosis']:.3f}")
    
    # Test for normality
    normality = test_normality(values)
    print(f"  Normality test (Shapiro-Wilk):")
    print(f"    p-value: {normality['p_value']:.4f}")
    print(f"    Normal? {'Yes' if normality['is_normal'] else 'No (use log-transform)'}")
    
    # Detect outliers
    outliers = detect_outliers(values, method='iqr')
    n_outliers = np.sum(outliers)
    pct_outliers = 100 * n_outliers / len(values)
    print(f"  Outliers (IQR method): {n_outliers} ({pct_outliers:.1f}%)")

# %% Log-Transform for Lognormal Data
print("\n\n4. Log-Transformed Statistics (for geochemical data)")
print("-" * 70)

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
    
    print(f"\nlog10({element}):")
    log_stats = descriptive_stats(log_values)
    print(f"  Mean ± SD: {log_stats['mean']:.3f} ± {log_stats['std']:.3f}")
    print(f"  Geometric mean: {10**log_stats['mean']:.2f}")
    
    # Test normality on log-transformed data
    log_normality = test_normality(log_values)
    print(f"  Normality of log-data:")
    print(f"    p-value: {log_normality['p_value']:.4f}")
    print(f"    Normal? {'Yes (lognormal distribution)' if log_normality['is_normal'] else 'No'}")

# %% Visualization
print("\n\n5. Creating Visualizations...")
print("-" * 70)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Geochemical Data: Exploratory Analysis', fontsize=16, fontweight='bold')

for idx, element in enumerate(elements):
    if element not in data.columns:
        continue
    
    values = data[element].dropna().values
    values_pos = values[values > 0]
    
    if len(values_pos) < 10:
        continue
    
    # Histogram (linear scale)
    ax1 = axes[0, idx]
    ax1.hist(values_pos, bins=50, edgecolor='black', alpha=0.7)
    ax1.set_xlabel(f'{element} (ppm)', fontsize=10)
    ax1.set_ylabel('Frequency', fontsize=10)
    ax1.set_title(f'{element} Distribution', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Histogram (log scale)
    ax2 = axes[1, idx]
    ax2.hist(values_pos, bins=50, edgecolor='black', alpha=0.7)
    ax2.set_xlabel(f'{element} (ppm)', fontsize=10)
    ax2.set_ylabel('Frequency', fontsize=10)
    ax2.set_title(f'{element} Distribution (log scale)', fontsize=11, fontweight='bold')
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3, which='both')
    
    # Add statistics text
    stats = get_element_stats(element)
    stats_text = f"n={stats['count']:,}\nMedian={stats['median']:.1f}\nP95={stats['p95']:.1f}"
    ax2.text(0.98, 0.97, stats_text, transform=ax2.transAxes,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             fontsize=8)

plt.tight_layout()
plt.savefig('examples/geochem_01_exploratory.png', dpi=150, bbox_inches='tight')
print("✓ Saved: examples/geochem_01_exploratory.png")

# %% Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"\nAnalyzed {len(data)} stream sediment samples for {len(elements)} elements")
print("\nKey Findings:")
print("• Geochemical data typically follows lognormal distribution")
print("• Log-transformation often improves normality")
print("• Outliers may represent mineralization or contamination")
print("• P95 values useful for identifying anomalies")
print("\nNext steps:")
print("• Example 2: Bivariate analysis and element correlations")
print("• Example 3: Spatial analysis and mapping")
print("• Example 4: Multivariate analysis (PCA, clustering)")

plt.show()
