"""
Geochemistry Example 2: Bivariate Analysis and Correlations

This example demonstrates bivariate relationships between elements,
correlation analysis, and scatter plot matrices.

Topics covered:
- Element correlations
- Scatter plots
- Linear regression
- Log-log plots
- Association indices
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from earthsciences.data import load_stream_sediments
from earthsciences.statistics import (
    correlation,
    linear_regression,
    bootstrap_confidence_interval
)

np.random.seed(42)

print("=" * 70)
print("GEOCHEMISTRY EXAMPLE 2: Bivariate Analysis")
print("=" * 70)

# %% Load Data
print("\n1. Loading data...")
print("-" * 70)

# Load ore-related elements
elements = ['Cu', 'Pb', 'Zn', 'Ag', 'Au', 'Mo']
data = load_stream_sediments(elements)

print(f"Loaded {len(data)} samples with {len(elements)} elements")

# Remove samples with too many missing values
data_clean = data.dropna(subset=elements, thresh=4)  # Require at least 4 elements
print(f"After filtering: {len(data_clean)} samples")

# %% Correlation Matrix
print("\n\n2. Correlation Analysis")
print("-" * 70)

# Calculate correlations (Pearson and Spearman)
print("\nPearson Correlations (linear relationships):")
print("-" * 70)

correlation_matrix = np.zeros((len(elements), len(elements)))
for i, elem1 in enumerate(elements):
    for j, elem2 in enumerate(elements):
        if elem1 not in data_clean.columns or elem2 not in data_clean.columns:
            correlation_matrix[i, j] = np.nan
            continue
        
        # Get common non-NaN values
        mask = data_clean[elem1].notna() & data_clean[elem2].notna()
        if mask.sum() < 10:
            correlation_matrix[i, j] = np.nan
            continue
        
        corr_result = correlation(
            data_clean.loc[mask, elem1].values,
            data_clean.loc[mask, elem2].values,
            method='pearson'
        )
        correlation_matrix[i, j] = corr_result['r']

# Print correlation matrix
print("\n     ", end="")
for elem in elements:
    print(f"{elem:>7}", end="")
print()

for i, elem1 in enumerate(elements):
    print(f"{elem1:>4}", end=" ")
    for j, elem2 in enumerate(elements):
        if np.isnan(correlation_matrix[i, j]):
            print(f"   {'--':>5}", end="")
        else:
            print(f"  {correlation_matrix[i, j]:>5.2f}", end="")
    print()

# %% Strong Correlations
print("\n\nStrong Correlations (|r| > 0.5):")
print("-" * 70)

strong_corrs = []
for i in range(len(elements)):
    for j in range(i+1, len(elements)):
        if abs(correlation_matrix[i, j]) > 0.5 and not np.isnan(correlation_matrix[i, j]):
            strong_corrs.append((elements[i], elements[j], correlation_matrix[i, j]))

strong_corrs.sort(key=lambda x: abs(x[2]), reverse=True)

for elem1, elem2, r in strong_corrs:
    print(f"{elem1:>4} - {elem2:<4}: r = {r:>6.3f}")

# %% Detailed Analysis: Cu-Zn Relationship
print("\n\n3. Detailed Analysis: Cu-Zn Relationship")
print("-" * 70)

if 'Cu' in data_clean.columns and 'Zn' in data_clean.columns:
    # Get clean data
    mask = data_clean['Cu'].notna() & data_clean['Zn'].notna()
    cu = data_clean.loc[mask, 'Cu'].values
    zn = data_clean.loc[mask, 'Zn'].values
    
    # Remove zeros for log-transform
    mask_pos = (cu > 0) & (zn > 0)
    cu_pos = cu[mask_pos]
    zn_pos = zn[mask_pos]
    
    print(f"Sample size: {len(cu_pos)}")
    
    # Linear correlation
    corr_result = correlation(cu_pos, zn_pos, method='pearson')
    print(f"\nPearson correlation: r = {corr_result['r']:.3f}")
    print(f"p-value: {corr_result['p_value']:.4e}")
    print(f"Significant? {'Yes' if corr_result['significant'] else 'No'}")
    
    # Spearman correlation (rank-based, better for non-linear)
    spear_result = correlation(cu_pos, zn_pos, method='spearman')
    print(f"\nSpearman correlation: ρ = {spear_result['r']:.3f}")
    print(f"(rank-based, less sensitive to outliers)")
    
    # Linear regression
    reg_result = linear_regression(cu_pos, zn_pos)
    print(f"\nLinear Regression: Zn = {reg_result['intercept']:.2f} + {reg_result['slope']:.4f} * Cu")
    print(f"R² = {reg_result['r_squared']:.3f}")
    print(f"p-value: {reg_result['p_value']:.4e}")
    
    # Bootstrap confidence intervals
    print("\nBootstrap 95% Confidence Intervals:")
    slope_ci = bootstrap_confidence_interval(
        cu_pos, zn_pos,
        statistic=lambda x, y: linear_regression(x, y)['slope'],
        n_iterations=1000,
        confidence=0.95
    )
    print(f"Slope: [{slope_ci['lower']:.4f}, {slope_ci['upper']:.4f}]")

# %% Visualization: Scatter Plot Matrix
print("\n\n4. Creating Visualizations...")
print("-" * 70)

# Select subset of elements for clarity
plot_elements = ['Cu', 'Pb', 'Zn', 'Mo']
plot_data = data_clean[[e for e in plot_elements if e in data_clean.columns]].copy()

# Remove zeros and take log10
for col in plot_data.columns:
    plot_data[col] = plot_data[col].replace(0, np.nan)
    plot_data[f'log_{col}'] = np.log10(plot_data[col])

# Create scatter plot matrix
fig = plt.figure(figsize=(14, 14))
fig.suptitle('Geochemical Scatter Plot Matrix (log₁₀ scale)', 
             fontsize=16, fontweight='bold', y=0.995)

n = len([e for e in plot_elements if e in data_clean.columns])
log_cols = [f'log_{e}' for e in plot_elements if e in data_clean.columns]

for i, col1 in enumerate(log_cols):
    for j, col2 in enumerate(log_cols):
        ax = plt.subplot(n, n, i*n + j + 1)
        
        if i == j:
            # Diagonal: histograms
            data_plot = plot_data[col1].dropna()
            ax.hist(data_plot, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
            ax.set_ylabel('Frequency' if j == 0 else '', fontsize=9)
            if i == n-1:
                elem = col1.replace('log_', '')
                ax.set_xlabel(f'log₁₀({elem})', fontsize=10, fontweight='bold')
        else:
            # Off-diagonal: scatter plots
            mask = plot_data[col1].notna() & plot_data[col2].notna()
            x = plot_data.loc[mask, col2].values
            y = plot_data.loc[mask, col1].values
            
            ax.scatter(x, y, alpha=0.3, s=10, c='steelblue', edgecolors='none')
            
            # Add regression line if significant correlation
            if len(x) > 10:
                try:
                    reg = linear_regression(x, y)
                    if reg['p_value'] < 0.01:  # Significant at 1%
                        x_line = np.linspace(x.min(), x.max(), 100)
                        y_line = reg['intercept'] + reg['slope'] * x_line
                        ax.plot(x_line, y_line, 'r-', linewidth=1.5, alpha=0.7)
                        
                        # Add R² text
                        ax.text(0.05, 0.95, f"R²={reg['r_squared']:.2f}",
                               transform=ax.transAxes, fontsize=8,
                               verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
                except:
                    pass
            
            if j == 0:
                elem = col1.replace('log_', '')
                ax.set_ylabel(f'log₁₀({elem})', fontsize=10, fontweight='bold')
            else:
                ax.set_ylabel('', fontsize=9)
            
            if i == n-1:
                elem = col2.replace('log_', '')
                ax.set_xlabel(f'log₁₀({elem})', fontsize=10, fontweight='bold')
            else:
                ax.set_xlabel('', fontsize=9)
        
        ax.grid(True, alpha=0.3)
        if i != n-1:
            ax.set_xticklabels([])
        if j != 0:
            ax.set_yticklabels([])

plt.tight_layout()
plt.savefig('examples/geochem_02_bivariate.png', dpi=150, bbox_inches='tight')
print("✓ Saved: examples/geochem_02_bivariate.png")

# %% Correlation Heatmap
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Pearson Correlation (r)', rotation=270, labelpad=20, fontsize=11)

# Set ticks and labels
ax.set_xticks(np.arange(len(elements)))
ax.set_yticks(np.arange(len(elements)))
ax.set_xticklabels(elements, fontsize=11)
ax.set_yticklabels(elements, fontsize=11)

# Add correlation values
for i in range(len(elements)):
    for j in range(len(elements)):
        if not np.isnan(correlation_matrix[i, j]):
            text = ax.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=9)

ax.set_title('Element Correlation Matrix', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('examples/geochem_02_correlation_matrix.png', dpi=150, bbox_inches='tight')
print("✓ Saved: examples/geochem_02_correlation_matrix.png")

# %% Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("\nCorrelation analysis reveals element associations:")
print("• Strong correlations suggest common source or similar behavior")
print("• Cu-Zn often correlate in porphyry and VMS deposits")
print("• Log-transformation improves linear relationships")
print("• Rank correlations (Spearman) handle outliers better")
print("\nNext steps:")
print("• Example 3: Spatial analysis and geochemical mapping")
print("• Example 4: Multivariate analysis (PCA, clustering)")

plt.show()
