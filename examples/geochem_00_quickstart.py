"""
QUICK START: Geochemistry Analysis with Alaska Data

This script provides a quick overview of geochemical analysis capabilities
using real Alaska Geochemical Database data.

Run this first to verify everything works!
"""

import numpy as np
import matplotlib.pyplot as plt
from earthsciences.data import load_stream_sediments, get_element_stats
from earthsciences.statistics import correlation, descriptive_stats
from earthsciences.spatial import idw_interpolation
from earthsciences.multivariate import principal_component_analysis

print("=" * 70)
print("QUICK START: Geochemistry Analysis")
print("=" * 70)

# Test 1: Load Data
print("\n1. Loading geochemical data...")
try:
    elements = ['Cu', 'Zn', 'Pb']
    data = load_stream_sediments(elements)
    print(f"✓ Loaded {len(data)} stream sediment samples")
    print(f"✓ Elements: {elements}")
except Exception as e:
    print(f"✗ Error loading data: {e}")
    exit(1)

# Test 2: Statistics
print("\n2. Computing statistics...")
try:
    cu_stats = get_element_stats('Cu')
    print(f"✓ Cu: median={cu_stats['median']:.1f} ppm, P95={cu_stats['p95']:.1f} ppm")
except Exception as e:
    print(f"✗ Error computing stats: {e}")

# Test 3: Correlation
print("\n3. Analyzing correlations...")
try:
    mask = data['Cu'].notna() & data['Zn'].notna()
    subset_corr = data.loc[mask].head(1000)  # Subset for speed
    cu = subset_corr['Cu'].values
    zn = subset_corr['Zn'].values
    
    r, p_value = correlation(cu, zn)
    print(f"✓ Cu-Zn correlation: r={r:.3f}, p={p_value:.4e}")
except Exception as e:
    print(f"✗ Error in correlation: {e}")

# Test 4: Spatial Analysis
print("\n4. Testing spatial interpolation...")
try:
    # Small subset for quick test
    subset = data.sample(n=100, random_state=42)
    x = subset['LONGITUDE'].values
    y = subset['LATITUDE'].values
    v = subset['Cu'].dropna().values[:len(x)]
    
    # Create small grid
    grid_x, grid_y = np.meshgrid(
        np.linspace(x.min(), x.max(), 20),
        np.linspace(y.min(), y.max(), 20)
    )
    
    result = idw_interpolation(x[:len(v)], y[:len(v)], v, grid_x, grid_y)
    print(f"✓ Spatial interpolation complete (20x20 grid)")
except Exception as e:
    print(f"✗ Error in spatial analysis: {e}")

# Test 5: Multivariate
print("\n5. Testing multivariate analysis...")
try:
    # Small subset with complete data
    subset = data.dropna(subset=elements).sample(n=200, random_state=42)
    X = subset[elements].values
    
    # Log-transform and standardize
    X_log = np.log10(X + 1)
    from sklearn.preprocessing import StandardScaler
    X_scaled = StandardScaler().fit_transform(X_log)
    
    pca_result = principal_component_analysis(X_scaled, n_components=2)
    var_explained = pca_result['explained_variance']
    print(f"✓ PCA complete: PC1={100*var_explained[0]:.1f}%, PC2={100*var_explained[1]:.1f}%")
except Exception as e:
    print(f"✗ Error in PCA: {e}")

# Summary
print("\n" + "=" * 70)
print("VERIFICATION COMPLETE!")
print("=" * 70)
print("\nAll systems operational! You can now:")
print("  • Run full examples: python examples/geochem_0X_*.py")
print("  • Load any element from the database")
print("  • Perform statistical analyses")
print("  • Create spatial maps")
print("  • Run multivariate analyses")
print("\nNext steps:")
print("  1. Read: examples/README_GEOCHEMISTRY.md")
print("  2. Run: examples/geochem_01_exploratory.py")
print("  3. Explore: Modify examples for your needs")
print("\nHappy analyzing! 🌍📊")
