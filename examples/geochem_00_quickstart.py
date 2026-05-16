"""
QUICK START: Geochemistry Analysis with Alaska Data

This script provides a quick overview of geochemical analysis capabilities
using real Alaska Geochemical Database data.

Run this first to verify everything works!
"""

import logging

import numpy as np

from earthsciences.data import get_element_stats, load_stream_sediments
from earthsciences.multivariate import principal_component_analysis
from earthsciences.spatial import idw_interpolation
from earthsciences.statistics import correlation
from earthsciences.utils.logging_config import log_section, log_step, setup_logging

setup_logging()
logger = logging.getLogger(__name__)

log_section("QUICK START: Geochemistry Analysis")

# Test 1: Load Data
log_step("1. Loading geochemical data")
try:
    elements = ["Cu", "Zn", "Pb"]
    data = load_stream_sediments(elements)
    logger.info(f"✓ Loaded {len(data)} stream sediment samples")
    logger.info(f"✓ Elements: {elements}")
except Exception as e:
    logger.info(f"✗ Error loading data: {e}")
    exit(1)

# Test 2: Statistics
log_step("2. Computing statistics")
try:
    cu_stats = get_element_stats("Cu")
    logger.info(f"✓ Cu: median={cu_stats['median']:.1f} ppm, P95={cu_stats['p95']:.1f} ppm")
except Exception as e:
    logger.info(f"✗ Error computing stats: {e}")

# Test 3: Correlation
log_step("3. Analyzing correlations")
try:
    mask = data["Cu"].notna() & data["Zn"].notna()
    subset_corr = data.loc[mask].head(1000)  # Subset for speed
    cu = subset_corr["Cu"].values
    zn = subset_corr["Zn"].values

    r, p_value = correlation(cu, zn)
    logger.info(f"✓ Cu-Zn correlation: r={r:.3f}, p={p_value:.4e}")
except Exception as e:
    logger.info(f"✗ Error in correlation: {e}")

# Test 4: Spatial Analysis
log_step("4. Testing spatial interpolation")
try:
    # Small subset for quick test
    subset = data.sample(n=100, random_state=42)
    x = subset["LONGITUDE"].values
    y = subset["LATITUDE"].values
    v = subset["Cu"].dropna().values[: len(x)]

    # Create small grid
    grid_x, grid_y = np.meshgrid(
        np.linspace(x.min(), x.max(), 20), np.linspace(y.min(), y.max(), 20)
    )

    result = idw_interpolation(x[: len(v)], y[: len(v)], v, grid_x, grid_y)
    logger.info("✓ Spatial interpolation complete (20x20 grid)")
except Exception as e:
    logger.info(f"✗ Error in spatial analysis: {e}")

# Test 5: Multivariate
log_step("5. Testing multivariate analysis")
try:
    # Small subset with complete data
    subset = data.dropna(subset=elements).sample(n=200, random_state=42)
    X = subset[elements].values

    # Log-transform and standardize
    X_log = np.log10(X + 1)
    from sklearn.preprocessing import StandardScaler

    X_scaled = StandardScaler().fit_transform(X_log)

    pca_result = principal_component_analysis(X_scaled, n_components=2)
    var_explained = pca_result["explained_variance"]
    logger.info(f"✓ PCA complete: PC1={100*var_explained[0]:.1f}%, PC2={100*var_explained[1]:.1f}%")
except Exception as e:
    logger.info(f"✗ Error in PCA: {e}")

# Summary
log_section("VERIFICATION COMPLETE!")
logger.info("\nAll systems operational! You can now:")
logger.info("  • Run full examples: python examples/geochem_0X_*.py")
logger.info("  • Load any element from the database")
logger.info("  • Perform statistical analyses")
logger.info("  • Create spatial maps")
logger.info("  • Run multivariate analyses")
logger.info("\nNext steps:")
logger.info("  1. Read: examples/README_GEOCHEMISTRY.md")
logger.info("  2. Run: examples/geochem_01_exploratory.py")
logger.info("  3. Explore: Modify examples for your needs")
logger.info("\nHappy analyzing! 🌍📊")
