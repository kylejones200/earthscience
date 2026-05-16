"""
Geochemistry Example 3: Spatial Analysis and Mapping

This example demonstrates spatial analysis of geochemical data including
spatial interpolation, variogram modeling, and geochemical mapping.

Topics covered:
- Spatial distribution of elements
- Variogram analysis
- Kriging interpolation
- Geochemical anomaly mapping
- Hot spot analysis
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from earthsciences.data import load_stream_sediments
from earthsciences.spatial import (
    compute_variogram,
    fit_variogram_model,
    idw_interpolation,
    ordinary_kriging,
)
from earthsciences.utils.logging_config import log_section, setup_logging

setup_logging()
logger = logging.getLogger(__name__)

np.random.seed(42)

log_section("GEOCHEMISTRY EXAMPLE 3: Spatial Analysis")

# %% Load Data
logger.info("\n1. Loading spatial geochemical data...")
# Load copper data (good indicator element)
elements = ["Cu"]
data = load_stream_sediments(elements)

# Filter for valid coordinates and values
data = data[(data["Cu"].notna()) & (data["Cu"] > 0)].copy()

logger.info(f"Loaded {len(data)} stream sediment samples with Cu data")
logger.info("Spatial extent:")
logger.info(f"  Latitude:  {data['LATITUDE'].min():.2f}° to {data['LATITUDE'].max():.2f}°")
logger.info(f"  Longitude: {data['LONGITUDE'].min():.2f}° to {data['LONGITUDE'].max():.2f}°")

# Extract coordinates and values
x = data["LONGITUDE"].values
y = data["LATITUDE"].values
cu_values = data["Cu"].values

logger.info(f"\nCu concentration range: {cu_values.min():.1f} - {cu_values.max():.1f} ppm")
logger.info(f"Median Cu: {np.median(cu_values):.1f} ppm")
logger.info(f"95th percentile: {np.percentile(cu_values, 95):.1f} ppm")

# %% Variogram Analysis
logger.info("\n\n2. Variogram Analysis")
# Subsample for variogram (computationally expensive for large datasets)
if len(data) > 500:
    sample_idx = np.random.choice(len(data), 500, replace=False)
    x_var = x[sample_idx]
    y_var = y[sample_idx]
    v_var = cu_values[sample_idx]
    logger.info(f"Using {len(x_var)} samples for variogram (subsampled for speed)")
else:
    x_var = x
    y_var = y
    v_var = cu_values

# Compute experimental variogram
max_dist = 2.0  # degrees (roughly 200 km in Alaska)
n_lags = 15

logger.info("\nComputing experimental variogram...")
logger.info(f"  Max distance: {max_dist:.1f} degrees")
logger.info(f"  Number of lags: {n_lags}")

variogram_result = compute_variogram(x_var, y_var, v_var, max_dist=max_dist, n_lags=n_lags)

lags = variogram_result["lags"]
semivariance = variogram_result["semivariance"]
n_pairs = variogram_result["n_pairs"]

logger.info("\nVariogram computed successfully")
logger.info(f"  Distance range: 0 to {lags[-1]:.2f} degrees")
logger.info(f"  Total pairs analyzed: {sum(n_pairs):,}")

# Fit variogram model
logger.info("\nFitting variogram models...")

models = ["spherical", "exponential", "gaussian"]
best_model = None
best_params = None
best_residual = np.inf

for model in models:
    params = fit_variogram_model(lags, semivariance, model=model)

    # Calculate model fit
    if model == "spherical":
        model_values = params["nugget"] + (params["sill"] - params["nugget"]) * (
            (3 * lags) / (2 * params["range"]) - (lags**3) / (2 * params["range"] ** 3)
        )
        model_values[lags > params["range"]] = params["sill"]
    elif model == "exponential":
        model_values = params["nugget"] + (params["sill"] - params["nugget"]) * (
            1 - np.exp(-3 * lags / params["range"])
        )
    else:  # gaussian
        model_values = params["nugget"] + (params["sill"] - params["nugget"]) * (
            1 - np.exp(-(lags**2) / params["range"] ** 2)
        )

    residual = np.sum((semivariance - model_values) ** 2)

    logger.info(f"\n  {model.capitalize()} Model:")
    logger.info(f"    Nugget: {params['nugget']:.2f}")
    logger.info(f"    Sill:   {params['sill']:.2f}")
    logger.info(f"    Range:  {params['range']:.2f} degrees")
    logger.info(f"    Residual: {residual:.2f}")

    if residual < best_residual:
        best_residual = residual
        best_model = model
        best_params = params

logger.info(f"\n✓ Best model: {best_model.capitalize()}")

# %% Spatial Interpolation
logger.info("\n\n3. Spatial Interpolation")
# Create interpolation grid
x_min, x_max = x.min(), x.max()
y_min, y_max = y.min(), y.max()

# Add buffer
buffer = 0.5
x_min -= buffer
x_max += buffer
y_min -= buffer
y_max += buffer

# Create grid (coarser for speed)
grid_size = 50
grid_x, grid_y = np.meshgrid(
    np.linspace(x_min, x_max, grid_size), np.linspace(y_min, y_max, grid_size)
)

logger.info(f"Interpolation grid: {grid_size}x{grid_size} = {grid_size**2} points")

# IDW interpolation (fast)
logger.info("\nPerforming IDW interpolation...")
cu_idw = idw_interpolation(x, y, cu_values, grid_x, grid_y, power=2.0)
logger.info("✓ IDW complete")

# Kriging interpolation (slower but better)
logger.info("\nPerforming ordinary kriging...")
logger.info("  (This may take a few seconds...)")


# Create variogram function from best model
def variogram_func(h):
    if best_model == "spherical":
        gamma = best_params["nugget"] + (best_params["sill"] - best_params["nugget"]) * (
            (3 * h) / (2 * best_params["range"]) - (h**3) / (2 * best_params["range"] ** 3)
        )
        gamma[h > best_params["range"]] = best_params["sill"]
    elif best_model == "exponential":
        gamma = best_params["nugget"] + (best_params["sill"] - best_params["nugget"]) * (
            1 - np.exp(-3 * h / best_params["range"])
        )
    else:  # gaussian
        gamma = best_params["nugget"] + (best_params["sill"] - best_params["nugget"]) * (
            1 - np.exp(-(h**2) / best_params["range"] ** 2)
        )
    return gamma


cu_kriged, cu_variance = ordinary_kriging(
    x, y, cu_values, grid_x, grid_y, variogram_func, return_variance=True
)
logger.info("✓ Kriging complete")

# %% Anomaly Detection
logger.info("\n\n4. Geochemical Anomaly Detection")
# Calculate anomaly threshold (e.g., P95)
threshold = np.percentile(cu_values, 95)
logger.info(f"Anomaly threshold (P95): {threshold:.1f} ppm")

# Count anomalies
n_anomalies = np.sum(cu_values > threshold)
pct_anomalies = 100 * n_anomalies / len(cu_values)
logger.info(f"Anomalous samples: {n_anomalies} ({pct_anomalies:.1f}%)")

# Identify high anomaly areas in kriged map
anomaly_mask = cu_kriged > threshold
n_grid_anomalies = np.sum(anomaly_mask)
pct_grid_anomalies = 100 * n_grid_anomalies / cu_kriged.size
logger.info(f"Anomalous grid cells: {n_grid_anomalies} ({pct_grid_anomalies:.1f}%)")

# %% Visualization
logger.info("\n\n5. Creating Maps...")
fig = plt.figure(figsize=(18, 12))
fig.suptitle("Copper Geochemical Maps - Alaska Stream Sediments", fontsize=16, fontweight="bold")

# Sample locations
ax1 = plt.subplot(2, 3, 1)
scatter = ax1.scatter(
    x,
    y,
    c=cu_values,
    s=20,
    cmap="YlOrRd",
    norm=LogNorm(vmin=max(1, cu_values.min()), vmax=cu_values.max()),
    edgecolors="black",
    linewidths=0.5,
    alpha=0.7,
)
ax1.set_xlabel("Longitude (°)", fontsize=10)
ax1.set_ylabel("Latitude (°)", fontsize=10)
ax1.set_title("Sample Locations (log scale)", fontsize=12, fontweight="bold")
ax1.grid(True, alpha=0.3)
cbar1 = plt.colorbar(scatter, ax=ax1)
cbar1.set_label("Cu (ppm)", rotation=270, labelpad=20)

# IDW interpolation
ax2 = plt.subplot(2, 3, 2)
im2 = ax2.contourf(grid_x, grid_y, cu_idw, levels=20, cmap="YlOrRd")
ax2.scatter(x, y, c="black", s=5, alpha=0.3, marker=".")
ax2.set_xlabel("Longitude (°)", fontsize=10)
ax2.set_ylabel("Latitude (°)", fontsize=10)
ax1.set_title("IDW Interpolation", fontsize=12, fontweight="bold")
ax2.grid(True, alpha=0.3)
cbar2 = plt.colorbar(im2, ax=ax2)
cbar2.set_label("Cu (ppm)", rotation=270, labelpad=20)

# Kriging interpolation
ax3 = plt.subplot(2, 3, 3)
im3 = ax3.contourf(grid_x, grid_y, cu_kriged, levels=20, cmap="YlOrRd")
ax3.scatter(x, y, c="black", s=5, alpha=0.3, marker=".")
ax3.set_xlabel("Longitude (°)", fontsize=10)
ax3.set_ylabel("Latitude (°)", fontsize=10)
ax3.set_title("Ordinary Kriging", fontsize=12, fontweight="bold")
ax3.grid(True, alpha=0.3)
cbar3 = plt.colorbar(im3, ax=ax3)
cbar3.set_label("Cu (ppm)", rotation=270, labelpad=20)

# Kriging variance (uncertainty)
ax4 = plt.subplot(2, 3, 4)
im4 = ax4.contourf(grid_x, grid_y, np.sqrt(cu_variance), levels=20, cmap="viridis")
ax4.scatter(x, y, c="white", s=5, alpha=0.5, marker=".")
ax4.set_xlabel("Longitude (°)", fontsize=10)
ax4.set_ylabel("Latitude (°)", fontsize=10)
ax4.set_title("Kriging Standard Error", fontsize=12, fontweight="bold")
ax4.grid(True, alpha=0.3)
cbar4 = plt.colorbar(im4, ax=ax4)
cbar4.set_label("Std Error (ppm)", rotation=270, labelpad=20)

# Variogram
ax5 = plt.subplot(2, 3, 5)
ax5.scatter(
    lags,
    semivariance,
    c="blue",
    s=100,
    alpha=0.6,
    edgecolors="black",
    linewidths=1,
    label="Experimental",
)

# Plot model
h_model = np.linspace(0, lags[-1], 100)
gamma_model = variogram_func(h_model)
ax5.plot(h_model, gamma_model, "r-", linewidth=2, label=f"{best_model.capitalize()} model")

ax5.axhline(
    y=best_params["sill"],
    color="green",
    linestyle="--",
    alpha=0.7,
    label=f"Sill = {best_params['sill']:.1f}",
)
ax5.axhline(
    y=best_params["nugget"],
    color="orange",
    linestyle="--",
    alpha=0.7,
    label=f"Nugget = {best_params['nugget']:.1f}",
)
ax5.axvline(
    x=best_params["range"],
    color="purple",
    linestyle="--",
    alpha=0.7,
    label=f"Range = {best_params['range']:.2f}°",
)

ax5.set_xlabel("Distance (degrees)", fontsize=10)
ax5.set_ylabel("Semivariance", fontsize=10)
ax5.set_title("Experimental Variogram & Model", fontsize=12, fontweight="bold")
ax5.legend(fontsize=8, loc="lower right")
ax5.grid(True, alpha=0.3)

# Anomaly map
ax6 = plt.subplot(2, 3, 6)
anomaly_binary = (cu_kriged > threshold).astype(int)
im6 = ax6.contourf(
    grid_x,
    grid_y,
    cu_kriged,
    levels=[0, threshold, cu_kriged.max()],
    colors=["lightblue", "red"],
    alpha=0.6,
)
ax6.scatter(
    x[cu_values > threshold],
    y[cu_values > threshold],
    c="darkred",
    s=50,
    marker="*",
    edgecolors="black",
    linewidths=0.5,
    label=f"Anomalies (>{threshold:.0f} ppm)",
    zorder=5,
)
ax6.scatter(
    x[cu_values <= threshold],
    y[cu_values <= threshold],
    c="gray",
    s=10,
    alpha=0.3,
    marker=".",
    label="Background",
)
ax6.set_xlabel("Longitude (°)", fontsize=10)
ax6.set_ylabel("Latitude (°)", fontsize=10)
ax6.set_title("Geochemical Anomalies (P95 threshold)", fontsize=12, fontweight="bold")
ax6.legend(fontsize=8, loc="upper right")
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("examples/geochem_03_spatial.png", dpi=150, bbox_inches="tight")
logger.info("✓ Saved: examples/geochem_03_spatial.png")

# %% Summary
log_section("SUMMARY")
logger.info(f"\nSpatial analysis of {len(data)} Cu measurements:")
logger.info(f"• Variogram model: {best_model.capitalize()}")
logger.info(f"• Spatial range: ~{best_params['range']*111:.0f} km")
logger.info(f"• Identified {n_anomalies} anomalous samples")
logger.info("\nKey insights:")
logger.info("• Kriging provides smooth interpolation with uncertainty estimates")
logger.info("• Variogram reveals spatial correlation structure")
logger.info("• Anomaly maps highlight prospective areas")
logger.info("• Low kriging variance near sample locations")
logger.info("\nNext steps:")
logger.info("• Example 4: Multivariate analysis (PCA, clustering)")
logger.info("• Example 5: Advanced geochemistry (ratios, ternary plots)")

plt.show()
