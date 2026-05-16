"""
Geochemistry Example 4: Multivariate Analysis

This example demonstrates multivariate statistical techniques for
geochemical data including PCA, clustering, and compositional analysis.

Topics covered:
- Principal Component Analysis (PCA)
- K-means clustering
- Hierarchical clustering
- Element associations
- Geochemical signatures
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

from earthsciences.data import load_stream_sediments
from earthsciences.multivariate import (
    hierarchical_clustering,
    kmeans_clustering,
    principal_component_analysis,
)
from earthsciences.utils.logging_config import log_block, log_section, setup_logging

setup_logging()
logger = logging.getLogger(__name__)

np.random.seed(42)

log_section("GEOCHEMISTRY EXAMPLE 4: Multivariate Analysis")

# %% Load Data
logger.info("\n1. Loading multi-element geochemical data...")
# Load a comprehensive suite of elements
elements = ["Cu", "Pb", "Zn", "Ag", "Au", "Mo", "As", "Sb", "Ba", "Sr"]
data = load_stream_sediments(elements)

# Require at least 7 elements for meaningful multivariate analysis
data_mv = data.dropna(subset=elements, thresh=7).copy()

logger.info(f"Loaded {len(data_mv)} samples with >= 7 elements")
logger.info(f"Elements: {elements}")

# Extract element data
X = data_mv[elements].values

# Log-transform (geochemical data is typically lognormal)
X_log = np.log10(X + 1)  # Add 1 to handle zeros

logger.info(f"\nData matrix shape: {X_log.shape}")
logger.info("(samples × elements)")

# %% Principal Component Analysis
logger.info("\n\n2. Principal Component Analysis (PCA)")
logger.info("Performing PCA on log-transformed data...")

# Standardize data (important for PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_log)

# Perform PCA
pca_result = principal_component_analysis(X_scaled, n_components=5)

logger.info("\n✓ PCA complete")
logger.info(f"  Components extracted: {pca_result['n_components']}")

# Variance explained
var_explained = pca_result["explained_variance"]
cum_var = pca_result["cumulative_variance"]

logger.info("\nVariance Explained:")
for i in range(min(5, len(var_explained))):
    logger.info(f"  PC{i+1}: {100*var_explained[i]:.1f}% (cumulative: {100*cum_var[i]:.1f}%)")

logger.info(f"\n{100*cum_var[2]:.1f}% of variance explained by first 3 components")

# Loadings (element contributions to PCs)
loadings = pca_result["loadings"]

logger.info("\nPC1 Loadings (primary geochemical signature):")
pc1_loadings = loadings[:, 0]
sorted_idx = np.argsort(np.abs(pc1_loadings))[::-1]
for i in sorted_idx[:5]:
    if i < len(elements):
        logger.info(f"  {elements[i]:>4}: {pc1_loadings[i]:>7.3f}")

logger.info("\nPC2 Loadings (secondary signature):")
pc2_loadings = loadings[:, 1]
sorted_idx = np.argsort(np.abs(pc2_loadings))[::-1]
for i in sorted_idx[:5]:
    if i < len(elements):
        logger.info(f"  {elements[i]:>4}: {pc2_loadings[i]:>7.3f}")

# %% K-Means Clustering
logger.info("\n\n3. K-Means Clustering")
# Determine optimal number of clusters (try 2-6)
logger.info("Testing different numbers of clusters...")

inertias = []
for k in range(2, 7):
    km_result = kmeans_clustering(X_scaled, n_clusters=k, random_state=42)
    inertias.append(km_result["inertia"])
    logger.info(f"  k={k}: inertia={km_result['inertia']:.1f}")

# Use k=3 or k=4 (elbow method)
optimal_k = 4
logger.info(f"\nUsing k={optimal_k} clusters")

km_result = kmeans_clustering(X_scaled, n_clusters=optimal_k, random_state=42)
clusters = km_result["labels"]

logger.info("\nCluster sizes:")
for i in range(optimal_k):
    n_samples = np.sum(clusters == i)
    pct = 100 * n_samples / len(clusters)
    logger.info(f"  Cluster {i+1}: {n_samples:>5} samples ({pct:>5.1f}%)")

# Cluster centers (back-transform to original scale)
centers = km_result["centers"]
centers_unscaled = scaler.inverse_transform(centers)

logger.info("Cluster Geochemical Signatures (geometric mean ppm):")

cluster_header = f"{'Element':>8}" + "".join(f"  Cluster{i + 1:>1}" for i in range(optimal_k))
cluster_rows = [cluster_header]
for j, elem in enumerate(elements):
    row = f"{elem:>8}"
    for i in range(optimal_k):
        value = 10 ** centers_unscaled[i, j] - 1
        row += f"  {value:>9.1f}"
    cluster_rows.append(row)

log_block("\n".join(cluster_rows))

# %% Hierarchical Clustering
logger.info("\n\n4. Hierarchical Clustering")
# Use subset for visualization (hierarchical is O(n²))
if len(X_scaled) > 200:
    sample_idx = np.random.choice(len(X_scaled), 200, replace=False)
    X_hier = X_scaled[sample_idx]
    logger.info(f"Using {len(X_hier)} samples (subsampled for visualization)")
else:
    X_hier = X_scaled

hc_result = hierarchical_clustering(X_hier, n_clusters=optimal_k, linkage="ward")
logger.info("\n✓ Hierarchical clustering complete")
logger.info("  Linkage method: ward")
logger.info(f"  Clusters: {optimal_k}")

# %% Visualization
logger.info("\n\n5. Creating Visualizations...")
fig = plt.figure(figsize=(18, 12))
fig.suptitle("Multivariate Geochemical Analysis", fontsize=16, fontweight="bold")

# Scree plot (variance explained)
ax1 = plt.subplot(2, 3, 1)
n_components = min(len(var_explained), 10)
ax1.bar(
    range(1, n_components + 1),
    var_explained[:n_components] * 100,
    alpha=0.7,
    color="steelblue",
    edgecolor="black",
)
ax1.plot(
    range(1, n_components + 1),
    cum_var[:n_components] * 100,
    "ro-",
    linewidth=2,
    markersize=8,
    label="Cumulative",
)
ax1.set_xlabel("Principal Component", fontsize=10)
ax1.set_ylabel("Variance Explained (%)", fontsize=10)
ax1.set_title("Scree Plot", fontsize=12, fontweight="bold")
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3, axis="y")

# PCA Biplot
ax2 = plt.subplot(2, 3, 2)
scores = pca_result["scores"]
ax2.scatter(
    scores[:, 0],
    scores[:, 1],
    c=clusters,
    cmap="viridis",
    s=20,
    alpha=0.5,
    edgecolors="black",
    linewidths=0.3,
)

# Add loading vectors
scale = 3
for i, elem in enumerate(elements):
    if i < loadings.shape[0]:
        ax2.arrow(
            0,
            0,
            loadings[i, 0] * scale,
            loadings[i, 1] * scale,
            head_width=0.1,
            head_length=0.1,
            fc="red",
            ec="red",
            alpha=0.7,
            linewidth=1.5,
        )
        ax2.text(
            loadings[i, 0] * scale * 1.1,
            loadings[i, 1] * scale * 1.1,
            elem,
            fontsize=9,
            fontweight="bold",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="red", alpha=0.8),
        )

ax2.set_xlabel(f"PC1 ({100*var_explained[0]:.1f}% var)", fontsize=10)
ax2.set_ylabel(f"PC2 ({100*var_explained[1]:.1f}% var)", fontsize=10)
ax2.set_title("PCA Biplot (colored by cluster)", fontsize=12, fontweight="bold")
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0, color="k", linestyle="--", alpha=0.3)
ax2.axvline(x=0, color="k", linestyle="--", alpha=0.3)

# PC1 vs PC3
ax3 = plt.subplot(2, 3, 3)
ax3.scatter(
    scores[:, 0],
    scores[:, 2],
    c=clusters,
    cmap="viridis",
    s=20,
    alpha=0.5,
    edgecolors="black",
    linewidths=0.3,
)
ax3.set_xlabel(f"PC1 ({100*var_explained[0]:.1f}% var)", fontsize=10)
ax3.set_ylabel(f"PC3 ({100*var_explained[2]:.1f}% var)", fontsize=10)
ax3.set_title("PC1 vs PC3", fontsize=12, fontweight="bold")


# Cluster geographic distribution
ax4 = plt.subplot(2, 3, 4)
lons = data_mv["LONGITUDE"].values
lats = data_mv["LATITUDE"].values
scatter = ax4.scatter(
    lons, lats, c=clusters, cmap="viridis", s=30, edgecolors="black", linewidths=0.5, alpha=0.7
)
ax4.set_xlabel("Longitude (°)", fontsize=10)
ax4.set_ylabel("Latitude (°)", fontsize=10)
ax4.set_title("Cluster Geographic Distribution", fontsize=12, fontweight="bold")

cbar = plt.colorbar(scatter, ax=ax4)
cbar.set_label("Cluster", rotation=270, labelpad=15)

# Elbow plot for k-means
ax5 = plt.subplot(2, 3, 5)
k_values = range(2, 7)
ax5.plot(k_values, inertias, "bo-", linewidth=2, markersize=10)
ax5.axvline(x=optimal_k, color="red", linestyle="--", alpha=0.7, label=f"Selected k={optimal_k}")
ax5.set_xlabel("Number of Clusters (k)", fontsize=10)
ax5.set_ylabel("Within-Cluster Sum of Squares", fontsize=10)
ax5.set_title("Elbow Method for Optimal k", fontsize=12, fontweight="bold")
ax5.legend(fontsize=9)


# Dendrogram (simplified)
ax6 = plt.subplot(2, 3, 6)
if "dendrogram" in hc_result:
    # Plot simplified dendrogram
    from scipy.cluster.hierarchy import dendrogram

    dendrogram(
        hc_result["linkage_matrix"],
        ax=ax6,
        no_labels=True,
        color_threshold=0,
        above_threshold_color="steelblue",
    )
    ax6.set_xlabel("Sample Index", fontsize=10)
    ax6.set_ylabel("Distance", fontsize=10)
    ax6.set_title("Hierarchical Clustering Dendrogram", fontsize=12, fontweight="bold")

else:
    # Alternative: show cluster comparison
    for i in range(optimal_k):
        mask = clusters == i
        ax6.scatter(
            scores[mask, 0],
            scores[mask, 1],
            label=f"Cluster {i+1}",
            s=30,
            alpha=0.6,
            edgecolors="black",
            linewidths=0.5,
        )
    ax6.set_xlabel("PC1", fontsize=10)
    ax6.set_ylabel("PC2", fontsize=10)
    ax6.set_title("Clusters in PC Space", fontsize=12, fontweight="bold")
    ax6.legend(fontsize=9)


plt.tight_layout()
plt.savefig("examples/geochem_04_multivariate.png", dpi=150, bbox_inches="tight")
logger.info("✓ Saved: examples/geochem_04_multivariate.png")

# %% Summary
log_section("SUMMARY")
logger.info(f"\nMultivariate analysis of {len(data_mv)} samples:")
logger.info(f"• PCA: First 3 PCs explain {100*cum_var[2]:.1f}% of variance")
logger.info(f"• Identified {optimal_k} distinct geochemical populations")
logger.info("• Element associations revealed by loadings")
logger.info("• Clusters show spatial patterns")
logger.info("\nKey insights:")
logger.info("• PC1 often represents overall mineralization level")
logger.info("• PC2-3 distinguish different mineral deposit types")
logger.info("• Clusters may represent different geological units or processes")
logger.info("• PCA reduces dimensionality while preserving information")
logger.info("\nNext steps:")
logger.info("• Example 5: Advanced techniques (compositional data, ternary plots)")
logger.info("• Validate clusters with geological mapping")
logger.info("• Compare with known mineral deposits")

plt.show()
