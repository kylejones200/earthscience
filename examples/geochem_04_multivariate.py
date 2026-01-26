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

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from earthsciences.data import load_stream_sediments
from earthsciences.multivariate import (
    principal_component_analysis,
    kmeans_clustering,
    hierarchical_clustering,
    pca_biplot
)

np.random.seed(42)

print("=" * 70)
print("GEOCHEMISTRY EXAMPLE 4: Multivariate Analysis")
print("=" * 70)

# %% Load Data
print("\n1. Loading multi-element geochemical data...")
print("-" * 70)

# Load a comprehensive suite of elements
elements = ['Cu', 'Pb', 'Zn', 'Ag', 'Au', 'Mo', 'As', 'Sb', 'Ba', 'Sr']
data = load_stream_sediments(elements)

# Require at least 7 elements for meaningful multivariate analysis
data_mv = data.dropna(subset=elements, thresh=7).copy()

print(f"Loaded {len(data_mv)} samples with >= 7 elements")
print(f"Elements: {elements}")

# Extract element data
X = data_mv[elements].values

# Log-transform (geochemical data is typically lognormal)
X_log = np.log10(X + 1)  # Add 1 to handle zeros

print(f"\nData matrix shape: {X_log.shape}")
print(f"(samples × elements)")

# %% Principal Component Analysis
print("\n\n2. Principal Component Analysis (PCA)")
print("-" * 70)

print("Performing PCA on log-transformed data...")

# Standardize data (important for PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_log)

# Perform PCA
pca_result = principal_component_analysis(X_scaled, n_components=5)

print(f"\n✓ PCA complete")
print(f"  Components extracted: {pca_result['n_components']}")

# Variance explained
var_explained = pca_result['explained_variance']
cum_var = pca_result['cumulative_variance']

print(f"\nVariance Explained:")
for i in range(min(5, len(var_explained))):
    print(f"  PC{i+1}: {100*var_explained[i]:.1f}% (cumulative: {100*cum_var[i]:.1f}%)")

print(f"\n{100*cum_var[2]:.1f}% of variance explained by first 3 components")

# Loadings (element contributions to PCs)
loadings = pca_result['loadings']

print(f"\nPC1 Loadings (primary geochemical signature):")
pc1_loadings = loadings[:, 0]
sorted_idx = np.argsort(np.abs(pc1_loadings))[::-1]
for i in sorted_idx[:5]:
    if i < len(elements):
        print(f"  {elements[i]:>4}: {pc1_loadings[i]:>7.3f}")

print(f"\nPC2 Loadings (secondary signature):")
pc2_loadings = loadings[:, 1]
sorted_idx = np.argsort(np.abs(pc2_loadings))[::-1]
for i in sorted_idx[:5]:
    if i < len(elements):
        print(f"  {elements[i]:>4}: {pc2_loadings[i]:>7.3f}")

# %% K-Means Clustering
print("\n\n3. K-Means Clustering")
print("-" * 70)

# Determine optimal number of clusters (try 2-6)
print("Testing different numbers of clusters...")

inertias = []
for k in range(2, 7):
    km_result = kmeans_clustering(X_scaled, n_clusters=k, random_state=42)
    inertias.append(km_result['inertia'])
    print(f"  k={k}: inertia={km_result['inertia']:.1f}")

# Use k=3 or k=4 (elbow method)
optimal_k = 4
print(f"\nUsing k={optimal_k} clusters")

km_result = kmeans_clustering(X_scaled, n_clusters=optimal_k, random_state=42)
clusters = km_result['labels']

print(f"\nCluster sizes:")
for i in range(optimal_k):
    n_samples = np.sum(clusters == i)
    pct = 100 * n_samples / len(clusters)
    print(f"  Cluster {i+1}: {n_samples:>5} samples ({pct:>5.1f}%)")

# Cluster centers (back-transform to original scale)
centers = km_result['centers']
centers_unscaled = scaler.inverse_transform(centers)

print(f"\nCluster Geochemical Signatures (geometric mean ppm):")
print(f"{'Element':>8}", end="")
for i in range(optimal_k):
    print(f"  Cluster{i+1:>1}", end="")
print()

for j, elem in enumerate(elements):
    print(f"{elem:>8}", end="")
    for i in range(optimal_k):
        # Convert back from log10
        value = 10 ** centers_unscaled[i, j] - 1
        print(f"  {value:>9.1f}", end="")
    print()

# %% Hierarchical Clustering
print("\n\n4. Hierarchical Clustering")
print("-" * 70)

# Use subset for visualization (hierarchical is O(n²))
if len(X_scaled) > 200:
    sample_idx = np.random.choice(len(X_scaled), 200, replace=False)
    X_hier = X_scaled[sample_idx]
    print(f"Using {len(X_hier)} samples (subsampled for visualization)")
else:
    X_hier = X_scaled

hc_result = hierarchical_clustering(X_hier, n_clusters=optimal_k,
                                    linkage='ward')
print(f"\n✓ Hierarchical clustering complete")
print(f"  Linkage method: ward")
print(f"  Clusters: {optimal_k}")

# %% Visualization
print("\n\n5. Creating Visualizations...")
print("-" * 70)

fig = plt.figure(figsize=(18, 12))
fig.suptitle('Multivariate Geochemical Analysis',
             fontsize=16, fontweight='bold')

# Scree plot (variance explained)
ax1 = plt.subplot(2, 3, 1)
n_components = min(len(var_explained), 10)
ax1.bar(range(1, n_components+1), var_explained[:n_components]*100,
       alpha=0.7, color='steelblue', edgecolor='black')
ax1.plot(range(1, n_components+1), cum_var[:n_components]*100,
        'ro-', linewidth=2, markersize=8, label='Cumulative')
ax1.set_xlabel('Principal Component', fontsize=10)
ax1.set_ylabel('Variance Explained (%)', fontsize=10)
ax1.set_title('Scree Plot', fontsize=12, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3, axis='y')

# PCA Biplot
ax2 = plt.subplot(2, 3, 2)
scores = pca_result['scores']
ax2.scatter(scores[:, 0], scores[:, 1], c=clusters, cmap='viridis',
           s=20, alpha=0.5, edgecolors='black', linewidths=0.3)

# Add loading vectors
scale = 3
for i, elem in enumerate(elements):
    if i < loadings.shape[0]:
        ax2.arrow(0, 0, loadings[i, 0]*scale, loadings[i, 1]*scale,
                 head_width=0.1, head_length=0.1, fc='red', ec='red',
                 alpha=0.7, linewidth=1.5)
        ax2.text(loadings[i, 0]*scale*1.1, loadings[i, 1]*scale*1.1, elem,
                fontsize=9, fontweight='bold', ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                         edgecolor='red', alpha=0.8))

ax2.set_xlabel(f'PC1 ({100*var_explained[0]:.1f}% var)', fontsize=10)
ax2.set_ylabel(f'PC2 ({100*var_explained[1]:.1f}% var)', fontsize=10)
ax2.set_title('PCA Biplot (colored by cluster)', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax2.axvline(x=0, color='k', linestyle='--', alpha=0.3)

# PC1 vs PC3
ax3 = plt.subplot(2, 3, 3)
ax3.scatter(scores[:, 0], scores[:, 2], c=clusters, cmap='viridis',
           s=20, alpha=0.5, edgecolors='black', linewidths=0.3)
ax3.set_xlabel(f'PC1 ({100*var_explained[0]:.1f}% var)', fontsize=10)
ax3.set_ylabel(f'PC3 ({100*var_explained[2]:.1f}% var)', fontsize=10)
ax3.set_title('PC1 vs PC3', fontsize=12, fontweight='bold')


# Cluster geographic distribution
ax4 = plt.subplot(2, 3, 4)
lons = data_mv['LONGITUDE'].values
lats = data_mv['LATITUDE'].values
scatter = ax4.scatter(lons, lats, c=clusters, cmap='viridis', s=30,
                     edgecolors='black', linewidths=0.5, alpha=0.7)
ax4.set_xlabel('Longitude (°)', fontsize=10)
ax4.set_ylabel('Latitude (°)', fontsize=10)
ax4.set_title('Cluster Geographic Distribution', fontsize=12, fontweight='bold')

cbar = plt.colorbar(scatter, ax=ax4)
cbar.set_label('Cluster', rotation=270, labelpad=15)

# Elbow plot for k-means
ax5 = plt.subplot(2, 3, 5)
k_values = range(2, 7)
ax5.plot(k_values, inertias, 'bo-', linewidth=2, markersize=10)
ax5.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7,
           label=f'Selected k={optimal_k}')
ax5.set_xlabel('Number of Clusters (k)', fontsize=10)
ax5.set_ylabel('Within-Cluster Sum of Squares', fontsize=10)
ax5.set_title('Elbow Method for Optimal k', fontsize=12, fontweight='bold')
ax5.legend(fontsize=9)


# Dendrogram (simplified)
ax6 = plt.subplot(2, 3, 6)
if 'dendrogram' in hc_result:
    # Plot simplified dendrogram
    from scipy.cluster.hierarchy import dendrogram
    dendrogram(hc_result['linkage_matrix'], ax=ax6, no_labels=True,
              color_threshold=0, above_threshold_color='steelblue')
    ax6.set_xlabel('Sample Index', fontsize=10)
    ax6.set_ylabel('Distance', fontsize=10)
    ax6.set_title('Hierarchical Clustering Dendrogram', fontsize=12, fontweight='bold')
    
else:
    # Alternative: show cluster comparison
    for i in range(optimal_k):
        mask = clusters == i
        ax6.scatter(scores[mask, 0], scores[mask, 1],
                   label=f'Cluster {i+1}', s=30, alpha=0.6,
                   edgecolors='black', linewidths=0.5)
    ax6.set_xlabel(f'PC1', fontsize=10)
    ax6.set_ylabel(f'PC2', fontsize=10)
    ax6.set_title('Clusters in PC Space', fontsize=12, fontweight='bold')
    ax6.legend(fontsize=9)
    

plt.tight_layout()
plt.savefig('examples/geochem_04_multivariate.png', dpi=150, bbox_inches='tight')
print("✓ Saved: examples/geochem_04_multivariate.png")

# %% Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"\nMultivariate analysis of {len(data_mv)} samples:")
print(f"• PCA: First 3 PCs explain {100*cum_var[2]:.1f}% of variance")
print(f"• Identified {optimal_k} distinct geochemical populations")
print(f"• Element associations revealed by loadings")
print(f"• Clusters show spatial patterns")
print("\nKey insights:")
print("• PC1 often represents overall mineralization level")
print("• PC2-3 distinguish different mineral deposit types")
print("• Clusters may represent different geological units or processes")
print("• PCA reduces dimensionality while preserving information")
print("\nNext steps:")
print("• Example 5: Advanced techniques (compositional data, ternary plots)")
print("• Validate clusters with geological mapping")
print("• Compare with known mineral deposits")

plt.show()
