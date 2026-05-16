"""
Multivariate analysis

PCA, clustering, classification, compositional data, and dimensionality reduction.
"""

from .classification import *
from .clustering import *
from .compositional import *
from .decomposition import *
from .dimensionality import *
from .pca import *

__all__ = [
    # PCA
    "principal_component_analysis",
    "pca_biplot",
    "scree_plot",
    # Clustering
    "kmeans_clustering",
    "hierarchical_clustering",
    "dbscan_clustering",
    # Classification
    "discriminant_analysis",
    "confusion_matrix_metrics",
    # Decomposition
    "independent_component_analysis",
    "non_negative_matrix_factorization",
    "sparse_pca",
    # Dimensionality reduction
    "tsne",
    "umap_embedding",
    "isomap",
    # Compositional data (Aitchison)
    "alr",
    "alr_inv",
    "clr",
    "clr_inv",
    "ilr",
    "ilr_inv",
    "aitchison_distance",
    "closure",
    "perturbation",
    "powering",
    "compositional_mean",
    "variation_matrix",
]
