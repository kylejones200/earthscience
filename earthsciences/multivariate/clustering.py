"""
Clustering methods

K-means, hierarchical, DBSCAN, and cluster validation.
"""

import numpy as np
from scipy.cluster.hierarchy import linkage as scipy_linkage


def kmeans(
    X: np.ndarray,
    n_clusters: int = 3,
    max_iter: int = 300,
    n_init: int = 10,
    random_state: int | None = None,
) -> dict:
    """
    K-means clustering.

    Parameters
    ----------
    X : array_like
        Data matrix (n_samples, n_features)
    n_clusters : int
        Number of clusters
    max_iter : int
        Maximum iterations
    n_init : int
        Number of initializations
    random_state : int, optional
        Random seed

    Returns
    -------
    dict
        Dictionary with labels, centroids, inertia
    """
    from sklearn.cluster import KMeans

    X = np.asarray(X)

    model = KMeans(
        n_clusters=n_clusters, max_iter=max_iter, n_init=n_init, random_state=random_state
    )

    labels = model.fit_predict(X)

    return {
        "labels": labels,
        "centroids": model.cluster_centers_,
        "centers": model.cluster_centers_,
        "inertia": model.inertia_,
    }


def hierarchical(
    X: np.ndarray, n_clusters: int = 3, linkage: str = "ward", metric: str = "euclidean"
) -> dict:
    """
    Hierarchical clustering.

    Parameters
    ----------
    X : array_like
        Data matrix
    n_clusters : int
        Number of clusters
    linkage : str
        Linkage method: 'single', 'complete', 'average', 'ward'
    metric : str
        Distance metric

    Returns
    -------
    dict
        Dictionary with labels and linkage matrix
    """
    from sklearn.cluster import AgglomerativeClustering

    X = np.asarray(X)

    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)

    labels = model.fit_predict(X)

    linkage_matrix = scipy_linkage(X, method=linkage, metric=metric)

    return {"labels": labels, "linkage": linkage_matrix, "dendrogram": linkage_matrix}


def dbscan(
    X: np.ndarray, eps: float = 0.5, min_samples: int = 5, metric: str = "euclidean"
) -> dict:
    """
    DBSCAN clustering.

    Parameters
    ----------
    X : array_like
        Data matrix
    eps : float
        Maximum distance between neighbors
    min_samples : int
        Minimum samples in neighborhood
    metric : str
        Distance metric

    Returns
    -------
    dict
        Dictionary with labels (noise points = -1)
    """
    from sklearn.cluster import DBSCAN

    X = np.asarray(X)

    model = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
    labels = model.fit_predict(X)

    return {
        "labels": labels,
        "core_samples": model.core_sample_indices_,
        "n_clusters": len(set(labels)) - (1 if -1 in labels else 0),
    }


def gaussian_mixture(
    X: np.ndarray,
    n_components: int = 3,
    covariance_type: str = "full",
    max_iter: int = 100,
    random_state: int | None = None,
) -> dict:
    """
    Gaussian Mixture Model clustering.

    Parameters
    ----------
    X : array_like
        Data matrix
    n_components : int
        Number of mixture components
    covariance_type : str
        'full', 'tied', 'diag', 'spherical'
    max_iter : int
        Maximum iterations
    random_state : int, optional
        Random seed

    Returns
    -------
    dict
        Dictionary with labels and probabilities
    """
    from sklearn.mixture import GaussianMixture

    X = np.asarray(X)

    model = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        max_iter=max_iter,
        random_state=random_state,
    )

    model.fit(X)
    labels = model.predict(X)
    probabilities = model.predict_proba(X)

    return {
        "labels": labels,
        "probabilities": probabilities,
        "probs": probabilities,
        "means": model.means_,
        "covariances": model.covariances_,
    }


def silhouette_score(X: np.ndarray, labels: np.ndarray, metric: str = "euclidean") -> float:
    """
    Calculate silhouette score for cluster validation.

    Parameters
    ----------
    X : array_like
        Data matrix
    labels : array_like
        Cluster labels
    metric : str
        Distance metric

    Returns
    -------
    float
        Silhouette score (-1 to 1, higher is better)
    """
    from sklearn.metrics import silhouette_score as sklearn_silhouette

    X = np.asarray(X)
    labels = np.asarray(labels)

    return sklearn_silhouette(X, labels, metric=metric)


def davies_bouldin_index(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Calculate Davies-Bouldin index for cluster validation.

    Parameters
    ----------
    X : array_like
        Data matrix
    labels : array_like
        Cluster labels

    Returns
    -------
    float
        Davies-Bouldin index (lower is better)
    """
    from sklearn.metrics import davies_bouldin_score

    X = np.asarray(X)
    labels = np.asarray(labels)

    return davies_bouldin_score(X, labels)


def calinski_harabasz_score(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Calculate Calinski-Harabasz score for cluster validation.

    Parameters
    ----------
    X : array_like
        Data matrix
    labels : array_like
        Cluster labels

    Returns
    -------
    float
        Calinski-Harabasz score (higher is better)
    """
    from sklearn.metrics import calinski_harabasz_score as sklearn_ch

    X = np.asarray(X)
    labels = np.asarray(labels)

    return sklearn_ch(X, labels)


def elbow_method(X: np.ndarray, max_k: int = 10, random_state: int | None = None) -> dict:
    """
    Elbow method for finding optimal number of clusters.

    Parameters
    ----------
    X : array_like
        Data matrix
    max_k : int
        Maximum number of clusters to try
    random_state : int, optional
        Random seed

    Returns
    -------
    dict
        Dictionary with inertias and optimal_k
    """
    from sklearn.cluster import KMeans

    X = np.asarray(X)

    inertias = []
    for k in range(1, max_k + 1):
        model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        model.fit(X)
        inertias.append(model.inertia_)

    # Simple elbow detection using second derivative
    inertias = np.array(inertias)
    if len(inertias) > 2:
        second_deriv = np.diff(np.diff(inertias))
        optimal_k = np.argmax(second_deriv) + 2  # +2 because of double diff
    else:
        optimal_k = 2

    return {
        "inertias": inertias,
        "wcss": inertias,
        "optimal_k": optimal_k,
        "k_range": list(range(1, max_k + 1)),
    }
