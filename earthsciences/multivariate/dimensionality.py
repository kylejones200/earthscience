"""
Non-linear dimensionality reduction and manifold learning

t-SNE, UMAP, Isomap, and other manifold learning techniques.
"""

import logging

import numpy as np
from sklearn.manifold import MDS, TSNE, Isomap, LocallyLinearEmbedding

logger = logging.getLogger(__name__)


def tsne(
    data: np.ndarray,
    n_components: int = 2,
    perplexity: float = 30.0,
    learning_rate: float = 200.0,
    n_iter: int = 1000,
    random_state: int | None = None,
) -> dict:
    """
    t-Distributed Stochastic Neighbor Embedding (t-SNE).

    Non-linear dimensionality reduction for visualization.
    Preserves local structure.

    Parameters
    ----------
    data : array_like
        Data matrix (n_samples, n_features)
    n_components : int, optional
        Output dimensionality (default=2)
    perplexity : float, optional
        Perplexity parameter (default=30.0, typical 5-50)
    learning_rate : float, optional
        Learning rate (default=200.0)
    n_iter : int, optional
        Number of iterations (default=1000)
    random_state : int, optional
        Random seed

    Returns
    -------
    dict
        t-SNE embedding

    Examples
    --------
    >>> from sklearn.datasets import load_digits
    >>> digits = load_digits()
    >>> result = tsne(digits.data, n_components=2, perplexity=30)
    >>> embedding = result['embedding']
    >>>
    >>> import matplotlib.pyplot as plt
    >>> plt.scatter(embedding[:, 0], embedding[:, 1],
    ...            c=digits.target, cmap='tab10', s=5)
    >>> plt.colorbar()
    >>> plt.title('t-SNE visualization')
    >>> plt.show()
    """
    data = np.asarray(data)

    # Perform t-SNE
    tsne_model = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        learning_rate=learning_rate,
        n_iter=n_iter,
        random_state=random_state,
        verbose=0,
    )

    embedding = tsne_model.fit_transform(data)

    return {
        "embedding": embedding,
        "kl_divergence": tsne_model.kl_divergence_,
        "n_iter": tsne_model.n_iter_,
        "tsne_object": tsne_model,
    }


def umap_embedding(
    data: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    random_state: int | None = None,
) -> dict:
    """
    Uniform Manifold Approximation and Projection (UMAP).

    Fast, scalable dimensionality reduction.
    Preserves both local and global structure better than t-SNE.

    Parameters
    ----------
    data : array_like
        Data matrix
    n_components : int, optional
        Output dimensionality (default=2)
    n_neighbors : int, optional
        Number of neighbors (default=15)
    min_dist : float, optional
        Minimum distance in embedding (default=0.1)
    metric : str, optional
        Distance metric (default='euclidean')
    random_state : int, optional
        Random seed

    Returns
    -------
    dict
        UMAP embedding

    Examples
    --------
    >>> # Requires: pip install umap-learn
    >>> data = np.random.randn(500, 50)
    >>> try:
    ...     result = umap_embedding(data, n_components=2)
    ...     embedding = result['embedding']
    >>> except ImportError:
    ...     print("UMAP not installed. Install with: pip install umap-learn")
    """
    try:
        import umap
    except ImportError:
        raise ImportError("UMAP not installed. Install with: pip install umap-learn")

    data = np.asarray(data)

    # Perform UMAP
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )

    embedding = reducer.fit_transform(data)

    return {
        "embedding": embedding,
        "umap_object": reducer,
    }


def isomap(data: np.ndarray, n_components: int = 2, n_neighbors: int = 5) -> dict:
    """
    Isometric Mapping (Isomap).

    Non-linear dimensionality reduction using geodesic distances.

    Parameters
    ----------
    data : array_like
        Data matrix
    n_components : int, optional
        Output dimensionality (default=2)
    n_neighbors : int, optional
        Number of neighbors (default=5)

    Returns
    -------
    dict
        Isomap embedding

    Examples
    --------
    >>> # Swiss roll dataset
    >>> from sklearn.datasets import make_swiss_roll
    >>> X, color = make_swiss_roll(n_samples=1000)
    >>> result = isomap(X, n_components=2, n_neighbors=10)
    >>>
    >>> import matplotlib.pyplot as plt
    >>> plt.scatter(result['embedding'][:, 0],
    ...            result['embedding'][:, 1], c=color, cmap='viridis')
    >>> plt.title('Isomap Embedding')
    >>> plt.show()
    """
    data = np.asarray(data)

    # Perform Isomap
    iso = Isomap(n_components=n_components, n_neighbors=n_neighbors)
    embedding = iso.fit_transform(data)

    return {
        "embedding": embedding,
        "reconstruction_error": iso.reconstruction_error(),
        "isomap_object": iso,
    }


def mds(data: np.ndarray, n_components: int = 2, metric: bool = True, max_iter: int = 300) -> dict:
    """
    Multidimensional Scaling (MDS).

    Preserve pairwise distances in lower dimension.

    Parameters
    ----------
    data : array_like
        Data matrix or distance matrix
    n_components : int, optional
        Output dimensionality (default=2)
    metric : bool, optional
        If True, perform metric MDS (default=True)
    max_iter : int, optional
        Maximum iterations (default=300)

    Returns
    -------
    dict
        MDS embedding

    Examples
    --------
    >>> # City distances
    >>> from scipy.spatial.distance import pdist, squareform
    >>> cities = np.random.rand(20, 2) * 1000  # Simulated coordinates
    >>> distances = squareform(pdist(cities))
    >>>
    >>> result = mds(distances, n_components=2)
    >>> recovered_coords = result['embedding']
    """
    data = np.asarray(data)

    # Perform MDS
    mds_model = MDS(
        n_components=n_components,
        metric=metric,
        max_iter=max_iter,
        dissimilarity="precomputed" if data.shape[0] == data.shape[1] else "euclidean",
    )

    embedding = mds_model.fit_transform(data)

    return {
        "embedding": embedding,
        "stress": mds_model.stress_,
        "mds_object": mds_model,
    }


def lle(
    data: np.ndarray, n_components: int = 2, n_neighbors: int = 5, method: str = "standard"
) -> dict:
    """
    Locally Linear Embedding (LLE).

    Preserves local neighborhoods in embedding.

    Parameters
    ----------
    data : array_like
        Data matrix
    n_components : int, optional
        Output dimensionality (default=2)
    n_neighbors : int, optional
        Number of neighbors (default=5)
    method : str, optional
        Method: 'standard', 'modified', 'hessian', 'ltsa' (default='standard')

    Returns
    -------
    dict
        LLE embedding

    Examples
    --------
    >>> from sklearn.datasets import make_s_curve
    >>> X, color = make_s_curve(n_samples=1000)
    >>> result = lle(X, n_components=2, n_neighbors=10)
    >>>
    >>> import matplotlib.pyplot as plt
    >>> plt.scatter(result['embedding'][:, 0],
    ...            result['embedding'][:, 1], c=color, cmap='viridis')
    >>> plt.title('LLE Embedding')
    >>> plt.show()
    """
    data = np.asarray(data)

    # Perform LLE
    lle_model = LocallyLinearEmbedding(
        n_components=n_components, n_neighbors=n_neighbors, method=method
    )

    embedding = lle_model.fit_transform(data)

    return {
        "embedding": embedding,
        "reconstruction_error": lle_model.reconstruction_error_,
        "lle_object": lle_model,
    }


def spectral_embedding(
    data: np.ndarray,
    n_components: int = 2,
    affinity: str = "nearest_neighbors",
    n_neighbors: int = 10,
) -> dict:
    """
    Spectral Embedding (Laplacian Eigenmaps).

    Uses graph Laplacian for dimensionality reduction.

    Parameters
    ----------
    data : array_like
        Data matrix
    n_components : int, optional
        Output dimensionality (default=2)
    affinity : str, optional
        Affinity matrix: 'nearest_neighbors', 'rbf' (default='nearest_neighbors')
    n_neighbors : int, optional
        Number of neighbors (default=10)

    Returns
    -------
    dict
        Spectral embedding

    Examples
    --------
    >>> data = np.random.randn(200, 50)
    >>> result = spectral_embedding(data, n_components=2, n_neighbors=10)
    >>> embedding = result['embedding']
    """
    from sklearn.manifold import SpectralEmbedding

    data = np.asarray(data)

    # Perform spectral embedding
    se = SpectralEmbedding(n_components=n_components, affinity=affinity, n_neighbors=n_neighbors)

    embedding = se.fit_transform(data)

    return {
        "embedding": embedding,
        "affinity_matrix": se.affinity_matrix_,
        "se_object": se,
    }


def compare_embeddings(
    data: np.ndarray, methods: list = ["pca", "tsne", "isomap"], n_components: int = 2
) -> dict:
    """
    Compare multiple dimensionality reduction methods.

    Parameters
    ----------
    data : array_like
        Data matrix
    methods : list, optional
        Methods to compare
    n_components : int, optional
        Output dimensionality

    Returns
    -------
    dict
        Embeddings from all methods

    Examples
    --------
    >>> from sklearn.datasets import load_digits
    >>> digits = load_digits()
    >>> results = compare_embeddings(digits.data,
    ...                             methods=['pca', 'tsne', 'isomap'])
    >>>
    >>> import matplotlib.pyplot as plt
    >>> fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    >>> for i, method in enumerate(['pca', 'tsne', 'isomap']):
    ...     emb = results[method]
    ...     axes[i].scatter(emb[:, 0], emb[:, 1],
    ...                    c=digits.target, cmap='tab10', s=5)
    ...     axes[i].set_title(method.upper())
    >>> plt.show()
    """
    from ..multivariate.pca import principal_component_analysis

    data = np.asarray(data)
    results = {}

    for method in methods:
        if method == "pca":
            pca_result = principal_component_analysis(data, n_components=n_components)
            results["pca"] = pca_result["scores"]

        elif method == "tsne":
            tsne_result = tsne(data, n_components=n_components)
            results["tsne"] = tsne_result["embedding"]

        elif method == "isomap":
            iso_result = isomap(data, n_components=n_components)
            results["isomap"] = iso_result["embedding"]

        elif method == "lle":
            lle_result = lle(data, n_components=n_components)
            results["lle"] = lle_result["embedding"]

        elif method == "mds":
            mds_result = mds(data, n_components=n_components)
            results["mds"] = mds_result["embedding"]

        else:
            logger.warning(f"Unknown method: {method}")

    return results
