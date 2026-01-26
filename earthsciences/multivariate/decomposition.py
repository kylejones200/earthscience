"""
Advanced matrix decomposition methods

ICA, NMF, and tensor decomposition for multivariate data.
"""

import numpy as np
from sklearn.decomposition import FastICA, NMF, TruncatedSVD
from typing import Dict, Optional


def independent_component_analysis(data: np.ndarray,
                                   n_components: Optional[int] = None,
                                   max_iter: int = 200,
                                   random_state: Optional[int] = None) -> Dict:
    """
    Independent Component Analysis (ICA) for blind source separation.
    
    Useful for separating mixed signals (e.g., seismic signals, geochemical data).
    
    Parameters
    ----------
    data : array_like
        Data matrix (n_samples, n_features)
    n_components : int, optional
        Number of components (if None, use all)
    max_iter : int, optional
        Maximum iterations (default=200)
    random_state : int, optional
        Random seed
        
    Returns
    -------
    dict
        ICA results with sources and mixing matrix
        
    Examples
    --------
    >>> # Mix three independent sources
    >>> t = np.linspace(0, 10, 1000)
    >>> s1 = np.sin(2*np.pi*t)  # Sine wave
    >>> s2 = np.sign(np.sin(3*np.pi*t))  # Square wave
    >>> s3 = np.random.randn(1000)  # Noise
    >>> S = np.c_[s1, s2, s3]
    >>> 
    >>> # Create mixed signals
    >>> A = np.random.randn(3, 3)
    >>> X = S @ A.T
    >>> 
    >>> # Separate sources
    >>> result = independent_component_analysis(X, n_components=3)
    >>> recovered_sources = result['sources']
    """
    data = np.asarray(data)
    
    if n_components is None:
        n_components = min(data.shape)
    
    # Perform ICA
    ica = FastICA(n_components=n_components, max_iter=max_iter,
                  random_state=random_state)
    sources = ica.fit_transform(data)
    
    # Mixing matrix
    mixing_matrix = ica.mixing_
    
    return {
        'sources': sources,
        'mixing_matrix': mixing_matrix,
        'mean': ica.mean_,
        'ica_object': ica,
    }


def non_negative_matrix_factorization(data: np.ndarray,
                                     n_components: int = 10,
                                     init: str = 'random',
                                     max_iter: int = 200,
                                     random_state: Optional[int] = None) -> Dict:
    """
    Non-negative Matrix Factorization (NMF).
    
    Decomposes data into non-negative components.
    Useful for spectral unmixing, compositional data, etc.
    
    Parameters
    ----------
    data : array_like
        Data matrix (non-negative values)
    n_components : int, optional
        Number of components (default=10)
    init : str, optional
        Initialization: 'random', 'nndsvd' (default='random')
    max_iter : int, optional
        Maximum iterations (default=200)
    random_state : int, optional
        Random seed
        
    Returns
    -------
    dict
        NMF decomposition results
        
    Examples
    --------
    >>> # Spectral data (samples x wavelengths)
    >>> X = np.random.rand(100, 50) * 100  # Non-negative
    >>> result = non_negative_matrix_factorization(X, n_components=5)
    >>> 
    >>> W = result['W']  # Sample loadings
    >>> H = result['H']  # Component spectra
    >>> print(f"Reconstruction error: {result['reconstruction_error']:.2f}")
    """
    data = np.asarray(data)
    
    if np.any(data < 0):
        raise ValueError("NMF requires non-negative data")
    
    # Perform NMF
    model = NMF(n_components=n_components, init=init, max_iter=max_iter,
                random_state=random_state)
    W = model.fit_transform(data)
    H = model.components_
    
    # Reconstruction
    reconstruction = W @ H
    reconstruction_error = model.reconstruction_err_
    
    return {
        'W': W,  # (n_samples, n_components)
        'H': H,  # (n_components, n_features)
        'reconstruction': reconstruction,
        'reconstruction_error': reconstruction_error,
        'n_iter': model.n_iter_,
        'nmf_object': model,
    }


def truncated_svd(data: np.ndarray,
                 n_components: int = 10,
                 algorithm: str = 'randomized') -> Dict:
    """
    Truncated Singular Value Decomposition.
    
    Efficient dimensionality reduction for large sparse matrices.
    
    Parameters
    ----------
    data : array_like
        Data matrix
    n_components : int, optional
        Number of components (default=10)
    algorithm : str, optional
        Algorithm: 'randomized' or 'arpack' (default='randomized')
        
    Returns
    -------
    dict
        SVD results
        
    Examples
    --------
    >>> X = np.random.randn(100, 50)
    >>> result = truncated_svd(X, n_components=10)
    >>> print(f"Explained variance: {result['explained_variance_ratio']}")
    """
    data = np.asarray(data)
    
    # Perform truncated SVD
    svd = TruncatedSVD(n_components=n_components, algorithm=algorithm)
    transformed = svd.fit_transform(data)
    
    return {
        'transformed': transformed,
        'components': svd.components_,
        'singular_values': svd.singular_values_,
        'explained_variance': svd.explained_variance_,
        'explained_variance_ratio': svd.explained_variance_ratio_,
        'svd_object': svd,
    }


def sparse_pca(data: np.ndarray,
              n_components: int = 10,
              alpha: float = 1.0,
              max_iter: int = 100) -> Dict:
    """
    Sparse Principal Component Analysis.
    
    Finds sparse components for better interpretability.
    
    Parameters
    ----------
    data : array_like
        Data matrix
    n_components : int, optional
        Number of components (default=10)
    alpha : float, optional
        Sparsity parameter (default=1.0, higher = more sparse)
    max_iter : int, optional
        Maximum iterations (default=100)
        
    Returns
    -------
    dict
        Sparse PCA results
        
    Examples
    --------
    >>> X = np.random.randn(100, 20)
    >>> result = sparse_pca(X, n_components=5, alpha=1.0)
    >>> print(f"Number of non-zero loadings: {np.sum(result['components'] != 0)}")
    """
    from sklearn.decomposition import SparsePCA
    
    data = np.asarray(data)
    
    # Standardize
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    data_standardized = (data - mean) / (std + 1e-10)
    
    # Perform sparse PCA
    spca = SparsePCA(n_components=n_components, alpha=alpha, max_iter=max_iter)
    transformed = spca.fit_transform(data_standardized)
    
    return {
        'transformed': transformed,
        'components': spca.components_,
        'sparsity': np.sum(spca.components_ == 0) / spca.components_.size,
        'spca_object': spca,
    }


def canonical_correlation_analysis(X: np.ndarray,
                                   Y: np.ndarray,
                                   n_components: int = 2) -> Dict:
    """
    Canonical Correlation Analysis (CCA).
    
    Finds linear combinations of two sets of variables that are maximally correlated.
    
    Parameters
    ----------
    X, Y : array_like
        Two data matrices (same number of samples)
    n_components : int, optional
        Number of canonical components (default=2)
        
    Returns
    -------
    dict
        CCA results
        
    Examples
    --------
    >>> # Two related datasets
    >>> X = np.random.randn(100, 5)
    >>> Y = X @ np.random.randn(5, 3) + np.random.randn(100, 3)*0.5
    >>> result = canonical_correlation_analysis(X, Y, n_components=2)
    >>> print(f"Canonical correlations: {result['correlations']}")
    """
    from sklearn.cross_decomposition import CCA
    
    X = np.asarray(X)
    Y = np.asarray(Y)
    
    if len(X) != len(Y):
        raise ValueError("X and Y must have same number of samples")
    
    # Perform CCA
    cca = CCA(n_components=n_components)
    X_c, Y_c = cca.fit_transform(X, Y)
    
    # Calculate canonical correlations
    correlations = np.array([np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1]
                            for i in range(n_components)])
    
    return {
        'X_canonical': X_c,
        'Y_canonical': Y_c,
        'correlations': correlations,
        'X_weights': cca.x_weights_,
        'Y_weights': cca.y_weights_,
        'cca_object': cca,
    }


def kernel_pca(data: np.ndarray,
              n_components: int = 2,
              kernel: str = 'rbf',
              gamma: Optional[float] = None) -> Dict:
    """
    Kernel PCA for non-linear dimensionality reduction.
    
    Parameters
    ----------
    data : array_like
        Data matrix
    n_components : int, optional
        Number of components (default=2)
    kernel : str, optional
        Kernel: 'rbf', 'poly', 'sigmoid', 'cosine' (default='rbf')
    gamma : float, optional
        Kernel coefficient
        
    Returns
    -------
    dict
        Kernel PCA results
        
    Examples
    --------
    >>> # Non-linear data
    >>> theta = np.linspace(0, 4*np.pi, 100)
    >>> X = np.c_[theta * np.cos(theta), theta * np.sin(theta)]
    >>> X += np.random.randn(100, 2) * 0.1
    >>> 
    >>> result = kernel_pca(X, n_components=2, kernel='rbf')
    >>> transformed = result['transformed']
    """
    from sklearn.decomposition import KernelPCA
    
    data = np.asarray(data)
    
    # Perform kernel PCA
    kpca = KernelPCA(n_components=n_components, kernel=kernel, gamma=gamma)
    transformed = kpca.fit_transform(data)
    
    return {
        'transformed': transformed,
        'eigenvalues': kpca.eigenvalues_,
        'eigenvectors': kpca.eigenvectors_,
        'kpca_object': kpca,
    }


def dictionary_learning(data: np.ndarray,
                       n_components: int = 10,
                       alpha: float = 1.0,
                       max_iter: int = 100) -> Dict:
    """
    Dictionary learning for sparse representation.
    
    Learns a dictionary of atoms that can sparsely represent the data.
    
    Parameters
    ----------
    data : array_like
        Data matrix
    n_components : int, optional
        Number of dictionary atoms (default=10)
    alpha : float, optional
        Sparsity parameter (default=1.0)
    max_iter : int, optional
        Maximum iterations (default=100)
        
    Returns
    -------
    dict
        Dictionary learning results
        
    Examples
    --------
    >>> X = np.random.randn(100, 50)
    >>> result = dictionary_learning(X, n_components=20, alpha=1.0)
    >>> print(f"Dictionary shape: {result['dictionary'].shape}")
    >>> print(f"Sparse code shape: {result['code'].shape}")
    """
    from sklearn.decomposition import DictionaryLearning
    
    data = np.asarray(data)
    
    # Perform dictionary learning
    dl = DictionaryLearning(n_components=n_components, alpha=alpha,
                           max_iter=max_iter)
    code = dl.fit_transform(data)
    dictionary = dl.components_
    
    return {
        'code': code,  # Sparse representation
        'dictionary': dictionary,  # Learned dictionary
        'error': dl.error_,
        'n_iter': dl.n_iter_,
        'dl_object': dl,
    }
