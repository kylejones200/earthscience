"""
Principal Component Analysis 

Dimensionality reduction and data exploration.
"""

import numpy as np
from sklearn.decomposition import PCA
from typing import Dict, Tuple, Optional


def principal_component_analysis(data: np.ndarray,
                                 n_components: Optional[int] = None,
                                 standardize: bool = True) -> Dict:
    """
    Perform Principal Component Analysis.
    
    Parameters
    ----------
    data : array_like
        Data matrix (n_samples, n_features)
    n_components : int, optional
        Number of components to keep (if None, keep all)
    standardize : bool, optional
        Standardize variables to zero mean and unit variance (default=True)
        
    Returns
    -------
    dict
        Dictionary containing:
        - scores: principal component scores
        - loadings: principal component loadings
        - explained_variance: variance explained by each PC
        - explained_variance_ratio: proportion of variance explained
        - cumulative_variance: cumulative proportion of variance
        - eigenvalues: eigenvalues of covariance matrix
        - pca_object: sklearn PCA object
        - mean: mean values used for centering (if standardize=True)
        
    Warnings
    --------
    **DATA LEAKAGE PREVENTION**
    
    When using PCA for predictive modeling with train/test splits:
    
    1. **NEVER fit PCA on combined train+test data**
    2. **Always fit PCA on training data only**
    3. **Transform test data using training statistics**
    
    Incorrect usage (causes data leakage):
    
    >>> # ❌ WRONG - test statistics leak into training
    >>> X_full = np.concatenate([X_train, X_test])
    >>> pca_result = principal_component_analysis(X_full)  # Leakage!
    >>> scores = pca_result['scores']
    >>> train_scores, test_scores = scores[:n_train], scores[n_train:]
    
    Correct usage (no leakage):
    
    >>> # ✅ CORRECT - fit on training data only
    >>> from sklearn.model_selection import train_test_split
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    >>> 
    >>> # Fit PCA on training data
    >>> pca_result = principal_component_analysis(X_train, n_components=3)
    >>> X_train_pca = pca_result['scores']
    >>> 
    >>> # Transform test data using training statistics
    >>> pca_obj = pca_result['pca_object']
    >>> if standardize:
    ...     # Standardize test data using training mean/std
    ...     X_test_standardized = (X_test - np.mean(X_train, axis=0)) / np.std(X_train, axis=0, ddof=1)
    ...     X_test_pca = pca_obj.transform(X_test_standardized)
    ... else:
    ...     X_test_pca = pca_obj.transform(X_test)
    
    For exploratory analysis (no prediction), using full dataset is acceptable.
        
    Examples
    --------
    >>> # Generate correlated data
    >>> np.random.seed(42)
    >>> X = np.random.randn(100, 5)
    >>> X[:, 1] = X[:, 0] + np.random.randn(100) * 0.1
    >>> 
    >>> # Perform PCA for exploratory analysis
    >>> result = principal_component_analysis(X, n_components=3)
    >>> print(f"Variance explained: {result['explained_variance_ratio']}")
    >>> print(f"Cumulative variance: {result['cumulative_variance']}")
    """
    data = np.asarray(data)
    
    if standardize:
        # Standardize data
        data_mean = np.mean(data, axis=0)
        data_std = np.std(data, axis=0, ddof=1)
        data_standardized = (data - data_mean) / data_std
    else:
        data_standardized = data
    
    # Perform PCA
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(data_standardized)
    
    # Get loadings
    loadings = pca.components_.T
    
    # Variance explained
    explained_variance = pca.explained_variance_
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    return {
        'scores': scores,
        'loadings': loadings,
        'explained_variance': explained_variance,
        'explained_variance_ratio': explained_variance_ratio,
        'cumulative_variance': cumulative_variance,
        'eigenvalues': explained_variance,
        'mean': pca.mean_,
        'pca_object': pca,
    }


def pca_biplot(data: np.ndarray,
              feature_names: Optional[list] = None,
              pc1: int = 0,
              pc2: int = 1) -> Dict:
    """
    Prepare data for PCA biplot.
    
    A biplot shows both observations (scores) and variables (loadings).
    
    Parameters
    ----------
    data : array_like
        Data matrix
    feature_names : list, optional
        Names of features
    pc1, pc2 : int, optional
        Which principal components to plot (default=0, 1)
        
    Returns
    -------
    dict
        Data for creating biplot
        
    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> X = np.random.randn(50, 4)
    >>> feature_names = ['Var1', 'Var2', 'Var3', 'Var4']
    >>> biplot_data = pca_biplot(X, feature_names, pc1=0, pc2=1)
    >>> 
    >>> # Plot scores
    >>> plt.scatter(biplot_data['scores'][:, 0],
    ...            biplot_data['scores'][:, 1], alpha=0.5)
    >>> 
    >>> # Plot loadings
    >>> for i, (x, y) in enumerate(biplot_data['loadings']):
    ...     plt.arrow(0, 0, x, y, head_width=0.1, head_length=0.1, fc='r')
    ...     plt.text(x*1.1, y*1.1, biplot_data['feature_names'][i])
    >>> 
    >>> plt.xlabel(f"PC{pc1+1}")
    >>> plt.ylabel(f"PC{pc2+1}")
    >>> plt.show()
    """
    result = principal_component_analysis(data)
    
    if feature_names is None:
        feature_names = [f'Var{i+1}' for i in range(data.shape[1])]
    
    # Scale loadings for visualization
    loading_scale = 3  # Adjust this to make arrows visible
    loadings_plot = result['loadings'][:, [pc1, pc2]] * loading_scale
    
    return {
        'scores': result['scores'][:, [pc1, pc2]],
        'loadings': loadings_plot,
        'feature_names': feature_names,
        'explained_variance': result['explained_variance_ratio'][[pc1, pc2]],
    }


def scree_plot_data(data: np.ndarray) -> Dict:
    """
    Prepare data for scree plot (variance explained by each PC).
    
    Parameters
    ----------
    data : array_like
        Data matrix
        
    Returns
    -------
    dict
        Data for scree plot
        
    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> X = np.random.randn(100, 10)
    >>> scree_data = scree_plot_data(X)
    >>> 
    >>> # Scree plot
    >>> plt.figure(figsize=(10, 4))
    >>> plt.subplot(1, 2, 1)
    >>> plt.plot(range(1, len(scree_data['explained_variance'])+1),
    ...         scree_data['explained_variance'], 'o-')
    >>> plt.xlabel('Principal Component')
    >>> plt.ylabel('Variance Explained')
    >>> plt.title('Scree Plot')
    >>> 
    >>> plt.subplot(1, 2, 2)
    >>> plt.plot(range(1, len(scree_data['cumulative_variance'])+1),
    ...         scree_data['cumulative_variance'], 'o-')
    >>> plt.xlabel('Principal Component')
    >>> plt.ylabel('Cumulative Variance')
    >>> plt.axhline(y=0.95, color='r', linestyle='--', label='95%')
    >>> plt.legend()
    >>> plt.show()
    """
    result = principal_component_analysis(data)
    
    return {
        'explained_variance': result['explained_variance_ratio'],
        'cumulative_variance': result['cumulative_variance'],
        'eigenvalues': result['eigenvalues'],
    }


def factor_analysis(data: np.ndarray,
                   n_factors: int = 2) -> Dict:
    """
    Perform Factor Analysis (similar to PCA but with different assumptions).
    
    Parameters
    ----------
    data : array_like
        Data matrix
    n_factors : int, optional
        Number of factors (default=2)
        
    Returns
    -------
    dict
        Factor analysis results
        
    Warnings
    --------
    **DATA LEAKAGE PREVENTION**
    
    Same precautions as PCA apply: fit on training data only when doing
    predictive modeling. See principal_component_analysis() for details.
        
    Examples
    --------
    >>> X = np.random.randn(100, 6)
    >>> result = factor_analysis(X, n_factors=2)
    >>> print(f"Loadings shape: {result['loadings'].shape}")
    """
    from sklearn.decomposition import FactorAnalysis
    
    data = np.asarray(data)
    
    # Standardize
    data_mean = np.mean(data, axis=0)
    data_std = np.std(data, axis=0, ddof=1)
    data_standardized = (data - data_mean) / data_std
    
    # Factor analysis
    fa = FactorAnalysis(n_components=n_factors)
    scores = fa.fit_transform(data_standardized)
    
    return {
        'scores': scores,
        'loadings': fa.components_.T,
        'noise_variance': fa.noise_variance_,
        'fa_object': fa,
    }


def rotation_varimax(loadings: np.ndarray,
                    max_iter: int = 100,
                    tol: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    """
    Varimax rotation of factor loadings for better interpretability.
    
    Parameters
    ----------
    loadings : array_like
        Factor loadings matrix
    max_iter : int, optional
        Maximum iterations (default=100)
    tol : float, optional
        Convergence tolerance (default=1e-6)
        
    Returns
    -------
    rotated_loadings : ndarray
        Rotated loadings
    rotation_matrix : ndarray
        Rotation matrix
        
    Examples
    --------
    >>> result = principal_component_analysis(X, n_components=3)
    >>> rotated, R = rotation_varimax(result['loadings'])
    """
    from scipy.linalg import svd
    
    loadings = np.asarray(loadings)
    n_vars, n_factors = loadings.shape
    
    rotation_matrix = np.eye(n_factors)
    
    for _ in range(max_iter):
        # Rotate loadings
        rotated = loadings @ rotation_matrix
        
        # Varimax criterion
        normalized = rotated / np.sqrt(np.sum(rotated**2, axis=0))
        
        # SVD for rotation update
        u, _, vt = svd(loadings.T @ (normalized * rotated**2 - 
                                     (np.sum(normalized * rotated**2, axis=0) / n_vars)))
        
        new_rotation = u @ vt
        
        # Check convergence
        if np.allclose(rotation_matrix, new_rotation, atol=tol):
            break
        
        rotation_matrix = new_rotation
    
    rotated_loadings = loadings @ rotation_matrix
    
    return rotated_loadings, rotation_matrix
