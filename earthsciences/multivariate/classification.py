"""
Classification and discriminant analysis 

Supervised learning methods for classification.
"""

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, classification_report
from typing import Dict, Tuple, Optional


def discriminant_analysis(X_train: np.ndarray,
                         y_train: np.ndarray,
                         X_test: Optional[np.ndarray] = None,
                         method: str = 'linear') -> Dict:
    """
    Linear or Quadratic Discriminant Analysis.
    
    Parameters
    ----------
    X_train : array_like
        Training features (n_samples, n_features)
    y_train : array_like
        Training labels
    X_test : array_like, optional
        Test features (if None, use training data)
    method : str, optional
        'linear' or 'quadratic' (default='linear')
        
    Returns
    -------
    dict
        Classification results
        
    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=200, n_features=5,
    ...                            n_informative=3, n_redundant=0,
    ...                            n_classes=3, random_state=42)
    >>> 
    >>> # Split data
    >>> from sklearn.model_selection import train_test_split
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y, test_size=0.3, random_state=42)
    >>> 
    >>> # LDA
    >>> result = discriminant_analysis(X_train, y_train, X_test, method='linear')
    >>> print(f"Training accuracy: {result['train_accuracy']:.3f}")
    >>> print(f"Test accuracy: {result['test_accuracy']:.3f}")
    """
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    
    if X_test is None:
        X_test = X_train
    else:
        X_test = np.asarray(X_test)
    
    # Choose method
    if method == 'linear':
        model = LinearDiscriminantAnalysis()
    elif method == 'quadratic':
        model = QuadraticDiscriminantAnalysis()
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Fit model
    model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Probabilities
    y_train_proba = model.predict_proba(X_train)
    y_test_proba = model.predict_proba(X_test)
    
    # Accuracy
    train_accuracy = np.mean(y_train_pred == y_train)
    
    return {
        'model': model,
        'train_predictions': y_train_pred,
        'test_predictions': y_test_pred,
        'train_probabilities': y_train_proba,
        'test_probabilities': y_test_proba,
        'train_accuracy': train_accuracy,
        'classes': model.classes_,
    }


def confusion_matrix_metrics(y_true: np.ndarray,
                            y_pred: np.ndarray,
                            labels: Optional[list] = None) -> Dict:
    """
    Calculate confusion matrix and related metrics.
    
    Parameters
    ----------
    y_true : array_like
        True labels
    y_pred : array_like
        Predicted labels
    labels : list, optional
        List of label names
        
    Returns
    -------
    dict
        Confusion matrix and metrics
        
    Examples
    --------
    >>> y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    >>> y_pred = np.array([0, 2, 2, 0, 1, 1, 0, 1, 2])
    >>> metrics = confusion_matrix_metrics(y_true, y_pred)
    >>> print(f"Accuracy: {metrics['accuracy']:.3f}")
    >>> print(f"Confusion matrix:\\n{metrics['confusion_matrix']}")
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Overall accuracy
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    
    # Per-class metrics
    n_classes = cm.shape[0]
    precision = np.zeros(n_classes)
    recall = np.zeros(n_classes)
    f1_score = np.zeros(n_classes)
    
    for i in range(n_classes):
        # True positives
        tp = cm[i, i]
        
        # False positives
        fp = np.sum(cm[:, i]) - tp
        
        # False negatives
        fn = np.sum(cm[i, :]) - tp
        
        # Precision
        precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        # Recall
        recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # F1 score
        f1_score[i] = (2 * precision[i] * recall[i] / 
                       (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0)
    
    # Generate classification report
    report = classification_report(y_true, y_pred, output_dict=True)
    
    return {
        'confusion_matrix': cm,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'classification_report': report,
    }


def cross_validation_classification(X: np.ndarray,
                                   y: np.ndarray,
                                   model,
                                   k_folds: int = 5,
                                   random_state: Optional[int] = None) -> Dict:
    """
    K-fold cross-validation for classification.
    
    Parameters
    ----------
    X : array_like
        Features
    y : array_like
        Labels
    model : sklearn classifier
        Classification model with fit/predict methods
    k_folds : int, optional
        Number of folds (default=5)
    random_state : int, optional
        Random seed
        
    Returns
    -------
    dict
        Cross-validation results
        
    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    >>> X, y = make_classification(n_samples=200, n_features=10,
    ...                            n_classes=3, random_state=42)
    >>> 
    >>> model = LinearDiscriminantAnalysis()
    >>> result = cross_validation_classification(X, y, model, k_folds=5)
    >>> print(f"Mean accuracy: {result['mean_accuracy']:.3f} ± {result['std_accuracy']:.3f}")
    """
    from sklearn.model_selection import cross_val_score, KFold
    
    X = np.asarray(X)
    y = np.asarray(y)
    
    # K-fold cross-validation
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)
    
    # Calculate scores
    scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
    
    return {
        'scores': scores,
        'mean_accuracy': np.mean(scores),
        'std_accuracy': np.std(scores),
        'k_folds': k_folds,
    }


def mahalanobis_distance(X: np.ndarray,
                        center: Optional[np.ndarray] = None,
                        cov: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Calculate Mahalanobis distance from center (used in discriminant analysis).
    
    Parameters
    ----------
    X : array_like
        Data points
    center : array_like, optional
        Center point (if None, use mean of X)
    cov : array_like, optional
        Covariance matrix (if None, compute from X)
        
    Returns
    -------
    ndarray
        Mahalanobis distances
        
    Examples
    --------
    >>> X = np.random.randn(100, 3)
    >>> distances = mahalanobis_distance(X)
    >>> # Points with large distances are outliers
    >>> outliers = distances > np.percentile(distances, 95)
    >>> print(f"Number of outliers: {np.sum(outliers)}")
    """
    from scipy.spatial.distance import mahalanobis as scipy_mahal
    
    X = np.asarray(X)
    
    if center is None:
        center = np.mean(X, axis=0)
    
    if cov is None:
        cov = np.cov(X.T)
    
    # Inverse covariance
    cov_inv = np.linalg.inv(cov)
    
    # Calculate distances
    distances = np.array([scipy_mahal(x, center, cov_inv) for x in X])
    
    return distances


def roc_curve_analysis(y_true: np.ndarray,
                       y_proba: np.ndarray,
                       pos_label: int = 1) -> Dict:
    """
    Calculate ROC curve and AUC for binary classification.
    
    Parameters
    ----------
    y_true : array_like
        True binary labels
    y_proba : array_like
        Predicted probabilities for positive class
    pos_label : int, optional
        Label of positive class (default=1)
        
    Returns
    -------
    dict
        ROC curve data and AUC
        
    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    >>> from sklearn.model_selection import train_test_split
    >>> 
    >>> X, y = make_classification(n_samples=200, n_classes=2, random_state=42)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    >>> 
    >>> model = LinearDiscriminantAnalysis()
    >>> model.fit(X_train, y_train)
    >>> y_proba = model.predict_proba(X_test)[:, 1]
    >>> 
    >>> roc = roc_curve_analysis(y_test, y_proba)
    >>> print(f"AUC: {roc['auc']:.3f}")
    """
    from sklearn.metrics import roc_curve, auc
    
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_proba, pos_label=pos_label)
    
    # Calculate AUC
    roc_auc = auc(fpr, tpr)
    
    # Find optimal threshold (Youden's J statistic)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    return {
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds,
        'auc': roc_auc,
        'optimal_threshold': optimal_threshold,
        'optimal_tpr': tpr[optimal_idx],
        'optimal_fpr': fpr[optimal_idx],
    }


def lda_projection(X_train: np.ndarray,
                  y_train: np.ndarray,
                  X_test: Optional[np.ndarray] = None,
                  n_components: Optional[int] = None) -> Dict:
    """
    Project data onto Linear Discriminant Analysis axes.
    
    Useful for visualization and dimensionality reduction while
    maximizing class separability.
    
    Parameters
    ----------
    X_train : array_like
        Training features
    y_train : array_like
        Training labels
    X_test : array_like, optional
        Test features to project
    n_components : int, optional
        Number of components (max = n_classes - 1)
        
    Returns
    -------
    dict
        Projected data and discriminant axes
        
    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=200, n_features=10,
    ...                            n_classes=3, random_state=42)
    >>> 
    >>> result = lda_projection(X, y, n_components=2)
    >>> X_proj = result['X_train_projected']
    >>> print(f"Projected shape: {X_proj.shape}")  # (200, 2)
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    
    # LDA with dimensionality reduction
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    X_train_proj = lda.fit_transform(X_train, y_train)
    
    result = {
        'X_train_projected': X_train_proj,
        'explained_variance_ratio': lda.explained_variance_ratio_,
        'scalings': lda.scalings_,  # Discriminant axes
        'means': lda.means_,  # Class means
        'lda_object': lda,
    }
    
    if X_test is not None:
        X_test = np.asarray(X_test)
        X_test_proj = lda.transform(X_test)
        result['X_test_projected'] = X_test_proj
    
    return result
