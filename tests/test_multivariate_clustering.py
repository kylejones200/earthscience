"""
Tests for multivariate clustering module
"""

import pytest
import numpy as np
from earthsciences.multivariate import clustering


class TestKMeans:
    """Test k-means clustering."""
    
    def test_basic_functionality(self):
        """Test basic k-means."""
        np.random.seed(42)
        X = np.vstack([
            np.random.randn(30, 2),
            np.random.randn(30, 2) + 5
        ])
        
        result = clustering.kmeans(X, n_clusters=2)
        
        assert 'labels' in result
        assert 'centroids' in result or 'centers' in result
        assert len(result['labels']) == len(X)
        assert len(np.unique(result['labels'])) == 2
    
    def test_separates_clusters(self):
        """Test that k-means separates obvious clusters."""
        np.random.seed(42)
        cluster1 = np.random.randn(50, 2)
        cluster2 = np.random.randn(50, 2) + 10
        X = np.vstack([cluster1, cluster2])
        
        result = clustering.kmeans(X, n_clusters=2)
        labels = result['labels']
        
        # Most of first 50 should be one cluster
        assert np.sum(labels[:50] == labels[0]) > 40 or np.sum(labels[:50] == labels[1]) > 40


class TestHierarchicalClustering:
    """Test hierarchical clustering."""
    
    def test_basic_functionality(self):
        """Test hierarchical clustering."""
        np.random.seed(42)
        X = np.random.randn(20, 3)
        
        result = clustering.hierarchical(X, n_clusters=3)
        
        assert 'labels' in result
        assert 'linkage' in result or 'dendrogram' in result
        assert len(result['labels']) == len(X)
    
    def test_different_linkages(self):
        """Test different linkage methods."""
        np.random.seed(42)
        X = np.random.randn(15, 2)
        
        for method in ['single', 'complete', 'average', 'ward']:
            result = clustering.hierarchical(X, n_clusters=3, linkage=method)
            assert len(result['labels']) == len(X)


class TestDBSCAN:
    """Test DBSCAN clustering."""
    
    def test_basic_functionality(self):
        """Test DBSCAN."""
        np.random.seed(42)
        X = np.vstack([
            np.random.randn(30, 2),
            np.random.randn(30, 2) + 5
        ])
        
        result = clustering.dbscan(X, eps=1.0, min_samples=5)
        
        assert 'labels' in result
        assert len(result['labels']) == len(X)
    
    def test_noise_detection(self):
        """Test that DBSCAN detects noise points."""
        np.random.seed(42)
        # Dense cluster + scattered noise
        cluster = np.random.randn(50, 2)
        noise = np.random.randn(10, 2) * 10
        X = np.vstack([cluster, noise])
        
        result = clustering.dbscan(X, eps=1.5, min_samples=5)
        labels = result['labels']
        
        # Should have some noise points (-1)
        assert -1 in labels


class TestGaussianMixture:
    """Test Gaussian mixture models."""
    
    def test_basic_functionality(self):
        """Test GMM clustering."""
        np.random.seed(42)
        X = np.vstack([
            np.random.randn(40, 2),
            np.random.randn(40, 2) + 3
        ])
        
        result = clustering.gaussian_mixture(X, n_components=2)
        
        assert 'labels' in result
        assert 'probabilities' in result or 'probs' in result
        assert len(result['labels']) == len(X)


class TestClusterValidation:
    """Test cluster validation metrics."""
    
    def test_silhouette_score(self):
        """Test silhouette score calculation."""
        np.random.seed(42)
        X = np.vstack([
            np.random.randn(30, 2),
            np.random.randn(30, 2) + 5
        ])
        labels = np.array([0]*30 + [1]*30)
        
        score = clustering.silhouette_score(X, labels)
        
        assert -1 <= score <= 1
        assert score > 0.3  # Should be reasonably good
    
    def test_davies_bouldin_index(self):
        """Test Davies-Bouldin index."""
        np.random.seed(42)
        X = np.random.randn(50, 2)
        labels = np.random.randint(0, 3, 50)
        
        score = clustering.davies_bouldin_index(X, labels)
        
        assert score >= 0
    
    def test_calinski_harabasz_score(self):
        """Test Calinski-Harabasz score."""
        np.random.seed(42)
        X = np.vstack([
            np.random.randn(25, 2),
            np.random.randn(25, 2) + 4
        ])
        labels = np.array([0]*25 + [1]*25)
        
        score = clustering.calinski_harabasz_score(X, labels)
        
        assert score > 0


class TestElbowMethod:
    """Test elbow method for optimal k."""
    
    def test_find_optimal_k(self):
        """Test elbow method."""
        np.random.seed(42)
        X = np.vstack([
            np.random.randn(30, 2),
            np.random.randn(30, 2) + 5,
            np.random.randn(30, 2) + 10
        ])
        
        result = clustering.elbow_method(X, max_k=10)
        
        assert 'inertias' in result or 'wcss' in result
        assert 'optimal_k' in result or len(result['inertias']) == 10
