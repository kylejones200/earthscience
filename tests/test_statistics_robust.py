"""
Tests for robust statistics module
"""

import pytest
import numpy as np
from earthsciences.statistics import robust_stats


class TestMedianAbsoluteDeviation:
    """Test median absolute deviation."""
    
    def test_basic_functionality(self):
        """Test MAD calculation."""
        data = np.array([1, 2, 3, 4, 5, 100])  # 100 is outlier
        
        mad = robust_stats.mad(data)
        
        assert mad > 0
        assert mad < np.std(data)  # MAD is robust to outliers
    
    def test_no_outliers(self):
        """Test MAD on normal data."""
        np.random.seed(42)
        data = np.random.randn(100)
        
        mad = robust_stats.mad(data)
        std = np.std(data)
        
        # MAD ~= 0.6745 * std for normal data (with scale factor 1.4826)
        # Ratio should be around 0.6745 * 1.4826 = 1.0
        assert 0.5 < mad / std < 1.2


class TestRobustMean:
    """Test robust mean estimators."""
    
    def test_trimmed_mean(self):
        """Test trimmed mean."""
        data = np.array([1, 2, 3, 4, 5, 100, 200])
        
        trimmed = robust_stats.trimmed_mean(data, trim_percent=0.2)
        
        # Should be less affected by outliers
        assert trimmed < np.mean(data)
        assert trimmed > np.median(data)
    
    def test_winsorized_mean(self):
        """Test Winsorized mean."""
        data = np.array([1, 2, 3, 4, 5, 100])
        
        winsorized = robust_stats.winsorized_mean(data, limits=(0.3, 0.3))
        
        # Winsorized mean should be less affected by outlier
        assert winsorized < np.mean(data)
        # Should be between median and mean
        assert np.median(data) <= winsorized < np.mean(data)


class TestHuberM:
    """Test Huber M-estimator."""
    
    def test_location_estimate(self):
        """Test Huber location estimate."""
        np.random.seed(42)
        data = np.random.randn(50)
        data = np.append(data, [10, -10])  # Add outliers
        
        huber_loc = robust_stats.huber_location(data)
        
        # Should be closer to median than mean
        assert abs(huber_loc - np.median(data)) < abs(np.mean(data) - np.median(data))


class TestRobustRegression:
    """Test robust regression methods."""
    
    def test_ransac(self):
        """Test RANSAC regression."""
        np.random.seed(42)
        x = np.linspace(0, 10, 50)
        y = 2*x + 1 + np.random.randn(50) * 0.5
        
        # Add outliers
        y[0] = 100
        y[-1] = -100
        
        result = robust_stats.ransac_regression(x, y)
        
        assert 'slope' in result
        assert 'intercept' in result
        
        # Should recover true slope (~2)
        assert 1.5 < result['slope'] < 2.5
    
    def test_theil_sen(self):
        """Test Theil-Sen estimator."""
        np.random.seed(42)
        x = np.linspace(0, 10, 30)
        y = 3*x - 2 + np.random.randn(30) * 0.3
        
        result = robust_stats.theil_sen_regression(x, y)
        
        assert 'slope' in result
        assert 'intercept' in result
        
        # Should recover slope (~3)
        assert 2.5 < result['slope'] < 3.5


class TestOutlierDetection:
    """Test outlier detection methods."""
    
    def test_iqr_method(self):
        """Test IQR-based outlier detection."""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 100])
        
        outliers = robust_stats.detect_outliers_iqr(data)
        
        assert len(outliers) > 0
        assert 100 in data[outliers]
    
    def test_zscore_method(self):
        """Test Z-score outlier detection."""
        np.random.seed(42)
        data = np.random.randn(100)
        data = np.append(data, [5, -5])  # Add outliers
        
        outliers = robust_stats.detect_outliers_zscore(data, threshold=3)
        
        assert len(outliers) > 0
    
    def test_mad_method(self):
        """Test MAD-based outlier detection."""
        data = np.array([1, 2, 3, 4, 5, 6, 100])
        
        outliers = robust_stats.detect_outliers_mad(data)
        
        assert len(outliers) > 0
        assert 100 in data[outliers]


class TestBiweightEstimators:
    """Test biweight estimators."""
    
    def test_biweight_location(self):
        """Test biweight location estimate."""
        np.random.seed(42)
        data = np.random.randn(50)
        data = np.append(data, [10, -10])
        
        location = robust_stats.biweight_location(data)
        
        assert abs(location) < 1.0  # Should be near 0
    
    def test_biweight_scale(self):
        """Test biweight scale estimate."""
        np.random.seed(42)
        data = np.random.randn(50)
        
        scale = robust_stats.biweight_scale(data)
        
        assert scale > 0
        # Biweight scale should be close to standard deviation for normal data
        assert 0.5 < scale < 10.0
