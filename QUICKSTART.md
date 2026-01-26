# Quick Start Guide

## Installation

```bash
cd "earth science"
pip install -r requirements.txt
```

## Basic Usage

### Import the library

```python
import numpy as np
import earthsciences as es
```

### Statistics

```python
# Descriptive statistics
data = np.random.randn(100)
stats = es.statistics.descriptive_stats(data)
print(f"Mean: {stats['mean']:.3f}, Std: {stats['std']:.3f}")

# Linear regression
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])
result = es.statistics.linear_regression(x, y)
print(f"Slope: {result['slope']:.3f}, R²: {result['r_squared']:.3f}")
```

### Time-Series

```python
# Generate signal and compute power spectrum
t, signal = es.timeseries.generate_test_signal('mixed', duration=1, sampling_rate=1000)
freqs, power = es.timeseries.power_spectrum(signal, sampling_rate=1000)

# Apply filter
filtered = es.timeseries.lowpass_filter(signal, cutoff=20, sampling_rate=1000)
```

### Spatial Data

```python
# Interpolation
x = np.random.rand(30) * 10
y = np.random.rand(30) * 10
values = np.sin(x) + np.cos(y)

grid_x, grid_y = np.meshgrid(np.linspace(0, 10, 50), np.linspace(0, 10, 50))
interpolated = es.spatial.idw_interpolation(x, y, values, grid_x, grid_y)

# Variogram
lags, gamma, n_pairs = es.spatial.compute_variogram(x, y, values)
fit = es.spatial.fit_variogram_model(lags, gamma, model='spherical')
```

### Multivariate

```python
# PCA
from sklearn.datasets import make_blobs
X, _ = make_blobs(n_samples=100, n_features=5, centers=3)

pca_result = es.multivariate.principal_component_analysis(X, n_components=2)
print(f"Variance explained: {pca_result['explained_variance_ratio']}")

# Clustering
clusters = es.multivariate.kmeans_clustering(X, n_clusters=3)
print(f"Silhouette score: {clusters['silhouette_score']:.3f}")
```

### Directional Data

```python
# Circular statistics (angles in degrees)
angles = np.array([10, 350, 5, 15, 355])
mean_dir = es.directional.circular_mean(angles, degrees=True)
print(f"Mean direction: {mean_dir:.1f}°")

# Test for uniformity
test = es.directional.rayleigh_test(angles, degrees=True)
print(f"P-value: {test['p_value']:.4f}")
```

## Run Examples

```bash
# Run example script
cd examples
python example_statistics.py

# Start Jupyter for notebooks
jupyter notebook
```

## Module Structure

```
earthsciences/
├── statistics/          # Univariate & bivariate statistics
│   ├── univariate.py
│   ├── bivariate.py
│   ├── distributions.py
│   ├── hypothesis_tests.py
│   └── resampling.py
├── timeseries/          # Time-series & signal processing
│   ├── spectral.py
│   ├── filtering.py
│   ├── wavelets.py
│   └── signals.py
├── spatial/             # Spatial data analysis
│   ├── interpolation.py
│   ├── kriging.py
│   ├── variogram.py
│   └── point_patterns.py
├── multivariate/        # Multivariate methods
│   ├── pca.py
│   ├── clustering.py
│   └── classification.py
├── directional/         # Circular & spherical statistics
│   ├── circular.py
│   └── spherical.py
└── utils/               # Utilities
```

## Getting Help

- Documentation: See docstrings in each function
- Examples: Check the `examples/` directory
- Issues: Report bugs or request features via GitHub issues

## Next Steps

1. Explore example notebooks in `examples/`
2. Read function docstrings for detailed parameter information
3. Adapt code to your own earth sciences data analysis projects
