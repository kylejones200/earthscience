# Earth Sciences Examples

This directory contains Jupyter notebooks and Python scripts demonstrating the usage of the earthsciences library.

## Quick Start Examples

### Statistics

```python
import numpy as np
import earthsciences as es

# Generate data
data = np.random.randn(100)

# Descriptive statistics
stats = es.statistics.descriptive_stats(data)
print(f"Mean: {stats['mean']:.3f}, Std: {stats['std']:.3f}")

# Linear regression
x = np.array([1, 2, 3, 4, 5])
y = np.array([2.1, 3.9, 6.2, 7.8, 10.1])
result = es.statistics.linear_regression(x, y)
print(f"Slope: {result['slope']:.3f}, R²: {result['r_squared']:.3f}")

# Bootstrap confidence intervals
boot = es.statistics.bootstrap(data, statistic=np.mean, n_iterations=1000)
print(f"95% CI: {boot['confidence_interval']}")
```

### Time-Series Analysis

```python
# Generate test signal
t, signal = es.timeseries.generate_test_signal('mixed', duration=10, 
                                               sampling_rate=1000)

# Power spectrum
freqs, power = es.timeseries.power_spectrum(signal, sampling_rate=1000, 
                                            method='welch')

# Lowpass filter
filtered = es.timeseries.lowpass_filter(signal, cutoff=15, 
                                       sampling_rate=1000)

# Wavelet transform
result = es.timeseries.wavelet_transform(signal, sampling_rate=1000)
```

### Spatial Analysis

```python
# Random spatial data
x = np.random.rand(50) * 10
y = np.random.rand(50) * 10
values = np.sin(x) + np.cos(y)

# Create grid
grid_x, grid_y = np.meshgrid(np.linspace(0, 10, 50),
                             np.linspace(0, 10, 50))

# IDW interpolation
interpolated = es.spatial.idw_interpolation(x, y, values, 
                                           grid_x, grid_y, power=2)

# Compute variogram
lags, gamma, n_pairs = es.spatial.compute_variogram(x, y, values)

# Fit variogram model
fit = es.spatial.fit_variogram_model(lags, gamma, model='spherical')
print(f"Range: {fit['range']:.2f}, Sill: {fit['sill']:.2f}")
```

### Multivariate Analysis

```python
from sklearn.datasets import make_blobs

# Generate data
X, y = make_blobs(n_samples=200, centers=3, n_features=5, random_state=42)

# PCA
pca_result = es.multivariate.principal_component_analysis(X, n_components=2)
print(f"Variance explained: {pca_result['cumulative_variance']}")

# K-means clustering
clusters = es.multivariate.kmeans_clustering(X, n_clusters=3)
print(f"Silhouette score: {clusters['silhouette_score']:.3f}")

# Hierarchical clustering
hier = es.multivariate.hierarchical_clustering(X, n_clusters=3, method='ward')
```

### Directional Data

```python
# Circular data (e.g., wind directions)
angles = np.array([10, 350, 5, 15, 355])  # Degrees

# Circular mean
mean_dir = es.directional.circular_mean(angles, degrees=True)
print(f"Mean direction: {mean_dir:.1f}°")

# Rayleigh test for uniformity
test_result = es.directional.rayleigh_test(angles, degrees=True)
print(f"P-value: {test_result['p_value']:.4f}")

# Fit von Mises distribution
fit = es.directional.von_mises_fit(angles, degrees=True)
print(f"mu: {fit['mu']:.1f}°, kappa: {fit['kappa']:.2f}")

# Spherical data (e.g., paleomagnetic)
theta = np.array([45, 50, 42, 48])  # Colatitude
phi = np.array([10, 15, 8, 12])     # Azimuth
mean_theta, mean_phi = es.directional.spherical_mean(theta, phi, degrees=True)
```

## Python Scripts

### Core Examples

- **`example_statistics.py`** - Basic statistical demonstrations
- **`01_statistics_basics.ipynb`** - Interactive statistics notebook

### Domain-Specific Examples

1. **`05_geochronology.py`** - Radiometric Dating
   - Radioactive decay calculations
   - U-Pb concordia diagrams
   - Rb-Sr isochron dating
   - Radiocarbon calibration
   
2. **`06_directional_statistics.py`** - Structural & Paleomagnetic Data
   - Joint orientation analysis (rose diagrams)
   - Paleomagnetic directions (Fisher distribution)
   - Wind pattern analysis
   - Fault slip vectors

3. **`07_timeseries_analysis.py`** - Seismic & Climate Signals
   - Digital filtering (lowpass, highpass, bandpass)
   - Spectral analysis (Welch, multitaper)
   - Wavelet transforms (time-frequency)
   - Lomb-Scargle (irregular sampling)

4. **`08_image_analysis.py`** - Remote Sensing & Microscopy
   - Vegetation indices (NDVI, EVI, SAVI)
   - Image enhancement techniques
   - Grain size distribution
   - Shape and fabric analysis

### Geochemistry Examples

See `README_GEOCHEMISTRY.md` for detailed geochemistry tutorials:

- **`geochem_00_quickstart.py`** - Quick introduction
- **`geochem_01_exploratory.py`** - Data exploration and statistics
- **`geochem_02_bivariate.py`** - Element correlations
- **`geochem_03_spatial.py`** - Geochemical mapping (kriging)
- **`geochem_04_multivariate.py`** - PCA and clustering

### Performance Demos

- **`demo_kriging_performance.py`** - Optimized kriging benchmark

## Running the Examples

```bash
# Install dependencies
pip install -r ../requirements.txt

# Run any Python script
python 05_geochronology.py
python 06_directional_statistics.py
python 07_timeseries_analysis.py
python 08_image_analysis.py

# Or for geochemistry examples
python geochem_01_exploratory.py

# Start Jupyter for notebooks
jupyter notebook
```

## Generated Outputs

Each example generates publication-quality plots (300 DPI):

**Geochronology (Example 5)**
- `05_geochronology_decay.png` - Decay curves for dating systems
- `05_geochronology_concordia.png` - U-Pb concordia diagram
- `05_geochronology_isochron.png` - Rb-Sr isochron
- `05_geochronology_radiocarbon.png` - Radiocarbon decay

**Directional Statistics (Example 6)**
- `06_directional_joints.png` - Rose diagrams for joints
- `06_directional_paleomag.png` - Paleomagnetic directions
- `06_directional_winds.png` - Wind rose diagrams
- `06_directional_faults.png` - Fault slip analysis

**Time-Series (Example 7)**
- `07_timeseries_filtering.png` - Signal filtering examples
- `07_timeseries_spectral.png` - Power spectra
- `07_timeseries_wavelet.png` - Wavelet scalogram
- `07_timeseries_lombscargle.png` - Periodogram

**Image Analysis (Example 8)**
- `08_image_vegetation_indices.png` - NDVI, EVI, SAVI
- `08_image_enhancement.png` - Enhancement techniques
- `08_image_grain_analysis.png` - Automated grain analysis

## Example Gallery

### Geochronology
Demonstrates radiometric dating methods essential for determining the age of rocks and geological events. Includes U-Pb, Rb-Sr, and radiocarbon dating with concordia diagrams and isochrons.

### Directional Statistics
Shows how to analyze orientation data in structural geology and paleomagnetism. Features rose diagrams, Fisher distributions, and statistical tests for directional data.

### Time-Series Analysis
Covers signal processing for seismic, climate, and other time-varying earth science data. Includes filtering, spectral analysis, and wavelet transforms.

### Image Analysis
Demonstrates remote sensing and microscopy applications. Features vegetation indices from satellite data and automated grain analysis from thin sections.

### Geochemistry
Comprehensive workflows for geochemical data analysis using real datasets. Covers everything from data loading to multivariate analysis and geochemical mapping.
