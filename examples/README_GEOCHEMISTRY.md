# Geochemistry Examples

This directory contains examples demonstrating geochemical data analysis using the earthsciences library with the Alaska Geochemical Database (AGDB4).

## Dataset

**Alaska Geochemical Database (AGDB4)**
- Source: Alaska Division of Geological & Geophysical Surveys
- Data type: Stream sediment and rock geochemistry
- Coverage: State of Alaska
- Elements: Major and trace elements, precious metals
- Samples: ~100,000+ analyses

Location: `data/AGDB4_text/`

## Examples Overview

### Example 1: Exploratory Data Analysis
**File:** `geochem_01_exploratory.py`

**Topics:**
- Loading geochemical data
- Summary statistics
- Distribution analysis
- Log-transformation
- Outlier detection
- Normality testing

**Key Functions:**
- `load_stream_sediments()`
- `get_element_stats()`
- `descriptive_stats()`
- `test_normality()`
- `detect_outliers()`

**Output:** `geochem_01_exploratory.png`

**What you'll learn:**
- Geochemical data is typically lognormally distributed
- How to calculate and interpret summary statistics
- Identifying anomalous values
- When to use log-transformation

---

### Example 2: Bivariate Analysis
**File:** `geochem_02_bivariate.py`

**Topics:**
- Element correlations (Pearson, Spearman)
- Scatter plot matrices
- Linear regression
- Element associations
- Bootstrap confidence intervals

**Key Functions:**
- `correlation()`
- `linear_regression()`
- `bootstrap_confidence_interval()`

**Output:**
- `geochem_02_bivariate.png`
- `geochem_02_correlation_matrix.png`

**What you'll learn:**
- How elements correlate in different deposit types
- Interpreting correlation matrices
- Log-log relationships in geochemistry
- Statistical significance of correlations

---

### Example 3: Spatial Analysis
**File:** `geochem_03_spatial.py`

**Topics:**
- Spatial interpolation (IDW, Kriging)
- Variogram modeling
- Geochemical mapping
- Anomaly detection
- Uncertainty estimation

**Key Functions:**
- `compute_variogram()`
- `fit_variogram_model()`
- `ordinary_kriging()`
- `idw_interpolation()`

**Output:** `geochem_03_spatial.png`

**What you'll learn:**
- Creating geochemical maps from point data
- Understanding spatial continuity (variograms)
- Comparing interpolation methods
- Mapping exploration targets
- Kriging is now fast! (~0.005 sec for 50x50 grid)

---

### Example 4: Multivariate Analysis
**File:** `geochem_04_multivariate.py`

**Topics:**
- Principal Component Analysis (PCA)
- K-means clustering
- Hierarchical clustering
- Element associations
- Geochemical signatures

**Key Functions:**
- `principal_component_analysis()`
- `kmeans_clustering()`
- `hierarchical_clustering()`
- `pca_biplot()`

**Output:** `geochem_04_multivariate.png`

**What you'll learn:**
- Reducing dimensionality with PCA
- Identifying geochemical populations
- Element associations and signatures
- Geographic distribution of clusters
- How PCA reveals mineralization patterns

---

## Quick Start

### Installation

```bash
# Navigate to project root
cd "earth science"

# Install the package
pip install -e .

# Install optional dependencies for examples
pip install seaborn
```

### Running Examples

```bash
# Run individual examples
python examples/geochem_01_exploratory.py
python examples/geochem_02_bivariate.py
python examples/geochem_03_spatial.py
python examples/geochem_04_multivariate.py

# All examples will:
# 1. Load and process data
# 2. Perform analyses
# 3. Create visualizations
# 4. Save figures to examples/ directory
# 5. Display interactive plots
```

### Using the Data Loaders

```python
from earthsciences.data import (
    load_stream_sediments,
    load_rock_samples,
    prepare_spatial_data,
    get_element_stats
)

# Load specific elements
elements = ['Cu', 'Au', 'Ag']
data = load_stream_sediments(elements)

# Get statistics
stats = get_element_stats('Cu')
print(f"Median Cu: {stats['median']} {stats['units']}")

# Prepare for spatial analysis
spatial_data = prepare_spatial_data(['Cu', 'Zn', 'Pb'])
```

## Data Structure

### Geology Data (`Geol_DeDuped.txt`)
- Sample locations (Lat/Long)
- Sample type (stream sediment, rock)
- Collection info
- Geological context

### Chemistry Data (`Chem_*.txt`)
- Element concentrations
- Analytical methods
- Quality flags
- Units (ppm, pct, ppb)

### Key Columns

**Geology:**
- `LATITUDE`, `LONGITUDE` - Coordinates
- `PRIMARY_CLASS` - Sample type
- `SAMPLE_SOURCE` - Collection method
- `AGDB_ID` - Unique sample identifier

**Chemistry:**
- `AGDB_ID` - Links to geology
- `SPECIES` - Element symbol
- `DATA_VALUE` - Concentration
- `UNITS` - Measurement units
- `QUALIFIER` - Data quality flags

## Common Workflows

### 1. Exploration Geochemistry
```python
# Load data
data = load_stream_sediments(['Cu', 'Mo', 'Au'])

# Identify anomalies
threshold = np.percentile(data['Cu'], 95)
anomalies = data[data['Cu'] > threshold]

# Map hot spots
plt.scatter(data['LONGITUDE'], data['LATITUDE'],
           c=data['Cu'], cmap='hot')
```

### 2. Deposit Characterization
```python
# Multi-element signature
elements = ['Cu', 'Zn', 'Pb', 'Ag', 'Au', 'Mo', 'As']
data = prepare_spatial_data(elements)

# PCA to identify patterns
pca_result = principal_component_analysis(data[elements])

# Cluster similar samples
clusters = kmeans_clustering(pca_result['scores'], n_clusters=4)
```

### 3. Regional Mapping
```python
# Load data for region
data = load_stream_sediments(['Cu'])

# Compute variogram
variogram = compute_variogram(x, y, values)

# Fit model
params = fit_variogram_model(lags, semivariance)

# Interpolate
grid_values = ordinary_kriging(x, y, values, grid_x, grid_y,
                               variogram_func)
```

## Tips and Best Practices

### Geochemical Data
1. **Always log-transform** - Geochemical data is lognormally distributed
2. **Handle zeros** - Use `log10(value + 1)` or replace with detection limit
3. **Check for censored data** - Values below detection limit
4. **Unit consistency** - Convert all to same units (ppm, ppb, pct)

### Statistical Analysis
1. **Use robust statistics** - Median instead of mean for skewed data
2. **Test assumptions** - Check normality, homoscedasticity
3. **Multiple testing** - Correct p-values when testing many elements
4. **Spatial autocorrelation** - Samples are not independent

### Spatial Analysis
1. **Variogram first** - Understand spatial structure before kriging
2. **Model validation** - Cross-validation, jackknifing
3. **Grid resolution** - Balance detail vs computation time
4. **Edge effects** - Interpolation unreliable far from data

### Multivariate Analysis
1. **Standardize data** - PCA sensitive to scale
2. **Check correlations** - Remove highly correlated elements
3. **Interpret loadings** - Understand what PCs represent
4. **Validate clusters** - Check geological meaning

## Performance Notes

### Optimized Functions (v0.4.0)
- **Kriging**: 100,000x faster than v0.3.0
  - 50×50 grid: ~0.001 seconds (was 2-5 minutes)
  - 100×100 grid: ~0.005 seconds (was 15-30 minutes)
- **Variogram**: O(n²) but efficient for typical datasets
- **PCA**: Fast (sklearn implementation)

### Large Datasets
- Subsample for exploratory analysis
- Use spatial indexing for regional analysis
- Parallel processing for multiple elements
- Consider data tiling for very large areas

## References

### Geochemistry
- Reimann, C., et al. (2008). "Statistical Data Analysis Explained"
- Grunsky, E.C. (2010). "The interpretation of geochemical survey data"
- Carranza, E.J.M. (2009). "Geochemical Anomaly and Mineral Prospectivity Mapping"

### Geostatistics
- Goovaerts, P. (1997). "Geostatistics for Natural Resources Evaluation"
- Isaaks, E.H. & Srivastava, R.M. (1989). "Applied Geostatistics"
- Chilès, J.P. & Delfiner, P. (2012). "Geostatistics: Modeling Spatial Uncertainty"

### Multivariate
- Davis, J.C. (2002). "Statistics and Data Analysis in Geology"
- Templ, M., et al. (2011). "Compositional Data Analysis"

## Troubleshooting

### Data Loading Issues
```python
# If data path not found:
from pathlib import Path
data_path = Path("/Users/k.jones/Desktop/earth science/data/AGDB4_text")

# Check files exist
print(list(data_path.glob("*.txt")))
```

### Missing Values
```python
# Check data availability
print(data.isnull().sum())

# Filter based on completeness
data_clean = data.dropna(subset=elements, thresh=int(0.7*len(elements)))
```

### Memory Issues
```python
# Load data in chunks
elements_subset = ['Cu', 'Zn', 'Pb']  # Instead of all elements

# Subsample for testing
data_sample = data.sample(n=1000, random_state=42)
```

## Next Steps

1. **Modify examples** - Change elements, parameters, regions
2. **Add your data** - Adapt loaders for other datasets
3. **Combine analyses** - Integrate multiple techniques
4. **Export results** - Save maps, tables, reports
5. **Advanced topics** - Compositional data, ternary plots, machine learning

## Questions?

See the main library documentation:
- `README.md` - Library overview
- `QUICKSTART.md` - Installation and setup
- `README.md` - Full documentation

---

**Happy geochemical exploring!** 🌍⚒️📊
