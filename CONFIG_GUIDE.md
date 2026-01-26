# Config-Driven Architecture - User Guide

## Overview

The earthsciences package now supports **config-driven workflows** where you can run complete geostatistical analyses from a simple YAML configuration file—no Python coding required!

## Quick Start

### 1. Install Dependencies

```bash
pip install pydantic pyyaml click
```

Or reinstall the package:

```bash
cd "/Users/k.jones/Desktop/earth science"
pip install -e .
```

### 2. Generate a Template Config File

```bash
earthsciences init --output my_analysis.yaml --template full
```

Available templates:
- **full**: Complete config with all options
- **minimal**: Required fields only
- **kriging**: Focused on kriging parameters  
- **validation**: Focused on cross-validation

### 3. Edit the Config File

Open `my_analysis.yaml` and edit:
- Data file path and column names
- Grid bounds for interpolation
- Analysis parameters

```yaml
project:
  name: "My Analysis"
  output_dir: "./results"

data:
  input_file: "data/my_data.csv"
  x_column: "x"
  y_column: "y"
  z_column: "value"

kriging:
  grid:
    x_min: 0.0
    x_max: 100.0
    y_min: 0.0
    y_max: 100.0
    resolution: 1.0
```

### 4. Run the Analysis

```bash
earthsciences run --config my_analysis.yaml
```

### 5. View Results

Results are saved to the output directory:
- `predictions.csv` - Kriging predictions
- `variance.csv` - Kriging variance
- `variogram.png` - Variogram plot
- `kriging_map.png` - Interpolation map
- `report.txt` - Summary report

## CLI Commands

### `earthsciences run`
Run complete geostatistical analysis:
```bash
earthsciences run --config my_analysis.yaml
earthsciences run --config my_analysis.yaml --output ./custom_output
earthsciences run --config my_analysis.yaml --quiet
```

### `earthsciences init`
Generate template configuration:
```bash
earthsciences init
earthsciences init --output my_config.yaml
earthsciences init --template minimal
```

### `earthsciences validate`
Validate configuration file:
```bash
earthsciences validate my_analysis.yaml
```

### `earthsciences show`
Display configuration:
```bash
earthsciences show my_analysis.yaml
```

### `earthsciences summarize`
Generate configuration summary:
```bash
earthsciences summarize my_analysis.yaml
earthsciences summarize my_analysis.yaml --output summary.txt
```

### `earthsciences templates`
List available templates:
```bash
earthsciences templates
```

## Configuration Reference

### Project Settings

```yaml
project:
  name: "Analysis Name"
  output_dir: "./results"
  description: "Optional description"
```

### Data Input

```yaml
data:
  input_file: "path/to/data.csv"
  x_column: "longitude"
  y_column: "latitude"  
  z_column: "value"
```

Supported formats: CSV, Excel

### Preprocessing

```yaml
preprocessing:
  remove_outliers: true
  outlier_method: "iqr"  # iqr, zscore, mad
  outlier_threshold: 3.0
  transform: "log"  # log, boxcox, normal_score, null
  handle_negatives: true
```

### Variogram Modeling

```yaml
variogram:
  n_lags: 15
  max_lag: null  # Auto-calculate
  lag_tolerance: 0.5
  models:
    - spherical
    - exponential
    - gaussian
  auto_fit: true
  fit_method: "wls"  # ols, wls, ml
```

### Kriging Interpolation

```yaml
kriging:
  method: "ordinary"  # ordinary, simple, universal
  
  neighborhood:
    max_neighbors: 25
    min_neighbors: 3
    search_radius: null
  
  grid:
    x_min: 0.0
    x_max: 100.0
    y_min: 0.0
    y_max: 100.0
    resolution: 1.0
  
  compute_variance: true
```

### Cross-Validation

```yaml
validation:
  cross_validation: true
  n_folds: 5
  method: "kfold"  # kfold, loo, spatial
  metrics:
    - rmse
    - mae
    - r2
  stratified: false
```

### Visualization

```yaml
visualization:
  style: "minimalist"  # minimalist, default, seaborn
  plots:
    - variogram
    - kriging_map
    - variance_map
    - cross_validation
    - histogram
  colormap: "viridis"
  dpi: 300
```

### Output Options

```yaml
output:
  save_predictions: true
  save_variance: true
  save_plots: true
  save_report: true
  formats:
    - png
    - csv
  compression: false
```

## Python API

You can also use the config system from Python:

```python
from earthsciences.workflows import ConfigRunner

# Load and run analysis
runner = ConfigRunner('my_analysis.yaml')
results = runner.run()

# Access results
print(f"RMSE: {results['validation']['rmse']:.4f}")
print(f"R²: {results['validation']['r2']:.4f}")
print(f"Output: {results['output_dir']}")
```

## Example Workflows

### Minimal Quick Analysis

```yaml
project:
  name: "Quick Test"
  output_dir: "./quick_results"

data:
  input_file: "data.csv"
  x_column: "x"
  y_column: "y"
  z_column: "value"

kriging:
  grid:
    x_min: 0.0
    x_max: 100.0
    y_min: 0.0
    y_max: 100.0
    resolution: 2.0
```

### Full Production Analysis

See `examples/example_config.yaml` for a complete production example with all options configured.

## Benefits

✅ **No coding required** - Edit YAML files  
✅ **Reproducible** - Config files document everything  
✅ **Version control** - Track configs in git  
✅ **Easy sharing** - Send config to collaborators  
✅ **Batch processing** - Run multiple configs  
✅ **Validated** - Pydantic ensures correctness  

## Troubleshooting

### Config Validation Errors

Use `validate` command to check your config:
```bash
earthsciences validate my_config.yaml
```

### File Not Found

Ensure file paths in config are correct (absolute or relative to where you run the command).

### Import Errors

Reinstall with config dependencies:
```bash
pip install -e ".[full]"
```

## Next Steps

1. Try the example config: `examples/example_config.yaml`
2. Create your own configs for your data
3. Set up batch processing for multiple analyses
4. Integrate into your data processing pipelines

---

**For more information, see the main README and examples directory.**
