# Config-Driven Alaska Geochemistry Examples

This directory contains real-world examples using Alaska geochemistry data from the USGS AGDB4 database.

## Quick Start

### 1. Prepare the Data

First, process the raw Alaska data:

```bash
cd examples
python prepare_alaska_data.py
```

This creates processed CSV files in `data/processed/` for 8 elements:
- Gold (Au)
- Copper (Cu)
- Lead (Pb)
- Zinc (Zn)
- Silver (Ag)
- Molybdenum (Mo)
- Arsenic (As)
- Antimony (Sb)

### 2. Run an Example Analysis

Choose an example config and run it:

```bash
# Quick gold analysis
python -m earthsciences.cli run --config examples/config_alaska_gold_quick.yaml

# Comprehensive copper analysis
python -m earthsciences.cli run --config examples/config_alaska_copper_production.yaml

# Validation-focused zinc study
python -m earthsciences.cli run --config examples/config_alaska_zinc_validation.yaml

# High-resolution silver regional study
python -m earthsciences.cli run --config examples/config_alaska_silver_regional.yaml
```

### 3. View Results

Results are saved to the output directory specified in each config:
- `results/alaska_gold_quick/`
- `results/alaska_copper_production/`
- `results/alaska_zinc_validation/`
- `results/alaska_silver_southeast/`

## Example Configs

### 1. `config_alaska_gold_quick.yaml` - Quick Analysis

**Purpose**: Rapid reconnaissance mapping of gold anomalies

**Features**:
- Minimal configuration
- Fast processing (~5 min for 300K+ samples)
- Standard kriging parameters
- 5-fold cross-validation
- 0.5° resolution (~50 km grid)

**Use Case**: Initial exploration, quick look at regional patterns

**Command**:
```bash
python -m earthsciences.cli run --config examples/config_alaska_gold_quick.yaml
```

---

### 2. `config_alaska_copper_production.yaml` - Production Analysis

**Purpose**: Comprehensive production-ready copper analysis

**Features**:
- Full feature set enabled
- 10-fold cross-validation
- Higher resolution (0.25° ~ 25 km)
- Multiple evaluation metrics
- Detailed plots including scatter plots
- NPY format output for further analysis

**Use Case**: Detailed study, publication-quality maps, resource estimation

**Command**:
```bash
python -m earthsciences.cli run --config examples/config_alaska_copper_production.yaml
```

**Expected Runtime**: ~10-15 minutes (depends on grid size)

---

### 3. `config_alaska_zinc_validation.yaml` - Validation Study

**Purpose**: Assess kriging performance for zinc

**Features**:
- Validation-focused
- 10-fold stratified cross-validation
- Multiple metrics (RMSE, MAE, R², MAPE)
- Q-Q plots for residual analysis
- No prediction grid (validation only)

**Use Case**: Method comparison, parameter tuning, quality assessment

**Command**:
```bash
python -m earthsciences.cli run --config examples/config_alaska_zinc_validation.yaml
```

**Expected Runtime**: ~5 minutes (no grid interpolation)

---

### 4. `config_alaska_silver_regional.yaml` - Regional Study

**Purpose**: High-resolution silver mapping for Southeast Alaska

**Features**:
- Regional focus (Southeast Alaska panhandle)
- High resolution (0.1° ~ 10 km)
- Defined search radius (100 km)
- Custom colormap for silver
- Detailed variance mapping

**Use Case**: Detailed regional exploration, prospect targeting

**Command**:
```bash
python -m earthsciences.cli run --config examples/config_alaska_silver_regional.yaml
```

**Expected Runtime**: ~8-12 minutes (high resolution)

## Data Overview

### Alaska Geochemistry Database (AGDB4)

**Source**: USGS Alaska Geochemical Database version 4.0

**Samples**: 375,265 samples with coordinates

**Elements Processed**:
| Element | Samples | Median (ppm) | Range (ppm) |
|---------|---------|--------------|-------------|
| Au (Gold) | 327,753 | 5.0 | 0.00 - 100,000 |
| Cu (Copper) | 345,472 | 30.0 | 0.01 - 466,000 |
| Pb (Lead) | 341,433 | 13.5 | 0.00 - 747,400 |
| Zn (Zinc) | 348,138 | 128.0 | 0.02 - 842,000 |
| Ag (Silver) | 337,277 | 0.75 | 0.00 - 600,000 |
| Mo (Molybdenum) | 279,152 | 5.0 | 0.01 - 46,000 |
| As (Arsenic) | 314,293 | 110.0 | 0.10 - 320,000 |
| Sb (Antimony) | 329,013 | 51.0 | 0.02 - 664,100 |

**Coordinate System**: Geographic (WGS84), Decimal Degrees

**Coverage**: Entire state of Alaska
- Longitude: -180° to -130°
- Latitude: 51° to 72°

## Customizing Configs

### Change the Element

Edit the `data` section:
```yaml
data:
  input_file: "data/processed/alaska_pb.csv"  # Change to any element
  x_column: "LONGITUDE"
  y_column: "LATITUDE"
  z_column: "Pb_PPM"  # Match the element name
```

### Adjust Grid Resolution

Higher resolution = more detail but slower:
```yaml
kriging:
  grid:
    resolution: 0.1  # High res (~10 km)
    # resolution: 0.5  # Medium res (~50 km)
    # resolution: 1.0  # Low res (~100 km)
```

### Focus on a Specific Region

Change grid bounds to zoom in:
```yaml
kriging:
  grid:
    # Example: Juneau region
    x_min: -136.0
    x_max: -133.0
    y_min: 57.5
    y_max: 59.0
    resolution: 0.05  # Very high res for small area
```

### Try Different Preprocessing

```yaml
preprocessing:
  transform: "log"       # Options: log, boxcox, normal_score
  outlier_method: "iqr"  # Options: iqr, zscore, mad
  outlier_threshold: 3.0 # Adjust sensitivity
```

## Output Files

Each analysis produces:

### Plots (PNG)
- `variogram.png` - Experimental variogram with fitted model
- `kriging_map.png` - Interpolated concentration map
- `variance_map.png` - Kriging uncertainty map
- `cross_validation.png` - Predicted vs actual plot
- `histogram.png` - Data distribution
- `scatter_actual_predicted.png` - Detailed validation plot (if enabled)

### Data Files (CSV)
- `predictions.csv` - Grid of predicted values and variance
- `variance.csv` - Standalone variance grid (if enabled)

### Report
- `report.txt` - Complete analysis summary with statistics

## Tips for Large Datasets

### Speed Optimization

1. **Coarser grid**: Use `resolution: 1.0` or higher
2. **Smaller region**: Reduce x_min/x_max and y_min/y_max bounds
3. **Fewer neighbors**: Reduce `max_neighbors` to 15-20
4. **Skip validation**: Set `cross_validation: false`

### Memory Management

For very large grids (>1 million points):
- Process in smaller regions
- Increase `resolution` value
- Use `compression: true` in output settings

## Batch Processing

Run multiple analyses:

```bash
# Create a batch script
for config in config_alaska_*.yaml; do
    echo "Processing $config..."
    python -m earthsciences.cli run --config "examples/$config"
done
```

## Validation Commands

Before running, validate your config:

```bash
# Check if config is valid
python -m earthsciences.cli validate examples/config_alaska_gold_quick.yaml

# View config details
python -m earthsciences.cli show examples/config_alaska_gold_quick.yaml

# Generate summary
python -m earthsciences.cli summarize examples/config_alaska_gold_quick.yaml
```

## Troubleshooting

### "File not found" Error
- Ensure you've run `prepare_alaska_data.py` first
- Check that paths in config match your directory structure

### Long Runtime
- Reduce grid resolution
- Reduce grid extent
- Decrease `max_neighbors`

### Memory Error
- Process smaller regions
- Increase `resolution` (coarser grid)
- Close other applications

### Poor Validation Metrics
- Try different variogram models
- Adjust preprocessing (transform type)
- Check for spatial clustering in data

## Next Steps

1. **Experiment**: Modify configs and rerun analyses
2. **Compare**: Run same element with different parameters
3. **Combine**: Create multi-element analyses
4. **Extend**: Add your own data following the CSV format

## References

- USGS Alaska Geochemical Database: https://www.usgs.gov/centers/alaska-science-center
- Geostatistics theory: See `MATHEMATICAL_REVIEW.md`
- Config format: See `CONFIG_GUIDE.md`

---

**Questions or issues?** Check the main `CONFIG_GUIDE.md` or `KNOWN_ISSUES.md`
