# Earth Sciences Recipes for Python

**ALPHA VERSION 0.3.0 - NOT PRODUCTION READY**

A modern Python library for earth sciences data analysis, providing algorithms for common analysis tasks.

**Status:** This is an early alpha version under active development. Not all functions have been validated, some features are incomplete, and performance is not optimized. Use with caution and validate results.

## Overview

This library provides algorithms for common data analysis tasks in earth sciences, including:

- **Univariate & Bivariate Statistics** - Descriptive statistics, distributions, hypothesis testing, regression
- **Time-Series Analysis** - Spectral analysis, filtering, wavelets, Lomb-Scargle periodograms
- **Signal Processing** - FFT, filtering, convolution, system analysis
- **Spatial Data Analysis** - Kriging, variogram modeling, interpolation, point distributions
- **Image Processing** - Image manipulation, enhancement, satellite image analysis
- **Multivariate Analysis** - PCA, clustering, ICA, dimensionality reduction
- **Directional Data** - Circular and spherical statistics
- **Geochronology** - Radioactive decay, radiometric dating methods
- **Geophysics** - Gravity, magnetic, and seismic data processing
- **Petroleum** - Petrophysics and production decline analysis
- **Hydrogeology** - Aquifer testing and well hydraulics
- **Seismology** - Earthquake analysis and waveform processing

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
import earthsciences as es

# Univariate statistics
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
stats = es.statistics.descriptive_stats(data)
print(stats)

# Time-series analysis
t, signal = es.timeseries.generate_test_signal()
spectrum = es.timeseries.power_spectrum(signal)

# Spatial analysis
variogram = es.spatial.compute_variogram(x, y, values)
interpolated = es.spatial.kriging(x, y, values, grid_x, grid_y)

# Geochronology
from earthsciences.geochronology import radioactive_decay, DECAY_CONSTANTS
age = radioactive_decay(N0=100, t=5730, half_life=5730)
```

## Structure

```
earthsciences/
├── __init__.py
├── statistics/         # Statistical analysis
├── timeseries/         # Time-series and signal processing
├── spatial/            # Spatial data analysis
├── multivariate/       # Multivariate analysis
├── directional/        # Circular and spherical statistics
├── geochronology/      # Radiometric dating
├── geophysics/         # Gravity, magnetic, seismic
├── petroleum/          # Petrophysics and production
├── hydrogeology/       # Groundwater analysis
├── seismology/         # Earthquake analysis
└── utils/              # Common utilities and helpers
```

## Examples

See the `examples/` directory for Python scripts and Jupyter notebooks demonstrating each module.

## Requirements

- Python 3.12+
- NumPy
- SciPy
- Matplotlib
- Pandas
- Scikit-learn
- Scikit-image
- PyWavelets

See `requirements.txt` for full list.

## Known Limitations

- Not all functions validated - some implementations need verification
- Some functions not fully implemented (will raise NotImplementedError)
- Performance not optimized - kriging and spatial operations may be slow
- Educational project - validate results before use in research

## Testing

```bash
# Run test suite
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=earthsciences --cov-report=html
```

Current test coverage: 232 tests passing across core modules.

## License

MIT License - See LICENSE file for details.

## Contributing

This is an alpha-stage educational project. Contributions welcome:

- Validate functions and add tests
- Add missing implementations
- Improve performance (especially spatial methods)
- Fix bugs and improve documentation

Please include tests with any contributions.
