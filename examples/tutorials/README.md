# Tutorials

Domain walkthrough scripts with publication-style figures. Run from **this directory**:

```bash
cd examples/tutorials
uv run python 05_geochronology.py
```

| Script | Topic |
|--------|--------|
| `example_statistics.py` | Core statistics API |
| `05_geochronology.py` | Radiometric dating |
| `06_directional_statistics.py` | Circular / spherical statistics |
| `07_timeseries_analysis.py` | Spectral analysis, wavelets |
| `08_image_analysis.py` | Remote sensing, grain analysis |
| `demo_kriging_performance.py` | Kriging performance |

Figures are written to this folder (`*.png`). Shared plotting helpers live in [`../plot_utils.py`](../plot_utils.py).
