# Quick start

## Install

```bash
uv sync
uv sync --extra dev   # optional: pytest, black, jupyter
```

## Library usage

```python
import logging
import numpy as np
import earthsciences as es
from earthsciences.utils.logging_config import setup_logging

setup_logging()

# Statistics
data = np.random.default_rng(0).normal(size=100)
stats = es.statistics.descriptive_stats(data)
logging.info("mean=%.3f std=%.3f", stats["mean"], stats["std"])

# Time series
t, signal = es.timeseries.generate_test_signal("mixed", duration=1, sampling_rate=1000)
freqs, power = es.timeseries.power_spectrum(signal, sampling_rate=1000)

# Spatial: variogram + kriging
x = np.random.default_rng(1).uniform(0, 10, 30)
y = np.random.default_rng(2).uniform(0, 10, 30)
z = np.sin(x) + np.cos(y)
lags, gamma, n_pairs = es.spatial.compute_variogram(x, y, z)
fit = es.spatial.fit_variogram_model(lags, gamma, n_pairs=n_pairs)
grid_x, grid_y = np.meshgrid(np.linspace(0, 10, 30), np.linspace(0, 10, 30))
z_grid = es.spatial.ordinary_kriging(x, y, z, grid_x, grid_y, fit["variogram_func"])
```

## Run examples

```bash
cd examples
uv run python geochem_00_quickstart.py
uv run python geochronology_metamorphic_terrane.py

cd tutorials
uv run python 05_geochronology.py
```

See [`examples/README.md`](examples/README.md) for the full catalog.

## CLI pipeline

```bash
uv run earthsciences init --template minimal -o my_project.yaml
# edit my_project.yaml, then:
uv run earthsciences run --config my_project.yaml
```

## Next steps

- [`README.md`](README.md) — overview and feature table
- [`examples/README_GEOCHEMISTRY.md`](examples/README_GEOCHEMISTRY.md) — Alaska geochemistry walkthrough
- [`examples/README_GEOCHRONOLOGY.md`](examples/README_GEOCHRONOLOGY.md) — radiometric dating case study
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — development workflow
