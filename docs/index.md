# earthsciences

**earthsciences** is a Python library for geoscience data analysis: statistics, geostatistics, time series, geochemistry, geochronology, and related fields.

## Install

```bash
pip install earthsciences
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add earthsciences
```

## Quick example

```python
import numpy as np
import earthsciences as es

x = np.random.default_rng(0).uniform(0, 10, 40)
y = np.random.default_rng(1).uniform(0, 10, 40)
z = np.sin(x / 2) + np.cos(y / 2)

lags, gamma, n_pairs = es.spatial.compute_variogram(x, y, z)
fit = es.spatial.fit_variogram_model(lags, gamma, n_pairs=n_pairs)

grid_x, grid_y = np.meshgrid(np.linspace(0, 10, 30), np.linspace(0, 10, 30))
surface = es.spatial.ordinary_kriging(x, y, z, grid_x, grid_y, fit["variogram_func"])
```

## Next steps

- [Getting started](getting-started.md)
- [User guide](guide/overview.md)
- [API reference](api/spatial.md)
