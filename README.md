# earthsciences

A Python toolkit for geoscience data analysis: statistics, time series, geostatistics, geochemistry, geochronology, and related domains. Built on NumPy and SciPy with reproducible workflows and tested implementations.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/kylejones200/earthsciences/actions/workflows/ci.yml/badge.svg)](https://github.com/kylejones200/earthsciences/actions/workflows/ci.yml)

> **Status:** Active development (v0.3). APIs may change between minor releases. Validate critical results against independent methods before publication or operational use.

## Features

| Domain | Capabilities |
|--------|----------------|
| **Statistics** | Descriptive stats, regression, hypothesis tests, extreme values, resampling |
| **Time series** | Spectral analysis, filtering, wavelets, changepoint detection |
| **Spatial** | Variograms, ordinary/universal kriging, IDW/RBF interpolation, spatial autocorrelation |
| **Multivariate** | PCA, clustering, classification, compositional data |
| **Directional** | Circular and spherical statistics, paleomagnetics |
| **Geochronology** | Decay systems, isochrons, U–Pb, radiocarbon |
| **Geophysics & more** | Gravity, magnetics, seismic, petroleum, hydrogeology, imaging |

Config-driven geostatistical pipelines are available via the CLI (`earthsciences run`).

## Installation

Requires **Python 3.12+**. [uv](https://docs.astral.sh/uv/) is recommended for reproducible installs.

```bash
git clone https://github.com/kylejones200/earthsciences.git
cd earthsciences

uv sync                  # runtime dependencies
uv sync --extra dev      # tests, linting, notebooks
uv sync --extra full     # optional: seaborn, astropy, spectrum
```

With pip:

```bash
pip install -e ".[dev]"
```

## Quick example

```python
import numpy as np
import earthsciences as es

# Descriptive statistics
data = np.random.randn(200)
stats = es.statistics.descriptive_stats(data)

# Variogram + kriging
rng = np.random.default_rng(42)
x, y = rng.uniform(0, 10, 40), rng.uniform(0, 10, 40)
z = np.sin(x / 2) + np.cos(y / 2) + rng.normal(0, 0.1, 40)

lags, gamma, n_pairs = es.spatial.compute_variogram(x, y, z)
fit = es.spatial.fit_variogram_model(lags, gamma, n_pairs=n_pairs)

grid_x, grid_y = np.meshgrid(np.linspace(0, 10, 40), np.linspace(0, 10, 40))
surface = es.spatial.ordinary_kriging(
    x, y, z, grid_x, grid_y, fit["variogram_func"]
)
```

## CLI workflow

```bash
# Create a config from a template
uv run earthsciences init --template minimal -o analysis.yaml

# Run variogram → kriging → validation pipeline
uv run earthsciences run --config configs/demos/config_complete_demo.yaml
```

See [`configs/README.md`](configs/README.md) and [`examples/README_CONFIG_EXAMPLES.md`](examples/README_CONFIG_EXAMPLES.md).

## Documentation

- **Online:** [https://kylejones200.github.io/earthscience/](https://kylejones200.github.io/earthscience/) (MkDocs on GitHub Pages)
- **Local:** `uv sync --extra docs && uv run mkdocs serve`

## Examples

| Path | Content |
|------|---------|
| [`examples/tutorials/`](examples/tutorials/) | Domain tutorials (`05_geochronology.py`, …) |
| [`examples/geochem_*.py`](examples/) | Alaska geochemistry workflow |
| [`examples/geochronology_metamorphic_terrane.py`](examples/geochronology_metamorphic_terrane.py) | Geochronology case study |

```bash
cd examples/tutorials && uv run python 05_geochronology.py
cd examples && uv run python geochem_00_quickstart.py
```

See [`examples/README.md`](examples/README.md) and [`QUICKSTART.md`](QUICKSTART.md).

## Development

```bash
uv sync --extra dev
pre-commit install          # black, isort, ruff on every commit
make test                   # pytest + coverage gate (≥30%)
make lint
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Publishing

Tag a release to build and publish to PyPI (requires [trusted publishing](https://docs.pypi.org/trusted-publishers/) configured for this repo):

```bash
git tag v0.3.0
git push origin v0.3.0
```

## Project layout

```
earthsciences/     # Library source
tests/             # Pytest suite
examples/          # Scripts, notebooks, tutorial configs
configs/           # Pipeline YAML (demos & smoke tests)
```

Dependencies are locked in `uv.lock`; runtime requirements are declared in `pyproject.toml`.

## License

MIT — see [LICENSE](LICENSE).
