# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- GitHub issue templates and Dependabot config
- Docs badge and corrected repository URLs in README
- MkDocs site (`docs/`, GitHub Pages workflow)
- PyPI publish workflow on version tags (`v*`)
- Pre-commit hooks (black, isort, ruff)
- Coverage gate in CI (35% minimum on tested modules)
- `examples/tutorials/` for domain tutorial scripts
- `earthsciences.utils.logging_config` for consistent logging
- `examples/geochronology_metamorphic_terrane.py` geochronology case study
- `uv.lock` and uv-based CI workflow
- `configs/` directory for pipeline YAML organization

### Changed
- `earthsciences.spatial` explicit `__all__` and deprecation aliases (`nearest_neighbor_interpolation`, `lisa`, `spatial_autocorrelation`)
- Examples use `logging` instead of `print`
- README and contributor docs aligned with uv workflow

### Fixed
- Skipped variogram and natural-neighbor tests now run
- `fit_variogram_model` returns `variogram_func` key
- `rbf_interpolation`, `cross_validate`, and `gev_return_level` edge cases
- `earthsciences.data` package tracked in git (was excluded by `.gitignore`)
- `return_level` validates `return_period > 1` (consistent with `gev_return_level`)
- Timeseries `autocorrelation` name clash resolved (`spectral_autocorrelation` alias)

## [0.3.0] - 2024

Initial public alpha: statistics, spatial, time series, multivariate, directional, geochronology, config-driven geostatistics pipeline, and Alaska geochemistry examples.
