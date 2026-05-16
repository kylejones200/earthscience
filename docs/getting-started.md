# Getting started

## Requirements

- Python 3.12+
- NumPy, SciPy, pandas, matplotlib (installed automatically)

## Development install

```bash
git clone https://github.com/kylejones200/earthsciences.git
cd earthsciences
uv sync --extra dev
uv run pytest tests/ -v
```

## CLI workflows

Config-driven geostatistics:

```bash
uv run earthsciences init --template minimal -o analysis.yaml
uv run earthsciences run --config analysis.yaml
```

See [Examples](guide/examples.md) for tutorial scripts and case studies.
