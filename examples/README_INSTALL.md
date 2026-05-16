# Running the Examples

## Installation

Before running examples, install the package in development mode:

```bash
# From the root directory (earth science/)
pip install -e .

# Or install with notebook support
pip install -e ".[notebooks]"
```

## Running Python Scripts

```bash
cd examples/
python example_statistics.py
```

## Running Jupyter Notebooks

```bash
# Install notebook dependencies if needed
pip install -e ".[notebooks]"

# Start Jupyter
cd examples/
jupyter notebook
```

Then open `01_statistics_basics.ipynb` or other notebooks.

## Troubleshooting

### ModuleNotFoundError: No module named 'earthsciences'

Make sure you've installed the package:
```bash
cd "/Users/k.jones/Desktop/earth science"
pip install -e .
```

### Import errors for optional dependencies

Some examples may require additional packages:
```bash
pip install seaborn Pillow astropy spectrum
```

For geospatial examples (if they exist):
```bash
uv sync --extra geospatial
```

## Creating Your Own Scripts

When creating your own analysis scripts, make sure to:

1. Install the package: `uv sync` (from the repo root)
2. Import normally: `import earthsciences as es`
3. No need for `sys.path` hacks!

Example:
```python
import numpy as np
import earthsciences as es

# Your analysis here
data = np.random.randn(100)
stats = es.statistics.descriptive_stats(data)
print(stats)
```
