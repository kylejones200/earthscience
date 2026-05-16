# Configuration files

YAML configs drive the CLI workflow (`earthsciences run --config …`).

| Directory | Purpose |
|-----------|---------|
| `demos/` | Full pipeline demonstrations (variogram → kriging → validation) |
| `tests/` | Small fixtures for manual pipeline smoke tests |

Example configs tied to Alaska geochemistry tutorials live in [`examples/`](../examples/) (`config_alaska_*.yaml`).

```bash
# Generate a new config from a template
uv run earthsciences init --template minimal -o my_analysis.yaml

# Run an analysis
uv run earthsciences run --config configs/demos/config_complete_demo.yaml
```
