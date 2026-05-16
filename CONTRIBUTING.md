# Contributing

Thank you for helping improve **earthsciences**. This document covers setup, standards, and how to submit changes.

## Development setup

```bash
git clone https://github.com/kylejones200/earthscience.git
cd earthscience

uv sync --extra dev
uv run pytest tests/ -v    # confirm environment
```

Equivalent targets are available via `make install-dev`, `make test`, and `make lint`.

## What we need most

1. **Tests** — especially edge cases and regression tests for bug fixes
2. **Validation** — compare outputs to published results or reference implementations
3. **Documentation** — clear docstrings, examples, and README updates
4. **Performance** — profiling and optimization for spatial and large-array code

## Code standards

- **Python 3.12+** — type hints where they clarify public APIs
- **Formatting** — Black (100 columns), isort (profile black)
- **Linting** — ruff + black + isort (100-column lines)
- **Logging** — use `logging` in scripts and workflows; avoid bare `print` in library code
- **Tests** — add or update tests in `tests/` for any behavior change

```bash
pre-commit install   # once per clone
make format          # auto-format
make lint            # check before PR
make test            # full suite + coverage gate (≥35%)
make typecheck       # mypy (required in CI)
```

### Pre-commit

Hooks run **black**, **isort**, and **ruff** on commit. Install with `pre-commit install` after `uv sync --extra dev`.

### Documentation

```bash
uv sync --extra docs
uv run mkdocs serve   # http://127.0.0.1:8000
```

### PyPI releases

See [`.github/PUBLISHING.md`](.github/PUBLISHING.md) for trusted-publisher setup and release tagging.

## Pull request checklist

- [ ] Tests pass: `uv run pytest tests/ -v`
- [ ] Lint passes: `make lint`
- [ ] Docstrings updated for new or changed public functions
- [ ] `CHANGELOG.md` updated under `[Unreleased]` for user-visible changes
- [ ] No large data files or credentials committed

## Project structure

| Path | Role |
|------|------|
| `earthsciences/` | Library package |
| `tests/` | Pytest tests mirroring module layout |
| `examples/` | Runnable tutorials (not installed with the package) |
| `configs/` | Pipeline YAML for CLI workflows |

## Reporting issues

Open a [GitHub issue](https://github.com/kylejones200/earthsciences/issues) with:

- Python version and install method (`uv` / `pip`)
- Minimal code to reproduce the problem
- Expected vs actual behavior

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
