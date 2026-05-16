# Publishing

## GitHub Pages (documentation)

Pages is configured to deploy from the **Docs** workflow (`.github/workflows/docs.yml`) on every push to `main`.

- Site URL: https://kylejones200.github.io/earthscience/
- Manual run: Actions → Docs → Run workflow

Local preview:

```bash
uv sync --extra docs
uv run mkdocs serve
```

## PyPI (package `earthsciences`)

### One-time: Trusted publishing

1. Create the project on [pypi.org](https://pypi.org/) if it does not exist (`earthsciences`).
2. Open **Publishing** → **Add a new pending publisher** → **GitHub**.
3. Use exactly:

   | Field | Value |
   |-------|--------|
   | PyPI project name | `earthsciences` |
   | Owner | `kylejones200` |
   | Repository name | `earthscience` |
   | Workflow name | `publish.yml` |
   | Environment name | *(leave empty)* |

4. Save the pending publisher.

### Release

1. Bump `version` in `pyproject.toml`.
2. Update `CHANGELOG.md`.
3. Tag and push:

```bash
git tag v0.3.0
git push origin main
git push origin v0.3.0
```

The **Publish to PyPI** workflow builds with `uv build` and uploads via OIDC (no API token in the repo).

### Verify

```bash
pip install earthsciences==0.3.0
python -c "import earthsciences; print(earthsciences.__version__)"
```
