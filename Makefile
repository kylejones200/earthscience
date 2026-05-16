.PHONY: install install-dev test lint format run-examples clean

install:
	uv sync

install-dev:
	uv sync --extra dev

test:
	uv run pytest tests/ -v

test-cov:
	uv run pytest tests/ --cov=earthsciences --cov-report=term-missing --cov-fail-under=35

lint:
	uv run black --check earthsciences tests
	uv run isort --check-only earthsciences tests
	uv run ruff check earthsciences tests

pre-commit:
	uv run pre-commit run --all-files

docs:
	uv run mkdocs build --strict

docs-serve:
	uv run mkdocs serve

format:
	uv run black earthsciences tests examples
	uv run isort earthsciences tests

typecheck:
	uv run mypy earthsciences --ignore-missing-imports

ci: lint typecheck test

clean:
	rm -rf build dist .pytest_cache .coverage htmlcov *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
