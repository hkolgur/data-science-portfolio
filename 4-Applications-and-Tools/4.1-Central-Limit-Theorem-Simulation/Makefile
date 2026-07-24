.PHONY: help install install-dev test test-cov lint format type-check pre-commit clean

help:
	@echo "Central Limit Theorem Simulation - Development Commands"
	@echo ""
	@echo "Installation:"
	@echo "  make install          Install package with dependencies"
	@echo "  make install-dev      Install with development dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  make test             Run tests"
	@echo "  make test-cov         Run tests with coverage report"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint             Run linter (ruff)"
	@echo "  make format           Format code with black"
	@echo "  make type-check       Run type checker (mypy)"
	@echo "  make pre-commit       Run pre-commit hooks on all files"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean            Remove build artifacts and cache files"

install:
	uv sync

install-dev:
	uv sync --group dev

test:
	uv run pytest

test-cov:
	uv run pytest --cov=src/central_limit_theorem_simulation --cov-report=html --cov-report=term-missing
	@echo "Coverage report generated in htmlcov/index.html"

lint:
	uv run ruff check src/ tests/

format:
	uv run black src/ tests/
	uv run ruff check src/ tests/ --fix

type-check:
	uv run mypy src/

pre-commit:
	uv run pre-commit run --all-files

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf htmlcov
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info

.DEFAULT_GOAL := help
