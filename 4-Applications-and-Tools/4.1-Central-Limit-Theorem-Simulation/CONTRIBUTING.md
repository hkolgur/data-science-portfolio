# Contributing to Central Limit Theorem Simulation

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing.

## Code of Conduct

Be respectful and constructive in all interactions.

## Getting Started

1. Fork the repository
2. Clone your fork locally
3. Create a virtual environment:
   ```bash
   uv sync --all-extras
   ```
4. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Workflow

### Code Style

We use automated tools to maintain code quality:

```bash
# Format code with Black
black src/ tests/

# Lint with Ruff
ruff check src/ tests/ --fix

# Type check with mypy
mypy src/
```

### Pre-commit Hooks

Install pre-commit hooks to run checks automatically:

```bash
pre-commit install
pre-commit run --all-files  # Run on all files
```

### Testing

Write tests for all new features:

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=src/central_limit_theorem_simulation

# Run specific test
pytest tests/test_simulator.py::TestCentralLimitTheoremSimulator::test_simulator_initialization
```

Aim for high test coverage. All tests must pass before submitting a PR.

## Submitting Changes

1. Ensure all tests pass:
   ```bash
   pytest
   black --check src/ tests/
   ruff check src/ tests/
   mypy src/
   ```

2. Commit with clear messages:
   ```bash
   git commit -m "Add feature: description of what was added"
   ```

3. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

4. Open a Pull Request with:
   - Clear description of changes
   - Reference to any related issues
   - Evidence that tests pass

## Commit Message Guidelines

- Use imperative mood ("Add feature" not "Added feature")
- Keep first line under 50 characters
- Reference issues and PRs when relevant
- Example:
  ```
  Add exponential distribution support
  
  - Implements simulate_exponential method
  - Adds tests for exponential distribution
  - Fixes #42
  ```

## Pull Request Process

1. Update README.md and docs with any new features
2. Update version in `pyproject.toml` if needed
3. Ensure CI passes (tests, linting, type checking)
4. Request review from maintainers
5. Address review comments
6. Maintainers will merge when approved

## Reporting Issues

When reporting bugs, include:

- Python version
- Operating system
- Steps to reproduce
- Expected behavior
- Actual behavior
- Error messages/tracebacks

## Questions?

Feel free to open an issue or discussion for questions.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
