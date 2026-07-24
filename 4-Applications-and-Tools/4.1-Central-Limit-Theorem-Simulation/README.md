# Central Limit Theorem Simulation

A production-ready Python package for simulating and visualizing the Central Limit Theorem with various probability distributions.

## Features

- 🎲 Simulate the Central Limit Theorem with multiple distributions (Uniform, Exponential, Binomial)
- 📊 Comprehensive statistical analysis of simulation results
- 📈 Publication-quality visualizations with matplotlib:
  - Histogram + Q-Q plots for normality assessment
  - Convergence analysis (running mean & std dev)
  - Multi-distribution comparison plots
- 🎯 Reproducible simulations with seed support
- ✅ Full test coverage (100% on core modules)
- 📚 Type-safe with Python type hints
- 🚀 Production-ready code with pre-commit hooks and CI/CD ready

## Installation

### Using UV (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/central-limit-theorem-simulation.git
cd central-limit-theorem-simulation

# Install with UV
uv sync
```

### Using pip

```bash
pip install central-limit-theorem-simulation
```

## Quick Start ##

### As a Library (With Visualization)

```python
from central_limit_theorem_simulation import CentralLimitTheoremSimulator, CLTVisualizer

# Create simulator
simulator = CentralLimitTheoremSimulator(seed=42)

# Simulate uniform distribution
means = simulator.simulate_uniform(n_samples=1000, sample_size=30)

# Get statistics
print(f"Mean: {means.mean():.4f}")
print(f"Std Dev: {means.std():.4f}")

# Visualize the results
visualizer = CLTVisualizer()
visualizer.plot_distribution_comparison(means, "Uniform Distribution")
visualizer.plot_convergence(means, "Uniform Distribution")
```

### Via Command Line

```bash
# Simulate uniform distribution
clt-sim --seed 42 --n-samples 1000 uniform

# Simulate exponential distribution
clt-sim exponential --sample-size 50

# Simulate binomial distribution
clt-sim binomial --n-samples 500
```

### Running Visualization Demos

```bash
# Interactive demo - displays plots on screen
uv run python examples/demo_visualization.py

# Save plots to files (PNG format)
uv run python examples/demo_with_saves.py

# Plots will be saved to visualization_outputs/
```

### Running the Streamlit Web Application

For an interactive, modern web interface, use Streamlit:

```bash
# Install web dependencies
uv sync --group web

# Run the Streamlit app
streamlit run app.py
```

This launches an interactive web application where you can:
- **Adjust parameters dynamically** using sliders and dropdowns
- **View real-time statistics** as you change simulation parameters
- **Explore multiple visualizations** with tabs (Distribution Comparison, Convergence Analysis, Multiple Comparisons)
- **Learn about CLT** with built-in educational content
- **Export insights** directly from the web interface

The app runs on `http://localhost:8501` by default.

## Visualization

The package includes a powerful `CLTVisualizer` class for creating publication-quality plots:

### Distribution Comparison (Histogram + Q-Q Plot)
```python
from central_limit_theorem_simulation import CentralLimitTheoremSimulator, CLTVisualizer

simulator = CentralLimitTheoremSimulator(seed=42)
visualizer = CLTVisualizer()

# Simulate and visualize
means = simulator.simulate_exponential(n_samples=1000, sample_size=30)
visualizer.plot_distribution_comparison(means, "Exponential Distribution")
```

### Convergence Analysis
Track how sample means converge to the theoretical mean:
```python
visualizer.plot_convergence(means, "Exponential Distribution")
```

### Multi-Distribution Comparison
Compare sample means across different probability distributions:
```python
results = {
    "Uniform": simulator.simulate_uniform(n_samples=500, sample_size=30),
    "Exponential": simulator.simulate_exponential(n_samples=500, sample_size=30),
    "Binomial": simulator.simulate_binomial(n_samples=500, sample_size=30, n=10, p=0.5),
}
visualizer.plot_multiple_distributions(results)
```

### Save Plots to Files
All visualization methods accept an `output_path` parameter:
```python
from pathlib import Path

visualizer.plot_distribution_comparison(
    means,
    "My Distribution",
    output_path=Path("output/my_plot.png")
)
```

For detailed visualization guide and more examples, see [VISUALIZATION.md](VISUALIZATION.md).

## Development

### Setup Development Environment

```bash
# Install dependencies (including dev dependencies)
uv sync --group dev

# Run tests
make test

# Run tests with coverage
make test-cov

# Run linting and formatting
make format
make lint

# Type check
make type-check

# Run visualization demo
uv run python examples/demo_with_saves.py

# Setup pre-commit hooks
pre-commit install
pre-commit run --all-files
```

### Project Structure

```
central-limit-theorem-simulation/
├── src/
│   └── central_limit_theorem_simulation/
│       ├── __init__.py          # Package exports
│       ├── simulator.py         # Main simulator implementation
│       ├── cli.py              # Command-line interface
│       └── visualization.py     # Visualization module
├── tests/
│   ├── __init__.py
│   └── test_simulator.py       # Unit tests (100% coverage)
├── examples/
│   ├── demo_visualization.py    # Interactive visualization demo
│   └── demo_with_saves.py      # Demo that saves plots to files
├── docs/                        # Documentation directory
├── .github/
│   └── workflows/              # CI/CD workflows
├── pyproject.toml              # Project configuration
├── .gitignore                  # Git ignore rules
├── README.md                   # This file
├── LICENSE                     # MIT License
├── CONTRIBUTING.md             # Contributing guidelines
└── VISUALIZATION.md            # Detailed visualization guide
```

## Dependencies

### Core Dependencies
- `numpy>=1.24.0` - Numerical computing
- `matplotlib>=3.7.0` - Visualization
- `scipy>=1.10.0` - Scientific computing

### Development Dependencies
- `pytest>=7.0` - Testing framework
- `pytest-cov>=4.0` - Code coverage
- `black>=23.0` - Code formatter
- `ruff>=0.1.0` - Linter
- `mypy>=1.0` - Static type checker
- `pre-commit>=3.0` - Git hooks framework

### Optional Dependencies
- `streamlit>=1.28.0` - Interactive web interface (use `uv sync --group web`)

## Configuration

### Python Versions

This package supports Python 3.10+ and is tested on:
- Python 3.10
- Python 3.11
- Python 3.12

### Code Quality Tools

- **Black**: Line length of 100 characters
- **Ruff**: Configured for multiple rule sets (E, W, F, I, C, B)
- **MyPy**: Type checking enabled
- **Pytest**: 100% test coverage target

## Testing

```bash
# Run all tests
make test
# or
pytest

# Run specific test file
pytest tests/test_simulator.py

# Run with verbose output
pytest -v

# Run with coverage report
make test-cov
# or
pytest --cov=src/central_limit_theorem_simulation --cov-report=html

# Run visualization demos to verify plots work
uv run python examples/demo_visualization.py          # Interactive plots
uv run python examples/demo_with_saves.py            # Save plots to files
```

### Current Test Coverage
- **Simulator module**: 100% ✓
- **Package initialization**: 100% ✓
- **Overall**: 53% (core logic fully covered)

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this simulator in your research, please cite:

```bibtex
@software{clt_simulation_2024,
  title = {Central Limit Theorem Simulation},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/central-limit-theorem-simulation}
}
```

## Support

For support, please open an issue on [GitHub Issues](https://github.com/yourusername/central-limit-theorem-simulation/issues).

## Acknowledgments

Built with modern Python tooling:
- [UV](https://github.com/astral-sh/uv) - Fast Python package manager
- [Pytest](https://pytest.org/) - Testing framework
- [Black](https://github.com/psf/black) - Code formatter
- [Ruff](https://github.com/astral-sh/ruff) - Python linter
- [Matplotlib](https://matplotlib.org/) - Visualization library
- [NumPy](https://numpy.org/) - Numerical computing
- [SciPy](https://scipy.org/) - Scientific computing

## Sample Outputs

The package includes example visualization outputs in the `visualization_outputs/` directory:
- Histogram plots showing normality of sample means
- Q-Q plots for normality assessment
- Convergence analysis showing Law of Large Numbers
- Multi-distribution comparisons

Run `uv run python examples/demo_with_saves.py` to generate these visualizations.
