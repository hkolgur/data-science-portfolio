# Central Limit Theorem Visualization Guide

## Overview

The package now includes a comprehensive visualization module (`CLTVisualizer`) that creates publication-quality plots demonstrating the Central Limit Theorem with matplotlib.

## Features

### 1. Distribution Comparison Plots
Creates a 2-panel visualization showing:
- **Histogram with Normal Curve**: Sample means with overlaid theoretical normal distribution
- **Q-Q Plot**: Quantile-Quantile plot to assess normality of sample means

```python
from central_limit_theorem_simulation import CentralLimitTheoremSimulator, CLTVisualizer

simulator = CentralLimitTheoremSimulator(seed=42)
visualizer = CLTVisualizer()

# Generate sample means
means = simulator.simulate_uniform(n_samples=1000, sample_size=30)

# Create visualization
visualizer.plot_distribution_comparison(means, "Uniform Distribution")
```

### 2. Convergence Analysis
Shows how sample statistics converge over repeated sampling:
- **Running Mean**: How the sample mean converges to the theoretical mean
- **Running Std Dev**: How the standard deviation stabilizes

```python
visualizer.plot_convergence(means, "Uniform Distribution")
```

### 3. Multi-Distribution Comparison
Compare sample means distributions across multiple probability distributions:

```python
results = {
    "Uniform": simulator.simulate_uniform(n_samples=500, sample_size=30),
    "Exponential": simulator.simulate_exponential(n_samples=500, sample_size=30),
    "Binomial": simulator.simulate_binomial(n_samples=500, sample_size=30, n=10, p=0.5),
}

visualizer.plot_multiple_distributions(results)
```

## Running the Demos

### Interactive Demo (displays plots on screen)
```bash
uv run python examples/demo_visualization.py
```

### Save to Files Demo (saves plots as PNG)
```bash
uv run python examples/demo_with_saves.py
```

Generated plots will be saved to `visualization_outputs/` directory with these files:
- `01_uniform_distribution.png` - Uniform distribution analysis
- `02_exponential_distribution.png` - Exponential distribution analysis
- `03_binomial_distribution.png` - Binomial distribution analysis
- `04_convergence_uniform.png` - Convergence demonstration
- `05_distributions_comparison.png` - Multi-distribution comparison

## Using in Your Code

### Import the Visualizer
```python
from central_limit_theorem_simulation import CLTVisualizer
```

### Create a Visualizer Instance
```python
# Default 14x10 inch figure
visualizer = CLTVisualizer()

# Custom figure size
visualizer = CLTVisualizer(figsize=(16, 12))
```

### Save Plots to File
All visualization methods accept an optional `output_path` parameter:

```python
from pathlib import Path

output_dir = Path("my_plots")
output_dir.mkdir(exist_ok=True)

visualizer.plot_distribution_comparison(
    means,
    "My Distribution",
    output_path=output_dir / "my_plot.png"
)
```

## Plot Customization

The `CLTVisualizer` class provides:
- **High-quality output**: Saves at 300 DPI (publication-ready)
- **Customizable figure size**: Default 14x10 inches
- **Professional styling**: Grid, labels, legends included
- **Clear statistics**: Displays mean and std dev on plots

## Example Output

The visualizations demonstrate key CLT concepts:

1. **Normality of Sample Means**: Histogram shows convergence to bell curve regardless of source distribution
2. **Q-Q Plot**: Points follow the diagonal line, confirming normality
3. **Convergence**: Running mean stabilizes as sample size increases
4. **Distribution Independence**: All distributions show similar normal behavior in sample means

## Requirements

The visualization module uses:
- `matplotlib>=3.7.0` - For creating plots
- `numpy>=1.24.0` - For numerical operations

Both are already included in the project dependencies.

## Tips

- Use `seed` parameter for reproducible visualizations
- Increase `n_samples` for smoother distributions (1000+ recommended)
- Larger `sample_size` reduces variance in sample means
- Q-Q plots are more accurate with larger sample counts
- Convergence plots are more impressive with increasing sample counts
