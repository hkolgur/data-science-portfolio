# Stock Analysis & Height Analysis

A Jupyter notebook exploring two statistical concepts: correlation analysis of stock returns and the Central Limit Theorem via population height simulation.

## Contents

### Stock Returns Analysis
- Computes correlation between two stocks' monthly returns
- Demonstrates both manual covariance calculation and vectorized NumPy operations
- Uses Pearson correlation coefficient to quantify the relationship

### Central Limit Theorem (Height Analysis)
- Simulates a population of 1,000,000 people with uniformly distributed heights (180–250 cm)
- Repeatedly samples from the population to examine how sample means behave
- Visualizes the distribution of sample means with increasing sample sizes
- Demonstrates how the distribution of sample means approaches a normal distribution (CLT)

## Getting Started

### Prerequisites
- Python 3.8+
- Jupyter Notebook or JupyterLab

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/hkolgur/colab-vscode.git
   cd colab-vscode
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Open the notebook:
   ```bash
   jupyter notebook stock-analysis-height-analysis.ipynb
   ```

## Testing

The repository includes a pytest test suite that validates the statistical computations:

```bash
python3 -m pytest tests/ -v
```

**Test Coverage:**
- **`test_stock_correlation.py`** (8 tests): Validates covariance, standard deviation, and Pearson correlation calculations against NumPy and SciPy ground truth
- **`test_clt_simulation.py`** (5 tests): Validates Central Limit Theorem simulation logic, sampling behavior, and error reduction with larger sample sizes

All tests use seeded random numbers or hardcoded data for reproducibility.

## Development Notes

- This notebook was developed in VS Code using the Colab extension
- Originally executed on Google Colab with a Tesla T4 GPU
- Can be run locally without GPU acceleration

## License

Open for educational use.
