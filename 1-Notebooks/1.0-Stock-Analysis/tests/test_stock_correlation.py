import math
import numpy as np
import pytest
from scipy.stats import pearsonr


# Stock data from the notebook
stock_A = np.array([2.5, 3.0, -2.0, 4.0, 3.5, -1.5, 2.0, -0.5, 1.5, 2.5, -1.0, 3.0])
stock_B = np.array([1.5, 2.0, -1.0, 3.0, 2.5, -2.0, 1.0, -0.5, 1.0, 2.0, -0.5, 2.5])


def test_manual_covariance_loop():
    """Test loop-based covariance calculation against numpy ground truth."""
    mean_A = stock_A.mean()
    mean_B = stock_B.mean()
    n = len(stock_A)

    # Loop-based calculation
    covar_loop = 0
    for i in range(n):
        covar_loop += (stock_A[i] - mean_A) * (stock_B[i] - mean_B)
    covar_loop /= (n - 1)

    # NumPy ground truth
    covar_numpy = np.cov(stock_A, stock_B)[0, 1]

    assert np.isclose(covar_loop, covar_numpy, rtol=1e-10)


def test_vectorized_covariance():
    """Test vectorized covariance calculation against numpy ground truth."""
    mean_A = stock_A.mean()
    mean_B = stock_B.mean()
    n = len(stock_A)

    # Vectorized calculation
    covar_vec = np.sum((stock_A - mean_A) * (stock_B - mean_B)) / (n - 1)

    # NumPy ground truth
    covar_numpy = np.cov(stock_A, stock_B)[0, 1]

    assert np.isclose(covar_vec, covar_numpy, rtol=1e-10)


def test_both_covariance_approaches_agree():
    """Test that loop and vectorized covariance methods give the same result."""
    mean_A = stock_A.mean()
    mean_B = stock_B.mean()
    n = len(stock_A)

    # Loop-based
    covar_loop = 0
    for i in range(n):
        covar_loop += (stock_A[i] - mean_A) * (stock_B[i] - mean_B)
    covar_loop /= (n - 1)

    # Vectorized
    covar_vec = np.sum((stock_A - mean_A) * (stock_B - mean_B)) / (n - 1)

    assert np.isclose(covar_loop, covar_vec, rtol=1e-10)


def test_manual_std():
    """Test manual standard deviation calculation against numpy."""
    mean_A = stock_A.mean()
    n = len(stock_A)

    # Manual calculation
    std_A_manual = math.sqrt(np.sum((stock_A - mean_A) ** 2) / (n - 1))

    # NumPy ground truth (ddof=1 for sample std dev)
    std_A_numpy = np.std(stock_A, ddof=1)

    assert np.isclose(std_A_manual, std_A_numpy, rtol=1e-10)


def test_manual_pearson_correlation():
    """Test manual Pearson correlation against scipy."""
    mean_A = stock_A.mean()
    mean_B = stock_B.mean()
    n = len(stock_A)

    # Compute covariance
    covar = np.sum((stock_A - mean_A) * (stock_B - mean_B)) / (n - 1)

    # Compute standard deviations
    std_A = math.sqrt(np.sum((stock_A - mean_A) ** 2) / (n - 1))
    std_B = math.sqrt(np.sum((stock_B - mean_B) ** 2) / (n - 1))

    # Manual correlation
    corr_manual = covar / (std_A * std_B)

    # SciPy ground truth
    corr_scipy, _ = pearsonr(stock_A, stock_B)

    assert np.isclose(corr_manual, corr_scipy, rtol=1e-10)


def test_pearson_range():
    """Test that Pearson correlation is always in [-1, 1]."""
    mean_A = stock_A.mean()
    mean_B = stock_B.mean()
    n = len(stock_A)

    covar = np.sum((stock_A - mean_A) * (stock_B - mean_B)) / (n - 1)
    std_A = math.sqrt(np.sum((stock_A - mean_A) ** 2) / (n - 1))
    std_B = math.sqrt(np.sum((stock_B - mean_B) ** 2) / (n - 1))

    corr = covar / (std_A * std_B)

    assert -1.0 <= corr <= 1.0


def test_perfect_positive_correlation():
    """Test perfect positive correlation."""
    A = np.array([1.0, 2.0, 3.0])
    B = np.array([2.0, 4.0, 6.0])

    corr, _ = pearsonr(A, B)

    assert np.isclose(corr, 1.0, rtol=1e-10)


def test_perfect_negative_correlation():
    """Test perfect negative correlation."""
    A = np.array([1.0, 2.0, 3.0])
    B = np.array([-1.0, -2.0, -3.0])

    corr, _ = pearsonr(A, B)

    assert np.isclose(corr, -1.0, rtol=1e-10)
