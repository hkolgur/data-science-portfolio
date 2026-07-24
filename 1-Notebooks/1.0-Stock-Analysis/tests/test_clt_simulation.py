import numpy as np
import pytest


def test_population_bounds():
    """Test that simulated population heights are within expected bounds."""
    lower_bound, upper_bound = 180, 250
    population = np.random.uniform(lower_bound, upper_bound, 10_000)

    assert np.all(population >= lower_bound)
    assert np.all(population <= upper_bound)


def test_vectorized_sampling_shape():
    """Test that vectorized sampling produces correct output shape."""
    population_size = 100_000
    sample_size = 100
    n = 5

    # Simulate population
    population = np.random.uniform(180, 250, population_size)

    # Vectorized sampling: generate all indices at once
    random_indices = np.random.randint(0, population_size, size=(n, sample_size))
    samples = population[random_indices]

    assert samples.shape == (n, sample_size)


def test_sample_means_length():
    """Test that computing means along axis 1 produces correct length."""
    n = 5
    sample_size = 100
    population_size = 100_000

    population = np.random.uniform(180, 250, population_size)
    random_indices = np.random.randint(0, population_size, size=(n, sample_size))
    samples = population[random_indices]

    sample_means = samples.mean(axis=1)

    assert len(sample_means) == n


def test_sampling_error_decreases_with_more_trials():
    """Test that sampling error decreases as number of trials increases."""
    np.random.seed(42)

    population_size = 10_000
    sample_size = 50
    population = np.random.uniform(180, 250, population_size)
    pop_mean = population.mean()

    # Error with n=5 trials
    random_indices_5 = np.random.randint(0, population_size, size=(5, sample_size))
    sample_means_5 = population[random_indices_5].mean(axis=1)
    error_5 = abs(pop_mean - sample_means_5.mean())

    # Error with n=100 trials
    random_indices_100 = np.random.randint(0, population_size, size=(100, sample_size))
    sample_means_100 = population[random_indices_100].mean(axis=1)
    error_100 = abs(pop_mean - sample_means_100.mean())

    assert error_100 < error_5


def test_experiment_mean_close_to_population_mean():
    """Test that with large n, the mean of sample means approaches population mean."""
    np.random.seed(42)

    population_size = 10_000
    sample_size = 50
    n = 1000

    population = np.random.uniform(180, 250, population_size)
    pop_mean = population.mean()

    # Vectorized sampling
    random_indices = np.random.randint(0, population_size, size=(n, sample_size))
    sample_means = population[random_indices].mean(axis=1)
    exp_mean = sample_means.mean()

    error = abs(pop_mean - exp_mean)

    # With n=1000 and sample_size=50, error should be small
    assert error < 1.0
