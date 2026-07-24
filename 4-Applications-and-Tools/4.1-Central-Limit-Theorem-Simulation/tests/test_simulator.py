"""Tests for the Central Limit Theorem Simulator."""

import numpy as np

from central_limit_theorem_simulation import CentralLimitTheoremSimulator


class TestCentralLimitTheoremSimulator:
    """Test the CentralLimitTheoremSimulator class."""

    def test_simulator_initialization(self) -> None:
        """Test simulator can be initialized."""
        simulator = CentralLimitTheoremSimulator()
        assert simulator is not None

    def test_simulator_with_seed(self) -> None:
        """Test simulator with seed for reproducibility."""
        sim1 = CentralLimitTheoremSimulator(seed=42)
        sim2 = CentralLimitTheoremSimulator(seed=42)

        results1 = sim1.simulate_uniform(n_samples=10, sample_size=5)
        results2 = sim2.simulate_uniform(n_samples=10, sample_size=5)

        np.testing.assert_array_equal(results1, results2)

    def test_simulate_uniform(self) -> None:
        """Test uniform distribution simulation."""
        simulator = CentralLimitTheoremSimulator(seed=42)
        results = simulator.simulate_uniform(n_samples=100, sample_size=30)

        assert len(results) == 100
        assert 0 <= results.min() < results.max() <= 1

    def test_simulate_exponential(self) -> None:
        """Test exponential distribution simulation."""
        simulator = CentralLimitTheoremSimulator(seed=42)
        results = simulator.simulate_exponential(n_samples=100, sample_size=30)

        assert len(results) == 100
        assert results.min() >= 0

    def test_simulate_binomial(self) -> None:
        """Test binomial distribution simulation."""
        simulator = CentralLimitTheoremSimulator(seed=42)
        results = simulator.simulate_binomial(n_samples=100, sample_size=30, n=10, p=0.5)

        assert len(results) == 100
        assert 0 <= results.min() and results.max() <= 10
