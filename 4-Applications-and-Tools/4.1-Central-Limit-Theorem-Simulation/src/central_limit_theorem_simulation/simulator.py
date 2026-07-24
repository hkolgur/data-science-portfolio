"""Central Limit Theorem Simulator Implementation."""

from typing import Optional

import numpy as np
from numpy.typing import NDArray


class CentralLimitTheoremSimulator:
    """Simulate the Central Limit Theorem with various distributions."""

    def __init__(self, seed: Optional[int] = None):
        """Initialize the simulator.

        Args:
            seed: Random seed for reproducibility.
        """
        self.rng = np.random.default_rng(seed)

    def simulate_uniform(
        self,
        n_samples: int = 1000,
        sample_size: int = 30,
        low: float = 0.0,
        high: float = 1.0,
    ) -> NDArray[np.floating]:
        """Simulate CLT with uniform distribution.

        Args:
            n_samples: Number of samples to draw.
            sample_size: Size of each sample.
            low: Lower bound of uniform distribution.
            high: Upper bound of uniform distribution.

        Returns:
            Array of sample means.
        """
        means = []
        for _ in range(n_samples):
            sample = self.rng.uniform(low, high, sample_size)
            means.append(np.mean(sample))
        return np.array(means)

    def simulate_exponential(
        self,
        n_samples: int = 1000,
        sample_size: int = 30,
        scale: float = 1.0,
    ) -> NDArray[np.floating]:
        """Simulate CLT with exponential distribution.

        Args:
            n_samples: Number of samples to draw.
            sample_size: Size of each sample.
            scale: Scale parameter of exponential distribution.

        Returns:
            Array of sample means.
        """
        means = []
        for _ in range(n_samples):
            sample = self.rng.exponential(scale, sample_size)
            means.append(np.mean(sample))
        return np.array(means)

    def simulate_binomial(
        self,
        n_samples: int = 1000,
        sample_size: int = 30,
        n: int = 10,
        p: float = 0.5,
    ) -> NDArray[np.floating]:
        """Simulate CLT with binomial distribution.

        Args:
            n_samples: Number of samples to draw.
            sample_size: Size of each sample.
            n: Number of trials in binomial distribution.
            p: Probability of success in binomial distribution.

        Returns:
            Array of sample means.
        """
        means = []
        for _ in range(n_samples):
            sample = self.rng.binomial(n, p, sample_size)
            means.append(np.mean(sample))
        return np.array(means)
