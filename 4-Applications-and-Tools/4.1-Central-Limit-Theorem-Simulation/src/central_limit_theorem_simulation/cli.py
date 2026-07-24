"""Command Line Interface for Central Limit Theorem Simulator."""

import argparse

from .simulator import CentralLimitTheoremSimulator


def main() -> None:
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Central Limit Theorem Simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "distribution",
        choices=["uniform", "exponential", "binomial"],
        help="Distribution to simulate",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1000,
        help="Number of samples (default: 1000)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=30,
        help="Size of each sample (default: 30)",
    )

    args = parser.parse_args()

    simulator = CentralLimitTheoremSimulator(seed=args.seed)

    if args.distribution == "uniform":
        results = simulator.simulate_uniform(
            n_samples=args.n_samples,
            sample_size=args.sample_size,
        )
        print("Uniform Distribution Results:")
    elif args.distribution == "exponential":
        results = simulator.simulate_exponential(
            n_samples=args.n_samples,
            sample_size=args.sample_size,
        )
        print("Exponential Distribution Results:")
    elif args.distribution == "binomial":
        results = simulator.simulate_binomial(
            n_samples=args.n_samples,
            sample_size=args.sample_size,
        )
        print("Binomial Distribution Results:")

    print(f"Mean: {results.mean():.4f}")
    print(f"Std Dev: {results.std():.4f}")
    print(f"Min: {results.min():.4f}")
    print(f"Max: {results.max():.4f}")


if __name__ == "__main__":
    main()
