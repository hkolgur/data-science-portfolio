"""Demo script showing Central Limit Theorem visualizations."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from central_limit_theorem_simulation import CentralLimitTheoremSimulator, CLTVisualizer


def main():
    """Run visualization demonstrations."""
    print("=" * 70)
    print("Central Limit Theorem Visualization Demo")
    print("=" * 70)

    # Create simulator and visualizer
    simulator = CentralLimitTheoremSimulator(seed=42)
    visualizer = CLTVisualizer(figsize=(14, 10))

    # Demo 1: Uniform Distribution
    print("\n📊 Demo 1: Uniform Distribution")
    print("-" * 70)
    print("Simulating 1000 samples of size 30 from Uniform(0, 1)...")
    uniform_means = simulator.simulate_uniform(n_samples=1000, sample_size=30)
    print(f"✅ Results:")
    print(f"   Mean: {uniform_means.mean():.6f}")
    print(f"   Std Dev: {uniform_means.std():.6f}")
    print(f"   Min: {uniform_means.min():.6f}")
    print(f"   Max: {uniform_means.max():.6f}")
    print("\n📈 Generating distribution comparison plot...")
    visualizer.plot_distribution_comparison(uniform_means, "Uniform Distribution")

    # Demo 2: Exponential Distribution
    print("\n📊 Demo 2: Exponential Distribution")
    print("-" * 70)
    print("Simulating 1000 samples of size 30 from Exponential(1)...")
    exp_means = simulator.simulate_exponential(n_samples=1000, sample_size=30)
    print(f"✅ Results:")
    print(f"   Mean: {exp_means.mean():.6f}")
    print(f"   Std Dev: {exp_means.std():.6f}")
    print(f"   Min: {exp_means.min():.6f}")
    print(f"   Max: {exp_means.max():.6f}")
    print("\n📈 Generating distribution comparison plot...")
    visualizer.plot_distribution_comparison(exp_means, "Exponential Distribution")

    # Demo 3: Binomial Distribution
    print("\n📊 Demo 3: Binomial Distribution")
    print("-" * 70)
    print("Simulating 1000 samples of size 30 from Binomial(n=10, p=0.5)...")
    binom_means = simulator.simulate_binomial(n_samples=1000, sample_size=30, n=10, p=0.5)
    print(f"✅ Results:")
    print(f"   Mean: {binom_means.mean():.6f}")
    print(f"   Std Dev: {binom_means.std():.6f}")
    print(f"   Min: {binom_means.min():.6f}")
    print(f"   Max: {binom_means.max():.6f}")
    print("\n📈 Generating distribution comparison plot...")
    visualizer.plot_distribution_comparison(binom_means, "Binomial Distribution")

    # Demo 4: Convergence for Uniform
    print("\n📊 Demo 4: Convergence Analysis (Uniform)")
    print("-" * 70)
    print("Showing how sample means converge over repeated sampling...")
    print("\n📈 Generating convergence plot...")
    visualizer.plot_convergence(uniform_means, "Uniform Distribution")

    # Demo 5: Multiple Distributions Comparison
    print("\n📊 Demo 5: Multiple Distributions Comparison")
    print("-" * 70)
    print("Creating new samples for clean comparison...")
    sim_compare = CentralLimitTheoremSimulator(seed=123)
    results = {
        "Uniform": sim_compare.simulate_uniform(n_samples=500, sample_size=30),
        "Exponential": sim_compare.simulate_exponential(n_samples=500, sample_size=30),
        "Binomial": sim_compare.simulate_binomial(n_samples=500, sample_size=30, n=10, p=0.5),
    }
    print("\n📈 Generating comparison plot...")
    visualizer.plot_multiple_distributions(results)

    print("\n" + "=" * 70)
    print("✅ All visualizations completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
