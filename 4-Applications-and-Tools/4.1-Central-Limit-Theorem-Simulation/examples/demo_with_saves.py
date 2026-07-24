"""Demo script showing Central Limit Theorem visualizations with saved outputs."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from central_limit_theorem_simulation import CentralLimitTheoremSimulator, CLTVisualizer


def main():
    """Run visualization demonstrations and save outputs."""
    # Create output directory
    output_dir = Path("visualization_outputs")
    output_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("Central Limit Theorem Visualization Demo (with file outputs)")
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
    print("\n📈 Generating and saving distribution comparison plot...")
    visualizer.plot_distribution_comparison(
        uniform_means,
        "Uniform Distribution",
        output_path=output_dir / "01_uniform_distribution.png"
    )

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
    print("\n📈 Generating and saving distribution comparison plot...")
    visualizer.plot_distribution_comparison(
        exp_means,
        "Exponential Distribution",
        output_path=output_dir / "02_exponential_distribution.png"
    )

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
    print("\n📈 Generating and saving distribution comparison plot...")
    visualizer.plot_distribution_comparison(
        binom_means,
        "Binomial Distribution",
        output_path=output_dir / "03_binomial_distribution.png"
    )

    # Demo 4: Convergence for Uniform
    print("\n📊 Demo 4: Convergence Analysis (Uniform)")
    print("-" * 70)
    print("Showing how sample means converge over repeated sampling...")
    print("\n📈 Generating and saving convergence plot...")
    visualizer.plot_convergence(
        uniform_means,
        "Uniform Distribution",
        output_path=output_dir / "04_convergence_uniform.png"
    )

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
    print("\n📈 Generating and saving comparison plot...")
    visualizer.plot_multiple_distributions(
        results,
        output_path=output_dir / "05_distributions_comparison.png"
    )

    print("\n" + "=" * 70)
    print("✅ All visualizations completed and saved!")
    print(f"📁 Output directory: {output_dir.absolute()}")
    print("=" * 70)

    # List generated files
    print("\n📋 Generated files:")
    for i, file in enumerate(sorted(output_dir.glob("*.png")), 1):
        print(f"   {i}. {file.name}")


if __name__ == "__main__":
    main()
