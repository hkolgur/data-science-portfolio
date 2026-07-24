"""Interactive slider demo for Central Limit Theorem sample size exploration."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from central_limit_theorem_simulation import CentralLimitTheoremSimulator, CLTVisualizer


def main():
    """Run interactive slider demonstrations."""
    print("=" * 70)
    print("Central Limit Theorem - Interactive Sample Size Explorer")
    print("=" * 70)

    # Create simulator and visualizer
    simulator = CentralLimitTheoremSimulator(seed=42)
    visualizer = CLTVisualizer(figsize=(15, 6))

    print("\n🎯 Instructions:")
    print("  • Use the slider to adjust sample size (n) from 2 to 100")
    print("  • Observe how the distribution shape changes")
    print("  • Notice: Small n → jagged distribution, Large n → smooth normal curve")
    print("  • Q-Q plot shows convergence to normality\n")

    # Demo 1: Uniform Distribution
    print("📊 Demo 1: Uniform Distribution")
    print("-" * 70)
    print("Starting interactive slider for uniform distribution...")
    print("Slider range: n=2 to n=100\n")

    visualizer.plot_interactive_sample_size(
        simulator,
        distribution_type="uniform",
        n_samples=500,
        min_sample_size=2,
        max_sample_size=100,
    )

    # Demo 2: Exponential Distribution
    print("\n📊 Demo 2: Exponential Distribution")
    print("-" * 70)
    print("Starting interactive slider for exponential distribution...")
    print("Slider range: n=2 to n=100\n")

    visualizer.plot_interactive_sample_size(
        simulator,
        distribution_type="exponential",
        n_samples=500,
        min_sample_size=2,
        max_sample_size=100,
    )

    # Demo 3: Binomial Distribution
    print("\n📊 Demo 3: Binomial Distribution")
    print("-" * 70)
    print("Starting interactive slider for binomial distribution...")
    print("Parameters: n=10 (trials), p=0.5 (probability)")
    print("Slider range: sample_size=2 to sample_size=100\n")

    visualizer.plot_interactive_sample_size(
        simulator,
        distribution_type="binomial",
        n_samples=500,
        min_sample_size=2,
        max_sample_size=100,
        n=10,
        p=0.5,
    )

    print("\n" + "=" * 70)
    print("✅ Interactive exploration completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
