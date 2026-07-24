"""Visualization tools for Central Limit Theorem Simulator."""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


class CLTVisualizer:
    """Create visualizations of Central Limit Theorem simulations."""

    def __init__(self, figsize: tuple[int, int] = (14, 10)):
        """Initialize the visualizer.

        Args:
            figsize: Figure size (width, height) in inches.
        """
        self.figsize = figsize

    def plot_distribution_comparison(
        self,
        means: NDArray[np.floating],
        distribution_name: str,
        output_path: Optional[Path] = None,
    ) -> None:
        """Create visualization comparing sample means to normal distribution.

        Args:
            means: Array of sample means from simulation.
            distribution_name: Name of the distribution used.
            output_path: Optional path to save the figure.
        """
        fig, axes = plt.subplots(1, 2, figsize=self.figsize)

        # Histogram with normal curve overlay
        ax1 = axes[0]
        ax1.hist(means, bins=30, density=True, alpha=0.7, color="skyblue", edgecolor="black")

        # Overlay normal distribution
        mu, sigma = means.mean(), means.std()
        x = np.linspace(means.min(), means.max(), 100)
        ax1.plot(
            x,
            1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2),
            "r-",
            linewidth=2,
            label="Normal Distribution",
        )

        ax1.set_xlabel("Sample Mean", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Density", fontsize=12, fontweight="bold")
        ax1.set_title(
            f"{distribution_name}\nHistogram of Sample Means", fontsize=13, fontweight="bold"
        )
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Q-Q Plot for normality assessment
        ax2 = axes[1]
        sorted_means = np.sort(means)
        theoretical_quantiles = np.sort(np.random.standard_normal(len(means)))
        ax2.scatter(theoretical_quantiles, sorted_means, alpha=0.6, s=30)

        # Add reference line
        min_val = min(theoretical_quantiles.min(), sorted_means.min())
        max_val = max(theoretical_quantiles.max(), sorted_means.max())
        ax2.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect Normal")

        ax2.set_xlabel("Theoretical Quantiles", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Sample Quantiles", fontsize=12, fontweight="bold")
        ax2.set_title(f"{distribution_name}\nQ-Q Plot", fontsize=13, fontweight="bold")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        fig.suptitle(
            f"Central Limit Theorem: {distribution_name}", fontsize=15, fontweight="bold", y=1.00
        )
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"✅ Plot saved to: {output_path}")

        plt.show()

    def plot_convergence(
        self,
        means: NDArray[np.floating],
        distribution_name: str,
        output_path: Optional[Path] = None,
    ) -> None:
        """Plot convergence of sample means to theoretical mean.

        Args:
            means: Array of sample means from simulation.
            distribution_name: Name of the distribution used.
            output_path: Optional path to save the figure.
        """
        fig, axes = plt.subplots(2, 1, figsize=self.figsize)

        # Running mean
        ax1 = axes[0]
        running_mean = np.cumsum(means) / np.arange(1, len(means) + 1)
        ax1.plot(running_mean, linewidth=2, color="steelblue", label="Running Mean")
        ax1.axhline(
            y=means.mean(),
            color="r",
            linestyle="--",
            linewidth=2,
            label=f"Final Mean: {means.mean():.4f}",
        )
        ax1.set_ylabel("Mean Value", fontsize=12, fontweight="bold")
        ax1.set_title(
            f"{distribution_name}\nConvergence of Sample Means (Law of Large Numbers)",
            fontsize=13,
            fontweight="bold",
        )
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Running standard deviation
        ax2 = axes[1]
        running_std = np.array([np.std(means[: i + 1]) for i in range(len(means))])
        ax2.plot(running_std, linewidth=2, color="coral", label="Running Std Dev")
        ax2.axhline(
            y=means.std(),
            color="r",
            linestyle="--",
            linewidth=2,
            label=f"Final Std Dev: {means.std():.4f}",
        )
        ax2.set_xlabel("Number of Samples", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Standard Deviation", fontsize=12, fontweight="bold")
        ax2.set_title(
            f"{distribution_name}\nConvergence of Sample Std Dev", fontsize=13, fontweight="bold"
        )
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        fig.suptitle(
            f"Central Limit Theorem Convergence: {distribution_name}",
            fontsize=15,
            fontweight="bold",
            y=0.995,
        )
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"✅ Plot saved to: {output_path}")

        plt.show()

    def plot_multiple_distributions(
        self,
        results: dict[str, NDArray[np.floating]],
        output_path: Optional[Path] = None,
    ) -> None:
        """Compare multiple distributions side by side.

        Args:
            results: Dictionary mapping distribution names to sample means arrays.
            output_path: Optional path to save the figure.
        """
        n_dists = len(results)
        fig, axes = plt.subplots(1, n_dists, figsize=(6 * n_dists, 5))

        if n_dists == 1:
            axes = [axes]

        for ax, (dist_name, means) in zip(axes, results.items(), strict=False):
            ax.hist(means, bins=30, density=True, alpha=0.7, color="skyblue", edgecolor="black")

            # Overlay normal distribution
            mu, sigma = means.mean(), means.std()
            x = np.linspace(means.min(), means.max(), 100)
            ax.plot(
                x,
                1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2),
                "r-",
                linewidth=2,
            )

            ax.set_xlabel("Sample Mean", fontsize=11, fontweight="bold")
            ax.set_ylabel("Density", fontsize=11, fontweight="bold")
            ax.set_title(
                f"{dist_name}\n(μ={mu:.4f}, σ={sigma:.4f})", fontsize=12, fontweight="bold"
            )
            ax.grid(True, alpha=0.3)

        fig.suptitle(
            "Central Limit Theorem: Comparison of Distributions",
            fontsize=15,
            fontweight="bold",
            y=1.00,
        )
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"✅ Plot saved to: {output_path}")

        plt.show()

    def plot_interactive_sample_size(
        self,
        simulator,
        distribution_type: str = "uniform",
        n_samples: int = 500,
        min_sample_size: int = 2,
        max_sample_size: int = 100,
        **dist_kwargs,
    ) -> None:
        """Create interactive plot with slider to morph distribution by sample size.

        Allows real-time observation of how sample size (n) affects the distribution
        shape, demonstrating the Central Limit Theorem's convergence to normality.

        Args:
            simulator: CentralLimitTheoremSimulator instance.
            distribution_type: Type of distribution ('uniform', 'exponential', 'binomial').
            n_samples: Number of samples to draw.
            min_sample_size: Minimum sample size (default: 2).
            max_sample_size: Maximum sample size (default: 100).
            **dist_kwargs: Additional arguments for the distribution function.
        """
        from matplotlib.widgets import Slider

        # Create figure with space for slider
        fig, (ax_hist, ax_qq) = plt.subplots(1, 2, figsize=(15, 6))
        plt.subplots_adjust(bottom=0.25)

        # Initial data
        initial_sample_size = min_sample_size
        if distribution_type == "uniform":
            dist_name = "Uniform Distribution"
        elif distribution_type == "exponential":
            dist_name = "Exponential Distribution"
        elif distribution_type == "binomial":
            dist_name = "Binomial Distribution"
        else:
            raise ValueError(f"Unknown distribution type: {distribution_type}")

        # Function to update plots
        def update_plots(sample_size: int) -> None:
            """Update both plots based on current sample size."""
            sample_size = int(sample_size)

            # Generate new samples
            if distribution_type == "uniform":
                new_means = simulator.simulate_uniform(n_samples=n_samples, sample_size=sample_size)
            elif distribution_type == "exponential":
                new_means = simulator.simulate_exponential(
                    n_samples=n_samples, sample_size=sample_size
                )
            elif distribution_type == "binomial":
                new_means = simulator.simulate_binomial(
                    n_samples=n_samples, sample_size=sample_size, **dist_kwargs
                )

            # Clear previous plots
            ax_hist.clear()
            ax_qq.clear()

            # Update histogram with normal curve
            ax_hist.hist(
                new_means, bins=30, density=True, alpha=0.7, color="skyblue", edgecolor="black"
            )

            mu, sigma = new_means.mean(), new_means.std()
            x = np.linspace(new_means.min(), new_means.max(), 100)
            ax_hist.plot(
                x,
                1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2),
                "r-",
                linewidth=2.5,
                label="Normal Distribution",
            )

            ax_hist.set_xlabel("Sample Mean", fontsize=11, fontweight="bold")
            ax_hist.set_ylabel("Density", fontsize=11, fontweight="bold")
            ax_hist.set_title(
                f"{dist_name}\nn={sample_size} | μ={mu:.4f} | σ={sigma:.4f}",
                fontsize=12,
                fontweight="bold",
            )
            ax_hist.legend(fontsize=10)
            ax_hist.grid(True, alpha=0.3)

            # Update Q-Q plot
            sorted_means = np.sort(new_means)
            theoretical_quantiles = np.sort(np.random.standard_normal(len(new_means)))
            ax_qq.scatter(theoretical_quantiles, sorted_means, alpha=0.6, s=25, color="steelblue")

            min_val = min(theoretical_quantiles.min(), sorted_means.min())
            max_val = max(theoretical_quantiles.max(), sorted_means.max())
            ax_qq.plot(
                [min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect Normal"
            )

            ax_qq.set_xlabel("Theoretical Quantiles", fontsize=11, fontweight="bold")
            ax_qq.set_ylabel("Sample Quantiles", fontsize=11, fontweight="bold")
            ax_qq.set_title(
                f"{dist_name}\nQ-Q Plot (n={sample_size})", fontsize=12, fontweight="bold"
            )
            ax_qq.legend(fontsize=10)
            ax_qq.grid(True, alpha=0.3)

            fig.canvas.draw_idle()

        # Create slider
        ax_slider = plt.axes((0.2, 0.1, 0.6, 0.03))
        slider = Slider(
            ax_slider,
            "Sample Size (n)",
            min_sample_size,
            max_sample_size,
            valinit=initial_sample_size,
            valstep=1,
            color="steelblue",
        )

        # Connect slider to update function
        slider.on_changed(lambda val: update_plots(int(val)))

        # Initial plot
        update_plots(initial_sample_size)

        fig.suptitle(
            "Central Limit Theorem: Interactive Sample Size Explorer",
            fontsize=14,
            fontweight="bold",
            y=0.98,
        )

        print("\n📊 Interactive Slider Controls:")
        print("  • Drag the slider to change sample size (n)")
        print("  • Watch the distribution morph from jagged (small n) to smooth (large n)")
        print("  • Q-Q plot shows convergence to normality")
        print("  • Close the window to exit\n")

        plt.show()
