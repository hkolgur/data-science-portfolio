"""Streamlit web application for Central Limit Theorem Visualization."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from central_limit_theorem_simulation import CentralLimitTheoremSimulator, CLTVisualizer


def plot_distribution_comparison_streamlit(
    means: np.ndarray,
    distribution_name: str,
) -> None:
    """Create visualization comparing sample means to normal distribution.

    Args:
        means: Array of sample means from simulation.
        distribution_name: Name of the distribution used.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

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
    ax1.set_title(f"{distribution_name}\nHistogram of Sample Means", fontsize=13, fontweight="bold")
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
    st.pyplot(fig)
    plt.close(fig)


def plot_convergence_streamlit(
    means: np.ndarray,
    distribution_name: str,
) -> None:
    """Plot convergence of sample means to theoretical mean.

    Args:
        means: Array of sample means from simulation.
        distribution_name: Name of the distribution used.
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

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
    st.pyplot(fig)
    plt.close(fig)


def plot_multiple_distributions_streamlit(results: dict[str, np.ndarray]) -> None:
    """Compare multiple distributions side by side.

    Args:
        results: Dictionary mapping distribution names to sample means arrays.
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
    st.pyplot(fig)
    plt.close(fig)


@st.cache_resource
def get_simulator(seed: int) -> CentralLimitTheoremSimulator:
    """Get or create a simulator instance with caching."""
    return CentralLimitTheoremSimulator(seed=seed)


@st.cache_data
def run_simulation(
    distribution_type: str,
    n_samples: int,
    sample_size: int,
    seed: int = 42,
    **dist_kwargs,
) -> np.ndarray:
    """Run a simulation with caching."""
    simulator = get_simulator(seed)

    if distribution_type == "uniform":
        return simulator.simulate_uniform(n_samples=n_samples, sample_size=sample_size)
    elif distribution_type == "exponential":
        return simulator.simulate_exponential(n_samples=n_samples, sample_size=sample_size)
    elif distribution_type == "binomial":
        return simulator.simulate_binomial(
            n_samples=n_samples, sample_size=sample_size, **dist_kwargs
        )
    else:
        raise ValueError(f"Unknown distribution type: {distribution_type}")


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Central Limit Theorem Simulator",
        page_icon="📊",
        layout="wide",
    )

    st.title("📊 Central Limit Theorem Simulator")
    st.markdown(
        """
    Explore the Central Limit Theorem interactively by simulating sample means
    from different distributions and observing how they converge to a normal distribution.
    """
    )

    st.sidebar.header("⚙️ Simulation Settings")
    with st.sidebar:
        # Simulation parameters
        distribution = st.selectbox(
            "Select Distribution",
            ["uniform", "exponential", "binomial"],
            format_func=lambda x: x.title(),
        )

        col1, col2 = st.columns(2)
        with col1:
            n_samples = st.slider(
                "Number of Samples",
                min_value=50,
                max_value=2000,
                value=500,
                step=50,
                help="How many times to draw a sample and calculate its mean",
            )

        with col2:
            sample_size = st.slider(
                "Sample Size (n)",
                min_value=2,
                max_value=100,
                value=30,
                step=1,
                help="Size of each individual sample",
            )

        seed = st.number_input("Random Seed", value=42, step=1)

        # Distribution-specific parameters
        dist_kwargs = {}
        if distribution == "uniform":
            with st.expander("Uniform Distribution Parameters"):
                col1, col2 = st.columns(2)
                with col1:
                    low = st.number_input("Low", value=0.0, step=0.1)
                with col2:
                    high = st.number_input("High", value=1.0, step=0.1)
                dist_kwargs = {"low": low, "high": high}

        elif distribution == "exponential":
            with st.expander("Exponential Distribution Parameters"):
                scale = st.number_input("Scale", value=1.0, step=0.1)
                dist_kwargs = {"scale": scale}

        elif distribution == "binomial":
            with st.expander("Binomial Distribution Parameters"):
                col1, col2 = st.columns(2)
                with col1:
                    n = st.number_input("Number of Trials (n)", value=10, step=1, min_value=1)
                with col2:
                    p = st.slider("Probability (p)", min_value=0.0, max_value=1.0, value=0.5)
                dist_kwargs = {"n": n, "p": p}

    # Run simulation
    with st.spinner("Running simulation..."):
        means = run_simulation(
            distribution_type=distribution,
            n_samples=n_samples,
            sample_size=sample_size,
            seed=seed,
            **dist_kwargs,
        )

    # Display statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Mean", f"{means.mean():.4f}")
    with col2:
        st.metric("Std Dev", f"{means.std():.4f}")
    with col3:
        st.metric("Min", f"{means.min():.4f}")
    with col4:
        st.metric("Max", f"{means.max():.4f}")

    # Tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(
        ["📈 Distribution Comparison", "📊 Convergence Analysis", "🔄 Multiple Comparisons"]
    )

    with tab1:
        st.subheader("Distribution Comparison")
        st.write(
            """
        The left plot shows a histogram of sample means with a normal distribution overlay.
        The right plot (Q-Q plot) shows how closely the distribution matches a perfect normal distribution.
        """
        )
        plot_distribution_comparison_streamlit(means, distribution.title())

    with tab2:
        st.subheader("Convergence Analysis")
        st.write(
            """
        Watch how the sample mean (top) and standard deviation (bottom) converge
        as we collect more samples. This demonstrates the Law of Large Numbers.
        """
        )
        plot_convergence_streamlit(means, distribution.title())

    with tab3:
        st.subheader("Compare Multiple Distributions")
        st.write("See how different distributions converge to normality with the same parameters.")

        if st.button("Generate Comparison", key="compare_button"):
            with st.spinner("Generating comparison plots..."):
                sim_compare = CentralLimitTheoremSimulator(seed=123)
                comparison_results = {
                    "Uniform": sim_compare.simulate_uniform(
                        n_samples=n_samples, sample_size=sample_size
                    ),
                    "Exponential": sim_compare.simulate_exponential(
                        n_samples=n_samples, sample_size=sample_size
                    ),
                    "Binomial": sim_compare.simulate_binomial(
                        n_samples=n_samples, sample_size=sample_size, n=10, p=0.5
                    ),
                }
                plot_multiple_distributions_streamlit(comparison_results)

    # Footer with explanation
    st.divider()
    with st.expander("📚 Learn More About the Central Limit Theorem"):
        st.markdown(
            """
        ### What is the Central Limit Theorem?

        The **Central Limit Theorem (CLT)** states that:
        - When you take many independent samples from *any* distribution and calculate their means,
        - The distribution of these means will be approximately **normal** (bell-shaped),
        - Regardless of the shape of the original distribution!

        ### Key Observations:

        1. **Distribution Shape**: The original distribution can be uniform, exponential, or anything else,
           but the sample means always form a normal distribution.

        2. **Law of Large Numbers**: As you increase the sample size (n), the distribution becomes more
           tightly concentrated around the true population mean.

        3. **Standard Error**: The standard deviation of sample means equals σ/√n, where σ is the
           population standard deviation.

        ### Try This:

        - Increase **Sample Size (n)** to see the distribution become more normal (watch the Q-Q plot)
        - Increase **Number of Samples** to get a better estimate of the distribution shape
        - Try different distributions to see how they all converge to normality
        """
        )


if __name__ == "__main__":
    main()
