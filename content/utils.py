import random
from typing import Callable, Optional, Tuple, List, Union
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from matplotlib_inline import backend_inline

# plt.rcParams["figure.dpi"] = 600
# plt.rcParams["savefig.dpi"] = 600


# pylint: disable=too-many-arguments,too-many-locals
def plot_continuous_pdf_and_cdf(
    stats_dist: Callable,
    low: float,
    high: float,
    title="Uniform Distribution",
    lw: int = 3,
    alpha: float = 0.7,
    color: str = "darkred",
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    figsize: Tuple[float, float] = (10, 5),
) -> None:
    """Plots the PDF and CDF of a continuous distribution."""
    x = np.linspace(low, high, 5000)
    X = stats_dist  # symbolic X to represent the distribution

    pdf = X.pdf(x)
    cdf = X.cdf(x)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    if xlim is not None:
        axes[0].set_xlim(*xlim)

    if ylim is not None:
        axes[0].set_ylim(*ylim)

    axes[0].set_xlabel("x", fontsize=16)
    axes[0].set_ylabel("pdf(x)", fontsize=16)

    axes[0].plot(x, pdf, color=color, lw=lw, alpha=alpha)

    axes[0].set_title(f"PDF of {title}", fontsize=16)

    if xlim is not None:
        axes[1].set_xlim(*xlim)

    axes[1].set_ylim(
        0, 1.1
    )  # this can be hardcoded since the CDF is always between 0 and 1

    axes[1].set_xlabel("x", fontsize=16)
    axes[1].set_ylabel("cdf(x)", fontsize=16)

    axes[1].plot(x, cdf, color=color, lw=lw, alpha=alpha)

    axes[1].set_title(f"CDF of {title}", fontsize=16)

    plt.show()


def seed_all(seed: int = 1992) -> None:
    """Seed all random number generators."""
    print(f"Using Seed Number {seed}")

    # os.environ["PYTHONHASHSEED"] = str(seed)  # set PYTHONHASHSEED env var
    np.random.seed(seed)  # numpy pseudo-random generator
    random.seed(seed)  # built-in pseudo-random generator


def true_pmf(x: float, population: np.ndarray) -> float:
    """PMF of the true population: map X(\S) to a probability.

    Note:
        The PMF is completely determined if we know the true distribution.
    """
    return np.sum(population == x) / len(population)


def empirical_pmf(x: float, sample: np.ndarray):
    """Empirical distribution of the sample."""
    return np.sum(sample == x) / len(sample)


def use_svg_display():
    """Use the svg format to display a plot in Jupyter.
    Defined in :numref:`sec_calculus`"""
    backend_inline.set_matplotlib_formats("svg")


if __name__ == "__main__":
    seed_all()

    # # Standard Normal Distribution
    # mean, sigma = 0, 1
    # X = stats.norm(mean, sigma)

    # plot_continuous_pdf_and_cdf(
    #     X, -5, 5, title="Normal Distribution $\mu=0, \sigma=1$", figsize=(15, 5)
    # )

    # # Sum of Discrete Uniform Random Variables
    # plot_sum_of_uniform_distribution()
