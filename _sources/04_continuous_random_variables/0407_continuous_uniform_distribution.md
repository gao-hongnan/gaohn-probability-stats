---
jupytext:
  cell_metadata_filter: -all
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
mystnb:
  number_source_lines: true
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Continuous Uniform Distribution

## Definition

```{prf:definition} Continuous Uniform Distribution (PDF)
:label: def_continuous_uniform_distribution_pdf

$X$ is a continuous random variable with a **continuous uniform distribution** if the 
probability density function is given by:

$$
\pdf(x) = \begin{cases}
    \frac{1}{b-a} & \text{if } a \leq x \leq b \\
    0 & \text{otherwise}
\end{cases}
$$ (eq:def_continuous_uniform_distribution)

where $[a,b]$ is the interval on which $X$ is defined.

Some conventions:

1. We write $X \sim \text{Uniform}(a,b)$ to indicate that $X$ has a continuous uniform distribution on $[a,b]$.
```

```{prf:definition} Continuous Uniform Distribution (CDF)
:label: def_continuous_uniform_distribution_cdf

If $X$ is a continuous random variable with a continuous uniform distribution on $[a,b]$, then the CDF is given by
integrating the PDF defined in {prf:ref}`def_continuous_uniform_distribution_pdf`:

$$
\cdf(x) = \begin{cases}
    0 & \text{if } x < a \\
    \frac{x-a}{b-a} & \text{if } a \leq x \leq b \\
    1 & \text{if } x > b
\end{cases}
$$ (eq:def_continuous_uniform_distribution_cdf)
```


```{code-cell} ipython3
:tags: [hide-input]
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt

from scipy import stats

fig, axes = plt.subplots(2, 2, figsize=(10, 10))

u = np.linspace(-1, 9, 5000) # random variable realizations
U1 = stats.uniform(0.2, 0.8) 

axes[0, 0].plot(u, U1.pdf(u), "r-", lw=3, alpha=0.6, label="Uniform[0.2, 0.8]")
axes[0, 0].set_title("PDF of Uniform[0.2, 0.8]")
axes[0, 0].set_xlim(-0.1, 1.3)
axes[0, 0].set_ylim(0, 3)
axes[0, 0].set_xlabel("x")
axes[0, 0].set_ylabel("pdf(x)")

axes[0, 1].plot(u, U1.cdf(u), "r-", lw=3, alpha=0.6, label="Uniform[0.2, 0.8]")
axes[0, 1].set_title("CDF of Uniform[0.2, 0.8]")
axes[0, 1].set_xlim(-1, 10)
axes[0, 1].set_ylim(0, 1.1)
axes[0, 1].set_xlabel("x")
axes[0, 1].set_ylabel("cdf(x)")

U2 = stats.uniform(2, 6)

axes[1, 0].plot(u, U2.pdf(u), "r-", lw=3, alpha=0.6, label="Uniform[2, 8]")
axes[1, 0].set_title("PDF of Uniform[2, 8]")
axes[1, 0].set_xlim(1, 9)
axes[1, 0].set_ylim(0, 1.1)
axes[1, 0].set_xlabel("x")
axes[1, 0].set_ylabel("pdf(x)")

axes[1, 1].plot(u, U2.cdf(u), "r-", lw=3, alpha=0.6, label="Uniform[2, 8]")
axes[1, 1].set_title("CDF of Uniform[2, 8]")
axes[1, 1].set_xlim(1, 9)
axes[1, 1].set_ylim(0, 1.1)
axes[1, 1].set_xlabel("x")
axes[1, 1].set_ylabel("cdf(x)")


plt.show()
```

## Expectation and Variance

```{prf:theorem} Expectation and Variance of Continuous Uniform Distribution
:label: thm_continuous_uniform_distribution

If $X$ is a continuous random variable with a continuous uniform distribution on $[a,b]$, then

$$
\exp \lsq X \rsq = \frac{a+b}{2} \qquad \text{and} \qquad \var \lsq X \rsq = \frac{(b-a)^2}{12}
$$ (eq:thm_continuous_uniform_distribution)
```

```{prf:remark} Intuition for Expectation and Variance of Continuous Uniform Distribution
:label: rmk_continuous_uniform_distribution

The expectation of a continuous uniform distribution is the midpoint of the interval on which the random variable is defined. 

This should not be surprising, since the probability density function is constant over the interval, and the probability of any point in the interval is the same.

Let's say we have $X \sim \text{Uniform}(0, 10)$, then $\exp \lsq X \rsq = 5$.
```