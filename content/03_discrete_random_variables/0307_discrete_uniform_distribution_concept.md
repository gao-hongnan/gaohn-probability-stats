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

```{code-cell} ipython3
:tags: [remove-input]
import sys
from pathlib import Path
parent_dir = str(Path().resolve().parent)
sys.path.append(parent_dir)

import matplotlib.pyplot as plt
%matplotlib inline
import matplotlib_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('svg')
```

# Discrete Uniform Distribution

## PMF and CDF of Discrete Uniform Distribution

```{prf:definition} Discrete Uniform Distribution (PMF)
:label: def_discrete_uniform_pmf

Let $X$ be a **discrete random variable** that follows a Uniform distribution over the set $\S$.
This means that $X=x$ has an equally likely chance of being drawn.

Then the probability mass function (PMF) of $X$ is given by 

$$
\P(X=x) = \dfrac{1}{\lvert \S \rvert}
$$

More specifically, if $X$ is a discrete random variable that follows a Uniform distribution 
over the ordered set $\S$ where the lower bound is $a$ and the upper bound is $b$, 
then the PMF of $X$ is given by

$$
\P(X=x) = \dfrac{1}{b-a+1} \qquad \text{for } x = a, a+1, \ldots, b
$$

Note:

1. It is non-parametric because there are no parameters associated.
```

```{prf:definition} Discrete Uniform Distribution (CDF)
:label: def_discrete_uniform_cdf

The cumulative distribution function (CDF) of a discrete random variable $X$ that follows a Uniform distribution
is given by

$$
\cdf(x) = \P(X \leq x) = \begin{cases}
0 & \text{if } x < a \\
\dfrac{x-a+1}{b-a+1} & \text{if } a \leq x \leq b \\
1 & \text{if } x > b
\end{cases}
$$

where $a$ and $b$ are the lower and upper bounds of the set $\S$.
```

## Plotting PMF and CDF of Poisson Distribution

The below plot shows the PMF and its Empirical Histogram distribution for an Uniform distribution
with $a=1$ and $b=6$, essentially a dice roll.

```{code-cell} ipython3
:tags: [hide-input]
from plot import plot_discrete_uniform_pmf, plot_empirical_discrete_uniform

fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=125)
low, high = 1, 6  # [1, 6] for dice roll
plot_discrete_uniform_pmf(low, high, fig=fig, ax=axes[0])
plot_empirical_discrete_uniform(low, high, fig=fig, ax=axes[1])
plt.show()
```