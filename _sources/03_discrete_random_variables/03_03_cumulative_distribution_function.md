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

# Cumulative Distribution Function

The PMF is one way to describe the distribution of a discrete random variable. As we will see later on,
PMF cannot be defined for continuous random variables. The cumulative distribution function (CDF) 
of a random variable is another method to describe the distribution of random variables.
The advantage of the CDF is that it can be defined for any kind of random variable (discrete, 
continuous, and mixed). {cite}`Cumulati3:online`.

The take away lesson here is that CDF is another way to describe the distribution of a random variable.
In particular, in continuous random variables, we do not have an equivalent of PMF, so we use CDF instead.

## Definition

````{prf:definition} Cumulative Distribution Function
:label: def_cdf

Let $X$ be a discrete random variable with $\S = \lset \xi_1, \xi_2, \dots, \xi_n \rset$ where $\xi_i \in \R$ for all $i$.

Then the **cumulative distribution function** $\cdf$ is defined as 

$$
\cdf(\xi_k) \overset{\text{def}}{=} \P \lsq X \leq \xi_k \rsq = \sum_{\ell=1}^k \P \lsq X = \xi_{\ell} \rsq = \sum_{\ell=1}^k \pmf(\xi_{\ell})
$$ (eq:def_cdf)

Since $\P \lsq X = \xi_{\ell} \rsq$ is the probability mass function, we can also replace 
the symbol with the $\pmf$ symbol.

If $\S = \lset \ldots, -1, 0, 1, 2, \ldots \rset$, then we can write the CDF as

$$
\cdf(k) \overset{\text{def}}{=} \P \lsq X \leq k \rsq = \sum_{\ell=-\infty}^k \P \lsq X = \ell \rsq = \sum_{\ell=-\infty}^k \pmf(\ell)
$$
````

````{prf:example} CDF 
:label: example_cdf

Consider a random variable $X$ with the following probability mass function:

$$
\pmf(x) = \begin{cases}
    \frac{1}{4} & \text{if } x = 0 \\
    \frac{1}{2} & \text{if } x = 1 \\
    \frac{1}{4} & \text{if } x = 4 \\
\end{cases}
$$

Then by definition {prf:ref}`def_cdf`, we have the CDF of $X$ to be computed as:

$$
\begin{align}
    \cdf(0) & = \P \lsq X \leq 0 \rsq = \P \lsq X = 0 \rsq = \frac{1}{4}                                                                           \\
    \cdf(1) & = \P \lsq X \leq 1 \rsq = \P \lsq X = 0 \rsq + \P \lsq X = 1 \rsq = \frac{1}{4} + \frac{1}{2} = \frac{3}{4}                          \\
    \cdf(4) & = \P \lsq X \leq 4 \rsq = \P \lsq X = 0 \rsq + \P \lsq X = 1 \rsq + \P \lsq X = 4 \rsq = \frac{1}{4} + \frac{1}{2} + \frac{1}{4} = 1
\end{align}
$$

Thus, our CDF is given by:

$$
\cdf(x) = \begin{cases}
    \frac{1}{4} & \text{if } x \leq 0 \\
    \frac{3}{4} & \text{if } 0 < x \leq 1 \\
    1          & \text{if } x > 1
\end{cases}
$$
````

```{code-cell} ipython3
:tags: [hide-input]
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib.pyplot as plt
p = np.array([0.25, 0.5, 0.25])
x = np.array([0, 1, 4])
F = np.cumsum(p)
plt.stem(x,p,use_line_collection=True); plt.show()
plt.step(x,F); plt.show();
```