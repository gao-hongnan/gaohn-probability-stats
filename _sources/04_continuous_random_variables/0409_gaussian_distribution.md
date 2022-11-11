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

# Gaussian Distribution

Perhaps the most important distribution is the Gaussian Distribution, also known as the Normal Distribution.

## Definition

```{prf:definition} Gaussian Distribution (PDF)
:label: def_gaussian_distribution_pdf

$X$ is a continuous random variable with a **Gaussian distribution** if the probability density function is given by:

$$
\pdf(x) = \frac{1}{\sqrt{2 \pi \sigma^2}} \e \left\{-\frac{(x - \mu)^2}{2 \sigma^2}\right\}
$$ (eq:def_gaussian_distribution)

where $\mu$ is the mean and $\sigma^2$ is the variance and are parameters of the distribution.

Some conventions:
1. We write $X \sim \gaussian(\mu, \sigma^2)$ to indicate that $X$ has a Gaussian distribution with mean $\mu$ and variance $\sigma^2$.
2. We sometimes denote $\gaussian$ as $\normal$ or $\gaussiansymbol$.
```

```{prf:definition} Gaussian Distribution (CDF)
:label: def_gaussian_distribution_cdf

There is no closed form for the CDF of the Gaussian distribution. 
But we will see that it is easy to compute the CDF of the Gaussian distribution 
using the CDF of the standard normal distribution later.

Therefore, the CDF of the Gaussian distribution is given by:

$$
\cdf(x) = \int_{-\infty}^x \pdf(x) \, \mathrm{d}x
$$ (eq:def_gaussian_distribution_cdf)

Yes, you have to integrate the PDF to get the CDF but there is no "general" solution.
```

The PDF of the Gaussian distribution is shown below. The code is referenced and modified
from {cite}`foundations_of_data_science_with_python_2021`.

```{code-cell} ipython3
:tags: [hide-input]
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

mean_1, mean_2, mean_3, mean_4, mean_5 = -3, -0.3, 0, 1.2, 3
sigma = 1

x = np.linspace(-10, 10, 1000)

all_params = [
    (mean_1, sigma),
    (mean_2, sigma),
    (mean_3, sigma),
    (mean_4, sigma),
    (mean_5, sigma),
]

for params in all_params:
    N = stats.norm(params[0], params[1])
    label = f"Normal({N.mean()}, $\sigma=${N.std() :.1f})"
    plt.plot(x, N.pdf(x), label=label)

plt.xlim(-12, 12)
plt.ylim(0, 0.5)
plt.legend(prop={'size': 6})
plt.title("Fixing $\sigma$ = 1, and varying $\mu$")
plt.show()

mean = 0
sigma_1, sigma_2, sigma_3, sigma_4, sigma_5 = 0.3, 0.5, 1, 4, 6

x = np.linspace(-10, 10, 1000)

all_params = [
    (mean, sigma_1),
    (mean, sigma_2),
    (mean, sigma_3),
    (mean, sigma_4),
    (mean, sigma_5),
]

for params in all_params:
    N = stats.norm(params[0], params[1])
    label = f"Normal({N.mean()}, $\sigma=${N.std() :.1f})"
    plt.plot(x, N.pdf(x), label=label)

plt.xlim(-12, 12)
plt.ylim(0, 1.5)
plt.legend(prop={"size": 6})
plt.title("Fixing $\mu$ = 0, and varying $\sigma$")
plt.show()
```

```{prf:remark} Gaussian Distribution (PDF)
:label: rmk_gaussian_distribution_pdf

Note in the plots above, the PDF of the Gaussian distribution is symmetric about the mean $\mu$, and 
given a small enough $\sigma$, the PDF can be greater than 1, as long as the area under the curve is 1.

Note that the PDF curve moves left and right as $\mu$ increases and decreases, respectively.
Similarly, the PDF curve gets narrower and wider as $\sigma$ increases and decreases, respectively.
```

## Expectation and Variance

```{prf:theorem} Expectation and Variance of the Gaussian Distribution

The observant reader may notice that the mean and variance of the Gaussian distribution are parameters of the distribution.
However, this is not proven previously.

If $X$ is a continuous random variable with an gaussian distribution with mean $\mu$ and variance $\sigma^2$, then the
expectation and variance of $X$ are given by:

$$
\expectation \lsq X \rsq = \mu \qquad \text{and} \qquad \variance \lsq X \rsq = \sigma^2
$$ (eq:gaussian_distribution_expectation_variance)
```




## Further Readings

- Chan, Stanley H. "Chapter 4.6. Gaussian Random Variables." In Introduction to Probability for Data Science, 211-223. Ann Arbor, Michigan: Michigan Publishing Services, 2021. 
- Pishro-Nik, Hossein. "Chapter 4.2.3. Normal (Gaussian) Distribution" In Introduction to Probability, Statistics, and Random Processes, 253-260. Kappa Research, 2014. 