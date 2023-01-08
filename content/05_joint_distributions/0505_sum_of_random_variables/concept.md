# Concept

## Intuition through Convolutions

{cite}`chan_2021` section 5.5.1

## Convolutions of Random Variables

```{prf:theorem} Convolutions of Random Variables
:label: theorem:convolutions-of-random-variables

Let $X$ and $Y$ be two independent random variables with PDFs $f_X$ and $f_Y$, respectively.
Define $Z = X + Y$ where $Z$ is in itself a random variable.
Then, the PDF of $Z$ is given by

$$
f_Z(z) = \left(f_X \ast f_Y\right)(z) = \int_{-\infty}^{\infty} f_X(x) f_Y(z - x) \, \mathrm{d}x
$$

where $\ast$ denotes convolution.
```

## Sum of Common Distribution

There are proofs on how to derive the convolution of two or more PDFs using the convolution theorem.
We will not go into the details of the proof here.

````{prf:theorem} Sum of Common Distributions
:label: theorem:sum-of-common-distributions

Let $X_1$ and $X_2$ be two independent random variables that come from the same family of distributions.
Then, the PDF of $X_1 + X_2$ is given by

```{list-table} Sum of Common Distributions
:header-rows: 1
:name: table:sum-of-common-distributions

* - $X_1$
  - $X_2$
  - $X_1 + X_2$
* - $\bern(p)$
  - $\bern(p)$
  - $\binomial(n, p)$
* - $\binomial(n, p)$
  - $\binomial(m, p)$
  - $\binomial(m+n, p)$
* - $\poisson(\lambda_1)$
  - $\poisson(\lambda_2)$
  - $\poisson(\lambda_1 + \lambda_2)$
* - $\exponential(\lambda)$
  - $\exponential(\lambda)$
  - $\operatorname{Erlang}(2, \lambda)$
* - $\gaussian(\mu_1, \sigma_1)$
  - $\gaussian(\mu_2, \sigma_2)$
  - $\gaussian(\mu_1 + \mu_2, \sigma_1^2 + \sigma_2^2)$
```

This holds for $N$ random variables as well.
````

## Further Readings

- Chan, Stanley H. "Chapter 5.5. Sum of Two Random Variables." In Introduction to Probability for Data Science, 280-286. Ann Arbor, Michigan: Michigan Publishing Services, 2021. 
