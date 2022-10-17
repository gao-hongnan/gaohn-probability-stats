# Poisson Distribution

## Definition

```{prf:definition} Poisson Distribution
:label: def:poi

Let $X$ be a **Poisson random variable**. Then the probability mass function (PMF) of $X$ is given by

$$
\P \lsq X = k \rsq = \frac{\lambda^k e^{-\lambda}}{k!} \qquad \text{for } k = 0, 1, 2, \ldots
$$

where $\lambda > 0$ is called the Poisson parameter.

We write

$$
X \sim \poisson(\lambda)
$$

to say that $X$ is drawn from a Poisson distribution with parameter $\lambda$.
```

## Properties

```{prf:property} Expectation of Poisson Distribution
:label: prop:poi_exp

Let $X \sim \poisson(\lambda)$ be a Poisson random variable with parameter $\lambda$. Then the expectation of $X$ is given by

$$
\exp \lsq X \rsq = \sum_{k=0}^{\infty} k \cdot \P \lsq X = k \rsq = \lambda
$$
```

```{prf:property} Variance of Poisson Distribution
:label: prop:poi_var

Let $X \sim \poisson(\lambda)$ be a Poisson random variable with parameter $\lambda$. Then the variance of $X$ is given by

$$
\var \lsq X \rsq = \exp \lsq X^2 \rsq - \exp \lsq X \rsq^2 = \lambda
$$
```

## Poisson Approximation to Binomial Distribution

```{prf:theorem} Poisson Approximation to Binomial Distribution
:label: thm:poi_bin

For situations where the number of trials $n$ is large and the probability of success $p$ is small, 
the Poisson distribution is a good approximation to the Binomial distribution.

Recall the Binomial distribution is given by

$$
X \sim \binomial(n, p) \qquad \text{with PMF} \qquad \P \lsq X = k \rsq = \binom{n}{k} p^k (1-p)^{n-k} \qquad \text{for } k = 0, 1, 2, \ldots, n
$$

Then the approximation is given by

$$
\binom{n}{k} p^k (1-p)^{n-k} \approx \frac{\lambda^k e^{-\lambda}}{k!} \qquad \text{for } k = 0, 1, 2, \ldots
$$

where $\lambda = np$.
```