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

# Expectation

## Definition

```{prf:definition} Expectation
:label: def_expectation

Let $\P$ be a probability function defined over the probability space $\pspace$.

Let $X$ be a discrete random variable with $\S = \lset \xi_1, \xi_2, \ldots \rset$.

Then the **expectation** of $X$ is defined as:

$$
\exp(X) = \sum_{x \in X(\S)} x \cdot \P \lsq X = x \rsq
$$
```

## Existence of Expectation

```{prf:theorem} Existence of Expectation
:label: thm_existence_of_expectation

Let $\P$ be a probability function defined over the probability space $\pspace$.

A discrete random variable $X$ with $\S = \lset \xi_1, \xi_2, \ldots \rset$ has an **expectation** if 
and only if it is **absolutely summable**.

That is, 

$$
\exp \lsq \lvert X \rvert \rsq \overset{\text{def}}{=} \sum_{x \in X(\S)} \lvert x \rvert \cdot \P \lsq X = x \rsq < \infty
$$
```

## Properties of Expectation

Let $\P$ be a probability function defined over the probability space $\pspace$.

Let $X$ be a discrete random variable with $\S = \lset \xi_1, \xi_2, \ldots \rset$.

Then the ***expectation*** of $X$ has the following properties:

```{prf:property} Function
:label: prop_expectation_function

For any function $g$,

$$
\exp \lsq g(X) \rsq = \sum_{x \in X(\S)} g(x) \cdot \P \lsq X = x \rsq
$$
```

```{prf:property} Linearity
:label: prop_expectation_linearity

For any constants $a$ and $b$, 

$$
\exp \lsq aX + b \rsq = a \cdot \exp(X) + b
$$
```

```{prf:property} Scaling
:label: prop_expectation_scaling

For any constant $c$,

$$
\exp \lsq cX \rsq = c \cdot \exp(X)
$$
```

```{prf:property} DC Shift
:label: prop_expectation_dc_shift

For any constant $c$,

$$
\exp \lsq X + c \rsq = \exp(X)
$$
```

```{prf:property} Stronger Linearity
:label: prop_expectation_stronger_linearity

It follows that for any random variables $X_1$, $X_2$, ..., $X_n$,

$$
\exp \lsq \sum_{i=1}^n a_i X_i \rsq = \sum_{i=1}^n a_i \cdot \exp \lsq X_i \rsq
$$
```

A very important concept to remember here is that when we apply a function $g$ to a random variable $X$,
the PMF of the resulting random variable $g(X)$ is the same as the PMF of $X$.

For example consider the example of a dice roll, where $X$ is the random variable representing the outcome of a dice roll.

Then $g(X) = X^2$ is the random variable representing the square of the outcome of a dice roll.

We know the underlying PMF of $X$ is 

$$
\begin{align}
\P \lsq X = 1 \rsq &= \frac{1}{6} \\
\P \lsq X = 2 \rsq &= \frac{1}{6} \\
\P \lsq X = 3 \rsq &= \frac{1}{6} \\
\P \lsq X = 4 \rsq &= \frac{1}{6} \\
\P \lsq X = 5 \rsq &= \frac{1}{6} \\
\P \lsq X = 6 \rsq &= \frac{1}{6}
\end{align}
$$

Let $Y = g(X) = X^2$.

Then you can treat this as a "change of variable" where we now assign the outcomes in the sample space of $X$
to be all squared, essentially $\S = \lset 1^2, 2^2, 3^2, 4^2, 5^2, 6^2 \rset$.

There is no change in the underlying PMF of $X$, so the PMF of $Y$ is the same as the PMF of $X$.

## Concept


````{admonition} Concept
:class: important

**Expectation** is a measure of the average value of a random variable and is **deterministic**.

**Average** is a measure of the average value of a **random sample** from the true population
and is **random**.
````

Let $X$ be a discrete random variable that represents the height of a person in the true population.
