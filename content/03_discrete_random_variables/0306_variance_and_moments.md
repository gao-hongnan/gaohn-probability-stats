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

# Moments and Variance

Let $\P$ be a probability function defined over the probability space $\pspace$.

Let $X$ be a discrete random variable with $\S = \lset \xi_1, \xi_2, \ldots \rset$
unless otherwise stated.

## Moments

```{prf:definition} Moments
:label: def_moments

The **$k$-th moment** of $X$ is defined as:

$$
\exp \lsq X^k \rsq = \sum_{x \in X(\S)} x^k \cdot \P \lsq X = x \rsq
$$

This definition follows from {prf:ref}`prop_expectation_function` in {doc}`0305_expectation`.

Essentially, this means that the $k$-th moment is the **expectation** of $X^k$.
```

## Variance

```{prf:definition} Variance
:label: def_variance

The **variance** of $X$ is defined as:

$$
\var \lsq X \rsq = \exp \lsq \lpar X - \mu \rpar^2 \rsq
$$ (eq_var_1)

where $\mu = \exp \lsq X \rsq$ is the **expectation** of $X$.

We denote $\sigma^2$ as $\var$ for short-hand notation.
```

We also have an equivalent definition of variance, which is more used in practice.

```{prf:definition} Variance (Alternative)
:label: def_variance_alt

The **variance** of $X$ is defined as:

$$
\var \lsq X \rsq = \exp \lsq X^2 \rsq - \exp \lsq X \rsq^2
$$ (eq_var_2)
```


## Standard Deviation

```{prf:definition} Standard Deviation
:label: def_standard_deviation

In the definition of {prf:ref}`def_variance`, we have $\var \lsq X \rsq$ to have
a different unit than $X$. If $X$ is measured in meters, then $\var \lsq X \rsq$
is measured in meters squared. To solve this issue, we define a new measure
called the **standard deviation**, which is the square root of the variance {cite}`probability`.

$$
\std \lsq X \rsq = \sqrt{\var \lsq X \rsq}
$$ (eq_std_1)
```

## Properties of Moments and Variance

The properties of moments and variance are as follows:

```{prf:property} Scaling
:label: prop_scaling

For any constant $c$, we have:

$$
\exp \lsq c \cdot X \rsq = c^k \cdot \exp \lsq X \rsq
$$ (eq_scaling_1)

where $k$ is the order of the moment.
```

```{prf:property} DC Shift
:label: prop_dc_shift

For any constant $c$, we have:

$$
\exp \lsq (X + c) \rsq = \exp \lsq X \rsq
$$ (eq_dc_shift_1)

The intuition is that shifting the random variable by a constant does not change
the spread of the random variable.
```

```{prf:property} Linearity
:label: prop_linearity

Combining {prf:ref}`prop_scaling` and {prf:ref}`prop_dc_shift`, we have:

$$
\exp \lsq a \cdot X + b \rsq = a^k \cdot \exp \lsq X \rsq
$$ (eq_linearity_1)

where $k$ is the order of the moment.
```

## Further Readings

- [Variance](https://www.probabilitycourse.com/chapter3/3_2_4_variance.php) of {cite}`probability`.
- Chapter 3.4.4 of Introduction to Probability for Data Science.
