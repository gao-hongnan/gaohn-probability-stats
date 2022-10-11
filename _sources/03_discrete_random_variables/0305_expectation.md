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

```{prf:property} The Law of The Unconscious Statistician
:label: prop_expectation_function

For any function $g$,

$$
\exp \lsq g(X) \rsq = \sum_{x \in X(\S)} g(x) \cdot \P \lsq X = x \rsq
$$

This is not a trivial result, [proof](https://en.wikipedia.org/wiki/Law_of_the_unconscious_statistician)
can be found here.
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

## Concept

````{admonition} Concept
:class: important

**Expectation** is a measure of the average value of a random variable and is **deterministic**.

**Average** is a measure of the average value of a **random sample** from the true population
and is **random**.
````

## Further Readings

- [Functions of Random Variables](https://www.probabilitycourse.com/chapter3/3_2_3_functions_random_var.php) of {cite}`probability`.
- Chapter 3.4.3 of Introduction to Probability for Data Science.
