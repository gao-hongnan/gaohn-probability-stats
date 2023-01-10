# Functions of Random Variables

## Intuition

In {cite}`chan_2021`, the author used the following example to illustrate the 
concept of functions of random variables.

Conside a random variable $X$ with PDF $\pdf(x)$ and CDF $\cdf(x)$. Let $Y = g(X)$, 
where $g$ is a known and fixed function. We also further assume it is strictly monotonically
increasing for simplicity since such functions are guaranteed to be continuous
and invertible.

Then, the CDF of $Y$ can be derived as follows:

$$
\begin{aligned}
\cdf(y) \defa \P \lsq Y \leq y \rsq &\defb \P \lsq g(X) \leq y \rsq \\ 
                                    &\defc \P \lsq X \leq g^{-1}(y) \rsq \\
                                    &\defd \cdf(g^{-1}(y)) \\
\end{aligned}
$$

where $g^{-1}(y)$ is the inverse function of $g$. Note in particular that step $(c)$
assumes that $g$ is invertible, which is guaranteed by the assumption that $g$ is
strictly monotonically increasing. Step $(d)$ is just the definition of CDF.

Further intuition can be found in Chan's book, see {ref}`functions-of-random-variables-further-readings`.

## CDF of Functions of Random Variables

```{prf:remark} Finding CDF is easier

Given a random variable $X$ and a function $g$, finding the CDF of $Y = g(X)$ is easier
than directly finding the PDF of $Y$. This is because the CDF of $Y=g(X)$ is a monotonically
increasing function, and consequently, the input and output of $g$ are also monotonically
increasing, regardless of $g$. In contrast, the PDF of $Y$ is not necessarily monotonically increasing, and
can be non-linear, and $g$ can also be non-linear, their interactions can be complicated {cite}`chan_2021`.

After finding the CDF of $Y$, we can always find the PDF of $Y$ by taking the derivative of the CDF.
```

## PDF of Functions of Random Variables

As mentioned, we can find the PDF of $Y$ by taking the derivative of the CDF of $Y$. We formulate
it as follows.

```{prf:theorem} The Method of Transformations
:label: theorem:method-of-transformations

Let $X$ be a random variable with PDF $\pdf(x)$ and CDF $\cdf(x)$. 

Let $g$ be a known and fixed function with the following properties:

- $g$ is strictly monotonically increasing.
- $g$ is invertible, i.e., $g^{-1}$ exists.
- $g$ is continuous (differentiable).

Let $Y = g(X)$, then the PDF of $Y$ is given by

$$
f_Y(y) = \begin{cases} \frac{f_X\left(x\right)}{\left|g^{\prime}\left(x\right)\right|} = f_X\left(x\right) \cdot \lvert \frac{dx}{dy} \rvert & \text { where } g\left(x\right)=y \\ 0 & \text { if } g(x)=y \text { does not have a solution }\end{cases}
$$


where $g'(g^{-1}(y))$ is the derivative of $g$ at $g^{-1}(y)$.
```

(functions-of-random-variables-further-readings)=
## Further Readings

- Chan, Stanley H. "Chapter 4.7. Functions of Random Variables." In Introduction to Probability for Data Science, 223-229. Ann Arbor, Michigan: Michigan Publishing Services, 2021. 
- Pishro-Nik, Hossein. "Chapter 4.1.3. Functions of Continuous Random Variables." In Introduction to Probability, Statistics, and Random Processes, 236-242. Kappa Research, 2014. 
- [PSU Stat 414. Lesson 22: Functions of One Random Variable](https://online.stat.psu.edu/stat414/lesson/22)