# Naive Bayes Concept

## Notations

```{prf:definition} Underlying Distributions
:label: underlying-distributions

- $\mathcal{X}$: Input space consists of all possible inputs $\mathbf{x} \in \mathcal{X}$.
  
- $\mathcal{Y}$: Label space = $\{1, 2, \cdots, K\}$ where $K$ is the number of classes.
  
- The mapping between $\mathcal{X}$ and $\mathcal{Y}$ is given by $c: \mathcal{X} \rightarrow \mathcal{Y}$ where $c$ is called *concept* according to the PAC learning theory.
  
- $\mathcal{D}$: The fixed but unknown distribution of the data. Usually, this refers 
to the joint distribution of the input and the label, 

  $$
  \mathcal{D} &= \mathbb{P}(\mathcal{X}, \mathcal{Y} ; \boldsymbol{\theta}) \\
  &= \mathbb{P}_{\{\mathcal{X}, \mathcal{Y} ; \boldsymbol{\theta}\}}(\mathbf{x}, y)
  $$

  where $\mathbf{x} \in \mathcal{X}$ and $y \in \mathcal{Y}$, and $\boldsymbol{\theta}$ is the 
  parameter vector of the distribution $\mathcal{D}$.
```


```{prf:definition} Dataset
:label: dataset-definition

Now, consider a dataset $\mathcal{D}_{\{\mathbf{x}, y\}}$ consisting of $N$ samples (observations) and $D$ predictors (features) drawn **jointly** and **indepedently and identically distributed** (i.i.d.) from $\mathcal{D}$. Note we will refer to the dataset $\mathcal{D}_{\{\mathbf{x}, y\}}$ with the same notation as the underlying distribution $\mathcal{D}$ from now on. 
  
- The training dataset $\mathcal{D}$ can also be represented compactly as a set:
  
    $$
    \begin{align*}
    \mathcal{D} \overset{\mathbf{def}}{=} \mathcal{D}_{\{\mathbf{x}, y\}} &= \left\{\mathbf{x}^{(n)}, y^{(n)}\right\}_{n=1}^N \\
    &= \left\{\left(\mathbf{x}^{(1)}, y^{(1)}\right), \left(\mathbf{x}^{(2)}, y^{(2)}\right), \cdots, \left(\mathbf{x}^{(N)}, y^{(N)}\right)\right\} \\
    &= \left\{\mathbf{X}, \mathbf{y}\right\}
    \end{align*}
    $$

  where we often subscript $\mathbf{x}$ and $y$ with $n$ to denote the $n$-th sample from the dataset, i.e. 
  $\mathbf{x}^{(n)}$ and $y^{(n)}$. Most of the times, $\mathbf{x}^{(n)}$ is bolded since
  it represents a vector of $D$ number of features, while $y^{(n)}$ is not bolded since it is a scalar, though
  it is not uncommon for $y^{(n)}$ to be bolded as well if you represent it with K-dim one-hot vector.

- For the n-th sample $\mathbf{x}^{(n)}$, we often denote the $d$-th feature as $x_d^{(n)}$ and the representation of $\mathbf{x}^{(n)}$ as a vector as: 
  
  $$
  \mathbf{x}^{(n)} \in \mathbb{R}^{D} = \begin{bmatrix} x_1^{(n)} & x_2^{(n)} & \cdots & x_D^{(n)} \end{bmatrix}_{D \times 1}
  $$
  
  is a sample of size $D$, drawn (jointly with $y$) $\textbf{i.i.d.}$ from $\mathcal{D}$. 

- We often add an extra feature $x_0^{(n)} = 1$ to $\mathbf{x}^{(n)}$ to represent the bias term. 
i.e. 
  
  $$
  \mathbf{x}^{(n)} \in \mathbb{R}^{D+1} = \begin{bmatrix} x_0^{(n)} & x_1^{(n)} & x_2^{(n)} & \cdots & x_D^{(n)} \end{bmatrix}_{(D+1) \times 1}
  $$
  
- For the n-th sample's label $y^{(n)} \overset{\mathbf{def}}{=} c(\mathbf{x}^{(n)})$, if we were to represent it as K-dim one-hot vector, we would have:

  $$
  y^{(n)} \in \mathbb{R}^{K} = \begin{bmatrix} 0 & 0 & \cdots & 1 & \cdots & 0 \end{bmatrix}_{K \times 1}
  $$

  where the $1$ is at the $k$-th position, and $k$ is the class label of the n-th sample.

- Everything defined above is for **one single sample/data point**, to represent it as a matrix, we can define
a design matrix $\mathbf{X}$ and a label vector $\mathbf{y}$ as follows,

  $$
  \begin{aligned}
  \mathbf{X} \in \mathbb{R}^{N \times D} &= \begin{bmatrix} \mathbf{x}^{(1)} \\ \mathbf{x}^{(2)} \\ \vdots \\ \mathbf{x}^{(N)} \end{bmatrix} = \begin{bmatrix} x_1^{(1)} & x_2^{(1)} & \cdots & x_D^{(1)} \\ x_1^{(2)} & x_2^{(2)} & \cdots & x_D^{(2)} \\ \vdots & \vdots & \ddots & \vdots \\ x_1^{(N)} & x_2^{(N)} & \cdots & x_D^{(N)} \end{bmatrix}_{N \times D} \\
  \end{aligned}
  $$

  as the matrix of all samples. Note that each row is a sample and each column is a feature. We can append a column of 1's to the first column of $\mathbf{X}$ to represent the bias term.

  **In this section, we also talk about random vectors $\mathbf{X}$ so we will replace the design matrix $\mathbf{X}$ with $\mathbf{A}$ to avoid confusion.**

  Subsequently, for the label vector $\mathbf{y}$, we can define it as follows,

  $$
  \begin{aligned}
  \mathbf{y} \in \mathbb{R}^{N} = \begin{bmatrix} y^{(1)} \\ y^{(2)} \\ \vdots \\ y^{(N)} \end{bmatrix}
  \end{aligned}
  $$
```

```{prf:definition} The i.i.d. Assumption
:label: iid-assumption

In supervised learning, implicitly or explicitly, one *always* assumes that the training set
$\mathcal{D} = \left\{\left(\mathbf{x}^{(1)}, y^{(1)}\right), \left(\mathbf{x}^{(2)}, y^{(2)}\right), \cdots, \left(\mathbf{x}^{(N)}, y^{(N)}\right)\right\} $ is composed of $N$ input/response tuples $\left({\mathbf{X}}^{(n)} = \mathbf{x}^{(n)}, Y^{(n)} = y^{(n)}\right)$ that are *independently drawn from the same joint distribution* $\mathbb{P}_{\{\mathcal{X}, \mathcal{Y}, \boldsymbol{\theta}\}}(\mathbf{x}, y)$ with

$$
\mathbb{P}(\mathbf{X} = \mathbf{x}, Y = y ; \boldsymbol{\theta}) = \mathbb{P}(Y = y \mid \mathbf{X} = \mathbf{x}) \mathbb{P}(\mathbf{X} = \mathbf{x})
$$

and $\mathbb{P}(Y = y \mid \mathbf{X} = \mathbf{x})$ is the conditional probability of $Y$ given $\mathbf{X}$,
the relationship that the learner algorithm is trying to capture.

Mathematically, this i.i.d. assumption writes (also defined in {prf:ref}`def_iid`):

$$
\begin{aligned}
\left({\mathbf{X}}^{(n)}, Y^{(n)}\right) &\sim \mathbb{P}_{\{\mathcal{X}, \mathcal{Y}, \boldsymbol{\theta}\}}(\mathbf{x}, y)\\
\left({\mathbf{X}}^{(n)}, Y^{(n)}\right) &\text{ independent of } \left({\mathbf{X}}^{(m)}, Y^{(m)}\right) \quad \forall n \neq m \in \{1, 2, \ldots, N\}
\end{aligned}
$$

and we sometimes denote 

$$
\begin{aligned}
\left(\mathbf{x}^{(n)}, y^{(n)}\right) \overset{\text{i.i.d.}}{\sim} \mathbb{P}_{\{\mathcal{X}, \mathcal{Y}, \boldsymbol{\theta}\}}(\mathbf{x}, y)
\end{aligned}
$$
```

```{prf:example} Joint Distribution Example
:label: joint-distribution-example

For example, if the number of features, $D = 2$, then let's say

$$
\mathbf{X}^{(n)} = \begin{bmatrix} X^{(n)}_1 & X^{(n)}_2 \end{bmatrix} \in \mathbb{R}^2
$$

consists of two Gaussian random variables, 
with $\mu_1$ and $\mu_2$ being the mean of the two distributions,
and $\sigma_1$ and $\sigma_2$ being the variance of the two distributions;
furthermore, $Y^{(n)}$ is a Bernoulli random variable with parameter $\boldsymbol{\pi}$, then we have 

$$
\begin{aligned}
\boldsymbol{\theta} &= \begin{bmatrix} \mu_1 & \sigma_1 & \mu_2 & \sigma_2 & \boldsymbol{\pi}\end{bmatrix} \\
&= \begin{bmatrix} \boldsymbol{\mu} & \boldsymbol{\sigma} & \boldsymbol{\pi} \end{bmatrix}
\end{aligned}
$$
  
where $\boldsymbol{\mu} = \begin{bmatrix} \mu_1 & \mu_2 \end{bmatrix}$ and $\boldsymbol{\sigma} = \begin{bmatrix} \sigma_1 & \sigma_2 \end{bmatrix}$.
```

```{prf:remark} Some remarks
:label: some-remarks

- From now on, we will refer the realization of $Y$ as $k$ instead.
- For some sections, when I mention $\mathbf{X}$, it means the random vector which resides in the
$D$-dimensional space, not the design matrix. This also means that this random vector refers
to a single sample, not the entire dataset.
```

```{prf:definition} Joint and Conditional Probability
:label: joint-and-conditional-probability

We are often interested in finding the probability of a label given a sample, 

$$
\begin{aligned}
\mathbb{P}(Y = k \mid \mathbf{X} = \mathbf{x}) &= \mathbb{P}(Y = k \mid \mathbf{X} = \left(x_1, x_2, \ldots, x_D\right)) 
\end{aligned}
$$

where

$$
\mathbf{X} \in \mathbb{R}^{D} = \begin{bmatrix} X_1 & X_2 & \cdots & X_D \end{bmatrix} 
$$

is a random vector and its realizations,

$$
\mathbf{x} = \begin{bmatrix} x_1 & x_2 & \cdots & x_D \end{bmatrix}
$$

and therefore, $\mathbf{X}$ can be characterized by an $D$-dimensional PDF

$$
f_{\mathbf{X}}(\mathbf{x}) = f_{X_1, X_2, \ldots, X_D}(x_1, x_2, \ldots, x_D ; \boldsymbol{\theta})
$$

and

$$
Y \in \mathbb{Z} \quad \text{and} \quad k \in \mathbb{Z}
$$

is a discrete random variable (in our case classification) and its realization respectively, and therefore, $Y$ can be characterized by a discrete PDF (PMF)

$$
f_{Y}(k ; \boldsymbol{\pi}) \sim \text{Categorical}(\boldsymbol{\pi})
$$


**Note that we are talking about one single sample tuple $\left(\mathbf{x}, y\right)$ here. I did not
index the sample tuple with $n$ because this sample can be any sample in the unknown distribution $\mathbb{P}_{\mathcal{X}, \mathcal{Y}}(\mathbf{x}, y)$
and not only from our given dataset $\mathcal{D}$.**
```

```{prf:definition} Likelihood
:label: likelihood

We denote the likelihood function as $\mathbb{P}(\mathbf{X} = \mathbf{x} \mid Y = k)$, 
which is the probability of observing $\mathbf{x}$ given that the sample belongs to class $Y = k$. 
```

```{prf:definition} Prior
:label: prior

We denote the prior probability of class $k$ as $\mathbb{P}(Y = k)$, which usually
follows a discrete distribution such as the Categorical distribution.
```

```{prf:definition} Posterior
:label: posterior

We denote the posterior probability of class $k$ given $\mathbf{x}$ as $\mathbb{P}(Y = k \mid \mathbf{X} = \mathbf{x})$.
```

```{prf:definition} Marginal Distribution and Normalization Constant
:label: marginal-distribution-and-normalization-constant


We denote the normalizing constant as $\mathbb{P}(\mathbf{X} = \mathbf{x}) = \sum_{k=1}^K \mathbb{P}(Y = k) \mathbb{P}(\mathbf{X} = \mathbf{x} \mid Y = k)$.
``` 

Brain dump

1. The data points $\mathbf{x}^{(1)}, \mathbf{x}^{(2)}, \ldots, \mathbf{x}^{(N)}$ are **i.i.d.** (independent and identically distributed) realizations from the random variables (random vectors) $\mathbf{X}^{(1)}, \mathbf{X}^{(2)}, \ldots, \mathbf{X}^{(N)}$. 

2. Each $\mathbf{X}^{(n)}$ is a random vector, i.e. $\mathbf{X}^{(n)} \in \mathbb{R}^{D}$, where $D$ is the dimensionality of the feature space. This means that $\mathbf{X}^{(n)}$ can be characterized by an $D$-dimensional PDF $f_{\mathbf{X}^{(n)}}(\mathbf{x}^{(n)}) = f_{X_1^{(n)}, X_2^{(n)}, \ldots, X_D^{(n)}}(x_1^{(n)}, x_2^{(n)}, \ldots, x_D^{(n)})$.

This means $\mathbf{X}^{(n)}$ is a multi-dimensional joint distribution, 

$$
\begin{aligned}
\mathbf{X}^{(n)} \in \mathbb{R}^{D} = \begin{bmatrix} X_1^{(n)} \\ X_2^{(n)} \\ \vdots \\ X_D^{(n)} \end{bmatrix}
\end{aligned}
$$

and can be characterized by an $D$-dimensional PDF

$$
\begin{aligned}
f_{\mathbf{X}^{(n)}}(\mathbf{x}^{(n)}) = f_{\mathbf{X}^{(n)}}\left(\mathbf{x}^{(n)}\right) = f_{X_1^{(n)}, X_2^{(n)}, \ldots, X_D^{(n)}}\left(x_1^{(n)}, x_2^{(n)}, \ldots, x_D^{(n)}\right)
\end{aligned}
$$

For example, if $D = 3$, and the realizations of $\mathbf{X}^{(n)}$ are $\mathbf{x}^{(n)} = (22, 80, 1)$, 
then $f_{\mathbf{X}^{(n)}}\left(\mathbf{x}^{(n)}\right)$ is the probability density of observing this 3-dimensional vector in $\mathbb{R}^{3}$, within the
sample space $\Omega_{\mathbf{X}} = \mathbb{R}  \times \mathbb{R} \times \mathbb{R}$.

The confusion in the **i.i.d.** assumption is that we are not talking about the individual random variables 
$X_1^{(n)}, X_2^{(n)}, \ldots, X_D^{(n)}$ here, but the entire random vector $\mathbf{X}^{(n)}$.

This means there is no assumption of $X_1^{(n)}, X_2^{(n)}, \ldots, X_D^{(n)}$ being **i.i.d.**. Instead, the samples
$\mathbf{X}^{(1)}, \mathbf{X}^{(2)}, \ldots, \mathbf{X}^{(N)}$ are **i.i.d.**.


3. More confusion, is it iid with label $Y=k$, seems like it is together since we can indeed decompose 
  the $\mathbb{P}(Y=k \mid \mathbf{X} = \mathbf{x})$ into proportional $\mathbb{P}(Y=k) \mathbb{P}(\mathbf{X} = \mathbf{x} \mid Y = k)$.

In SE post, discriminative model do not need make assumptions of X and therefore they may or may not be i.i.d.

## Categorical Distribution

As mentioned earlier, both $Y^{(n)}$ and $\mathbf{X}^{(n)}$ are random variables/vectors. This means we need to estimate both of them.

We first conveniently assume that $Y^{(n)}$ is a discrete random variable, and
follows the **[Category distribution](https://en.wikipedia.org/wiki/Categorical_distribution)**[^categorical-distribution], 
an extension of the Bernoulli distribution to multiple classes. Instead of a single parameter $p$ (probability of success for Bernoulli), 
the Category distribution has $K$ parameters $\boldsymbol{\pi}_k$ for $k = 1, 2, \cdots, K$.

$$
Y^{(n)} \overset{\small{\text{i.i.d.}}}{\sim} \text{Category}(\boldsymbol{\pi}) \quad \text{where } \boldsymbol{\pi} = \begin{bmatrix} \pi_1 & \dots & \pi_K \end{bmatrix}
$$

Equivalently,

$$
\mathbb{P}(Y^{(n)} = k) = \pi_k \quad \text{for } k = 1, 2, \cdots, K
$$ (eq:category-distribution)

```{prf:definition} Categorical Distribution
:label: categorical-distribution

Let $Y$ be a discrete random variable with $K$ number of states. 
Then $Y$ follows a categorical distribution with parameters $\boldsymbol{\pi}$ if

$$
\mathbb{P}(Y = k) = \pi_k \quad \text{for } k = 1, 2, \cdots, K
$$

Consequently, the PMF of the categorical distribution is defined more compactly as,

$$
\mathbb{P}(Y = k) = \prod_{k=1}^K \pi_k^{I\{Y = k\}}
$$

where $I\{Y = k\}$ is the indicator function that is equal to 1 if $Y = k$ and 0 otherwise.
```

```{prf:definition} Categorical (Multinomial) Distribution
:label: categorical-multinomial-distribution

This formulation is adopted by Bishop's{cite}`bishop_2016`, the categorical distribution is defined as

$$
\mathbb{P}(\mathbf{Y} = \mathbf{y}; \boldsymbol{\pi}) = \prod_{k=1}^K \pi_k^{y_k}
$$

where

$$
\mathbf{y} = \begin{bmatrix} y_1 \\ y_2 \\ \vdots \\ y_K \end{bmatrix}
$$

is an one-hot encoded vector of size $K$, 

The $y_k$ is the $k$-th element of $\mathbf{y}$, and is equal to 1 if $Y = k$ and 0 otherwise.
The $\pi_k$ is the $k$-th element of $\boldsymbol{\pi}$, and is the probability of $Y = k$.

This notation alongside with the indicator notation in the previous definition allows us to manipulate
the likelihood function easier.
```


```{prf:example} Categorical Distribution Example
:label: categorical-distribution-example

Consider rolling a fair six-sided die. Let $Y$ be the random variable that represents the outcome of the die roll. Then $Y$ follows a categorical distribution with parameters $\boldsymbol{\pi}_k$ where $\pi_k = \frac{1}{6}$ for $k = 1, 2, \cdots, 6$.

$$
\mathbb{P}(Y = k) = \frac{1}{6} \quad \text{for } k = 1, 2, \cdots, 6
$$

For example, if we roll a 3, then $\mathbb{P}(Y = 3) = \frac{1}{6}$.

With the more compact notation, the indicator function is $I\{Y = k\} = 1$ if $Y = 3$ and $0$ otherwise. Therefore, the PMF is

$$
\mathbb{P}(Y = k) = \prod_{k=1}^6 \frac{1}{6}^{I\{Y = k\}} = \left(\frac{1}{6}\right)^0 \cdot \left(\frac{1}{6}\right)^0 \cdot \left(\frac{1}{6}\right)^1 \cdot \left(\frac{1}{6}\right)^0 \cdot \left(\frac{1}{6}\right)^0 \cdot \left(\frac{1}{6}\right)^0 = \frac{1}{6}
$$

Using Bishop's notation, the PMF is still the same, only the realization $\mathrm{y}$ is not a scalar,
but instead a vector of size $6$. In the case where $Y = 3$, the vector $\mathrm{y}$ is

$$
\mathrm{y} = \begin{bmatrix} 0 \\ 0 \\ 1 \\ 0 \\ 0 \\ 0 \end{bmatrix}
$$
```

## Derivation

Let $\mathcal{D} = \left \{ \left(\mathrm{X}^{(n)}, Y^{(n)} \right) \right \}_{n=1}^N = \left \{ \left(\mathrm{x}^{(n)}, y^{(n)} \right) \right \}_{n=1}^N$ be the dataset
with $N$ samples and $D$ predictors. All samples are assumed to be **independent and identically distributed (i.i.d.)** from the unknown but fixed joint distribution 
$\mathbb{P}(\mathcal{X}, \mathcal{Y} ; \boldsymbol{\theta})$,

$$
\left \{ \left(\mathrm{X}^{(n)}, Y^{(n)} \right) \right \} \overset{\small{\text{i.i.d.}}}{\sim} \mathbb{P}(\mathcal{X}, \mathcal{Y} ; \boldsymbol{\theta}) \quad \text{for } n = 1, 2, \cdots, N
$$

where $\boldsymbol{\theta}$ is the parameter vector of the joint distribution. See {prf:ref}`joint-distribution-example` for an example of such.

Recall that our goal in **INSERT INFERENCE SECTION ALGO** is to find the class $k \in \{1, 2, \cdots, K\}$ that maximizes the posterior probability
$\mathbb{P}(Y = k \mid \mathbf{X} = \mathbf{x}^{(q)} ; \boldsymbol{\theta})$. 

$$
\begin{aligned}
\arg \max_{k} \mathbb{P}(Y = k \mid \mathbf{X} = \mathbf{x}^{(q)} ; \boldsymbol{\theta}) &= \arg \max_{k} \frac{\mathbb{P}(Y = k, \mathbf{X} = \mathbf{x}^{(q)} ; \boldsymbol{\theta})}{\mathbb{P}(\mathbf{X} = \mathbf{x}^{(q)} ; \boldsymbol{\theta})} \\
&= \arg \max_{k} \frac{\mathbb{P}(Y = k ; \boldsymbol{\pi}) \mathbb{P}(\mathbf{X} \mid Y = k ; \boldsymbol{\theta}_{\{\mathbf{X} \mid Y\}})}{\mathbb{P}(\mathbf{X} = \mathbf{x}^{(q)} ; \boldsymbol{\theta})}\\
&\propto \arg \max_{k} \mathbb{P}(Y = k ; \boldsymbol{\pi}) \mathbb{P}\left(\mathbf{X} \mid Y = k ; \boldsymbol{\theta}_{\{\mathbf{X} \mid Y\}}\right)
\end{aligned}
$$

We have seen earlier in **iNSERT SECTION ON NORMALIZATING CONSTANT** that since the denominator 
is constant for all $k$, we can ignore it and just maximize the numerator, as shown by the proportional sign.

This suggests we need to find estimates for both **also insert inference section algo** the prior and the likelihood. This of course 
involves ur finding the $\boldsymbol{\pi}$ and $\boldsymbol{\theta}_{\{\mathbf{X} \mid Y\}}$ that maximize the likelihood function, which we will talk about later.

In order to meaningfully optimize the expression, we need to decompose the numerator into its components that contain the parameters we want to estimate.

$$
\begin{aligned}
\mathbb{P}(Y = k ; \boldsymbol{\pi}) \mathbb{P}(\mathbf{X} \mid Y = k ; \boldsymbol{\theta}) &= \mathbb{P}((Y, \mathbf{X}) = (k, \mathbf{x}^{(q)}) ; \boldsymbol{\theta}, \boldsymbol{\pi}) \\
&= \mathbb{P}(Y, X_1, X_2, \ldots X_D)
\end{aligned}
$$ (eq:joint-distribution)

which is actually the joint distribution of $\mathbf{X}$ and $Y$[^joint-distribution]. 

This joint distribution expression {eq}`eq:joint-distribution` can be further decomposed by the chain rule of probability[^chain-rule-of-probability] as

$$
\begin{aligned}
\mathbb{P}(Y, X_1, X_2, \ldots X_D) &= \mathbb{P}(Y) \mathbb{P}(X_1, X_2, \ldots X_D \mid Y) \\
&= \mathbb{P}(Y) \prod_{d=1}^D \mathbb{P}(X_d \mid Y, X_1, X_2, \ldots X_{d-1})
\end{aligned}
$$ (eq:joint-distribution-decomposed)

This alone does not get us any further, we still need to estimate roughly $2^{D}$ parameters **CITE D2L**, 
which is computationally expensive. Not to forget that we need to estimate for each class $k \in \{1, 2, 3, \ldots, K\}$
which has a complexity of $\sim \mathcal{O}(2^DK)$.

This is where the "Naive" assumption comes in. The Naive Bayes' classifier assumes that the features are conditionally independent[^conditional-independence] given the class label, i.e.
the features are conditionally independent given the class label.

More formally stated,

```{prf:definition} Conditional Independence
:label: conditional-independence

$$
\mathbb{P}(X_d \mid Y = k, X_{d^{'}}) = \mathbb{P}(X_d \mid Y = k) \quad \text{for all } d \neq d^{'}
$$ (eq:conditional-independence)
```

with this assumption, we can further simplify expression {eq}`eq:joint-distribution-decomposed` as

$$
\begin{aligned}
\mathbb{P}(Y, X_1, X_2, \ldots X_D) &= \mathbb{P}(Y ; \boldsymbol{\pi}) \prod_{d=1}^D \mathbb{P}(X_d \mid Y ; \theta_{d}) \\
\end{aligned}
$$

More precisely, after all the simplifications above,

$$
\begin{aligned}
\mathbb{P}(Y = k \mid \mathbf{X} = \mathbf{x} ; \boldsymbol{\theta}) & = \dfrac{\mathbb{P}(Y = k ; \boldsymbol{\pi}) \mathbb{P}(\mathbf{X} \mid Y = k ; \boldsymbol{\theta}_{\{\mathbf{X} \mid Y\}})}{\mathbb{P}(\mathbf{X})} \\
&= \dfrac{\mathbb{P}(Y, X_1, X_2, \ldots X_D)}{\mathbb{P}(\mathbf{X})} \\
&= \dfrac{\mathbb{P}(Y = k ; \boldsymbol{\pi}) \prod_{d=1}^D \mathbb{P}(X_d = x_d \mid Y = k ; \theta_{dk})}{\mathbb{P}(\mathbf{X} = \mathbf{x})} \\
&\propto \mathbb{P}(Y = k ; \boldsymbol{\pi}) \prod_{d=1}^D \mathbb{P}(X_d = x_d \mid Y = k ; \theta_{dk})
\end{aligned}
$$ (eq:naive-bayes-classifier-1)

Consequently, $\boldsymbol{\pi} = \begin{bmatrix} \pi_1 & \pi_2 & \dots & \pi_K \end{bmatrix}$ and
$\pi_k$ refers to the prior probability of class $k$, and $\theta_{dk}$ refers to the parameter of the
class conditional density for class $k$ and feature $d$ (**Cite murphy pp 358**). Furthermore,
the boldsymbol $\boldsymbol{\theta}$ is the parameter vector,

$$
\boldsymbol{\theta} = \left(\boldsymbol{\pi}, \{\theta_{dk}\}_{k=1}^K, \{d=1, \ldots, D\}\right) = \left(\boldsymbol{\pi}, \boldsymbol{\theta}_{\{\mathbf{X} \mid Y\}}\right)
$$


```{prf:definition} The Parameter Vector
:label: parameter-vector

There is not much to say about the categorical component $\boldsymbol{\pi}$, since we are
just estimating the prior probabilities of the classes. 

The parameter vector (matrix) $\boldsymbol{\theta}_{\{\mathbf{X} \mid Y\}}=\{\theta_{dk}\}_{k=1}^K, \{d=1, \ldots, D\}$ is a bit more complicated.
It resides in the $\mathbb{R}^{K \times D}$ space, where each element $\theta_{dk}$ is the parameter
associated with feature $d$ conditioned on class $k$.

So if $K=3$ and $D=2$, then the parameter vector $\boldsymbol{\theta}$ is a $3 \times 2$ matrix, i.e.

$$
\boldsymbol{\theta} = \begin{bmatrix}
\theta_{11} & \theta_{12} \\
\theta_{21} & \theta_{22} \\
\theta_{31} & \theta_{32}
\end{bmatrix}_{3 \times 2}
$$

This means we have effectively reduced our complexity from $\sim \mathcal{O}(2^D)$ to $\sim \mathcal{O}(DK + 1)$.

**We have also reduced the problem of estimating the joint distribution to just individual conditional distributions.**
```

```{prf:remark} Notation remark
A note, the notation $\boldsymbol{\theta}_{dk}$ should either be read as $\boldsymbol{\theta}_{kd}$ since
we say $\boldsymbol{\theta}_{\{\mathbf{X} \mid Y\}}$ is a $K \times D$ matrix.
```

### Inductive Bias

We still need to introduce some inductive bias into {eq}`eq:naive-bayes-classifier-1`, more concretely, we need to make some assumptions about the distribution 
of $\mathbb{P}(Y)$ and $\mathbb{P}(X_d \mid Y)$. 

For the target variable, we typically model it as a categorical distribution, 

$$
\mathbb{P}(Y) \sim \mathrm{Categorical}(\boldsymbol{\pi})
$$

For the conditional distribution of the features, we typically model it according to what type of features we have. For example, if we have binary features, then we can model it as a Bernoulli distribution, 

$$
\mathbb{P}(X_d \mid Y) \sim \mathrm{Bernoulli}(\theta_{dk})
$$

If we have continuous features, then we can model it as a Gaussian distribution,

$$
\mathbb{P}(X_d \mid Y) \sim \mathcal{N}(\mu_{dk}, \sigma_{dk}^2)
$$

See **KEVIN MURPHY pp 358 for more details**.

For simplicity sake, we assume that all features $X_d$ are of the same type, either all binary or all continuous.
In reality, this may not need to be the case.

### Model Fitting

Everything we have talked about is just 1 single sample, and that won't work in the realm of 
estimating the best parameters that fit the data. Since we are given a dataset $\mathcal{D}$
consisting of $N$ samples, we can estimate the parameters of the model by maximizing the likelihood of the data.

Since each sample is **i.i.d.**, we can write the joint probability distribution as the product of the individual probabilities of each sample:

$$
\begin{align*}
\mathbb{P}(\mathcal{D} ; \boldsymbol{\theta}) &= \mathbb{P}\left(\mathcal{D} ; \left\{\boldsymbol{\pi}, \boldsymbol{\theta}_{\{\mathbf{X} \mid Y\}} \right\}\right) \\
&= \mathbb{P}\left(\{\mathbf{X}^{(1)}, Y^{(1)}\}, \{\mathbf{X}^{(2)}, Y^{(2)}\}, \dots, \{\mathbf{X}^{(N)}, Y^{(N)}\} ; \left\{\boldsymbol{\pi}, \boldsymbol{\theta}_{\{\mathbf{X} \mid Y\}} \right\}\right) \\
&= \prod_{n=1}^N  \mathbb{P}\left(Y^{(n)}=k ; \boldsymbol{\pi}\right) \mathbb{P}\left(\mathrm{X}^{(n)} \mid Y^{(n)} = k ; \boldsymbol{\theta}_{\{\mathbf{X} \mid Y\}} \right)  \\
&= \prod_{n=1}^N  \left\{\mathbb{P}\left(Y^{(n)}=k ; \boldsymbol{\pi}\right) \prod_{d=1}^D \mathbb{P}\left(X_d^{(n)} \mid Y^{(n)} = k ; \boldsymbol{\theta}_{\{\mathbf{X} \mid Y\}} \right) \right\}  \\
\end{align*}
$$

Then we can maximize $\prod_{n=1}^N  \mathbb{P}\left(Y^{(n)}=k ; \boldsymbol{\pi}\right)$ 
individually since the above can be decomposed (**CITE MURPHY**).


### Previous

We denote the (joint) probability distribution of the observed data $\mathcal{D}$ as 

$$
\mathbb{P}(\mathcal{D} ; \boldsymbol{\theta}) = \mathbb{P}(\mathbf{X})
$$

In this context, since we are estimating $\boldsymbol{\pi}$ but do not really know the parameters of $\mathrm{X}$ just yet, we can simplify the expression to just

$$
\mathbb{P}(\mathcal{D} ; \left(\boldsymbol{\theta}, \boldsymbol{\pi} \right))
$$

where $\boldsymbol{\pi} = \begin{bmatrix} \pi_1 & \pi_2 & \dots & \pi_K \end{bmatrix}$ and 
$\boldsymbol{\theta}$ is the parameter vector of $\mathrm{X}$.

Since each sample is **i.i.d.**, we can write the joint probability distribution as the product of the individual probabilities of each sample:

$$
\begin{align*}
\mathbb{P}(\mathcal{D} ; \left(\boldsymbol{\pi}, \boldsymbol{\theta} \right)) &= \prod_{n=1}^N \mathbb{P}(\mathrm{X}^{(n)}, Y^{(n)} ;  \left(\boldsymbol{\theta}, \boldsymbol{\pi} \right)) \\
&= \prod_{n=1}^N \mathbb{P}(\mathrm{X}^{(n)} ; \boldsymbol{\theta}) \mathbb{P}(Y^{(n)} ; \boldsymbol{\pi})
\end{align*}
$$

There should be no confusion that both $\mathrm{X}$ and $\mathrm{Y}$ are included in the joint distribution,
since the dataset $\mathcal{D}$ is a joint distribution of $\mathrm{X}$ and $\mathrm{Y}$, and not just $\mathrm{X}$(?) (Verify this.)

Now, we are only interested in the term that depends on $\boldsymbol{\pi}$, so we can drop the term that depends on $\boldsymbol{\theta}$:

$$
\begin{align*}
\mathbb{P}(\mathcal{D} ; \boldsymbol{\pi}) &= \prod_{n=1}^N \mathbb{P}(Y^{(n)} ; \boldsymbol{\pi}) \\
\end{align*}
$$

The **likelihood function** is defined as

$$
\begin{align*}
\mathcal{L}(\left \{ \boldsymbol{\theta}, \boldsymbol{\pi} \right \} ; \mathcal{D}) &\overset{\mathrm{def}}{=} \mathbb{P}(\mathcal{D} ; \left(\boldsymbol{\theta}, \boldsymbol{\pi} \right)) \\
&= \prod_{n=1}^N \mathbb{P}(\mathrm{X}^{(n)}, Y^{(n)} ;  \left(\boldsymbol{\theta}, \boldsymbol{\pi} \right)) \\
\end{align*}
$$

but since we are only interested in the term that depends on $\boldsymbol{\pi}$, our likelihood function is

$$
\begin{align*}
\mathcal{L}(\boldsymbol{\pi} ; \mathcal{D}) &\overset{\mathrm{def}}{=} \mathbb{P}(\mathcal{D} ; \boldsymbol{\pi}) \\
&= \prod_{n=1}^N \mathbb{P}(Y^{(n)} ; \boldsymbol{\pi}) \\
&\overset{\mathrm{(a)}}{=} \prod_{n=1}^N \left(\prod_{k=1}^K \pi_k^{y^{(n)}_k} \right) \\
\end{align*}
$$

where $\left(\prod_{k=1}^K \pi_k^{y^{(n)}_k} \right)$ in equation $(a)$ is a consequence
of the definition of the Category distribution.

Subsequently, we can take the log of the likelihood function to get the **log-likelihood function** (for the ease of computation):

$$
\begin{align*}
\mathcal{L}(\boldsymbol{\pi} ; \mathcal{D}) &\overset{\mathrm{def}}{=} \log \mathcal{L}(\boldsymbol{\pi} ; \mathcal{D}) \\
&= \sum_{n=1}^N \log \left(\prod_{k=1}^K \pi_k^{y^{(n)}_k} \right) \\
&\overset{(b)}{=} \sum_{n=1}^N \sum_{k=1}^K y^{(n)}_k \log \pi_k \\
&\overset{(c)}{=} \sum_{k=1}^K N_k \log \pi_k \\
\end{align*}
$$

where $N_k$ is the number of samples that belong to the $k$-th category.

```{prf:remark} Notation Overload
:label: notation-overload

We note to ourselves that we are reusing, and hence abusing the notation $\mathcal{L}$ for the log-likelihood function to be the same as the likelihood function, this is just for the ease of re-defining a new symbol for the log-likelihood function, $\log \mathcal{L}$.
```

Equation $(c)$ is derived by expanding equation $(b)$,

$$
\begin{align*}
\sum_{n=1}^N \sum_{k=1}^K y^{(n)}_k \log \pi_k &= \sum_{n=1}^N \left( \sum_{k=1}^K y^{(n)}_k \log \pi_k \right) \\
&= y^{(1)}_1 \log \pi_1 + y^{(1)}_2 \log \pi_2 + \dots + y^{(1)}_K \log \pi_K \\
&+ y^{(2)}_1 \log \pi_1 + y^{(2)}_2 \log \pi_2 + \dots + y^{(2)}_K \log \pi_K \\
&+ \qquad \vdots \qquad \\
&+ y^{(N)}_1 \log \pi_1 + y^{(N)}_2 \log \pi_2 + \dots + y^{(N)}_K \log \pi_K \\
&\overset{(d)}{=} \left( y^{(1)}_1 + y^{(2)}_1 + \dots + y^{(N)}_1 \right) \log \pi_1 \\
&+ \left( y^{(1)}_2 + y^{(2)}_2 + \dots + y^{(N)}_2 \right) \log \pi_2 \\
&+ \qquad \vdots \qquad \\
&+ \left( y^{(1)}_K + y^{(2)}_K + \dots + y^{(N)}_K \right) \log \pi_K \\
&\overset{(e)}{=} N_1 \log \pi_1 + N_2 \log \pi_2 + \dots + N_K \log \pi_K \\
&= \sum_{k=1}^K N_k \log \pi_k \\
\end{align*}
$$

where $(d)$ is derived by summing each column, and $N_k = y^{(1)}_k + y^{(2)}_k + \dots + y^{(N)}_k$
is nothing but the number of samples that belong to the $k$-th category. One just need to recall that
if we have say 6 samples of class $(0, 1, 2, 0, 1, 1)$ where $K=3$, then the one-hot encoded
representation of the samples will be 

$$
\begin{align*}
\left[
\begin{array}{ccc}
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & 1 \\
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & 1 & 0 \\
\end{array}
\right]
\end{align*}
$$

and summing each column will give us $N_1 = 2$, $N_2 = 3$, and $N_3 = 1$.


## Maximum Likelihood Estimation of Categorical Distribution for Target Variable


aaa

## References



[^categorical-distribution]: [Category Distribution](https://en.wikipedia.org/wiki/Categorical_distribution)
[^joint-distribution]: [Joint Probability Distribution](https://en.wikipedia.org/wiki/Joint_probability_distribution#Discrete_case)
[^chain-rule-of-probability]: [Chain Rule of Probability](https://en.wikipedia.org/wiki/Chain_rule_(probability))
[^conditional-independence]: [Conditional Independence](https://en.wikipedia.org/wiki/Conditional_independence)