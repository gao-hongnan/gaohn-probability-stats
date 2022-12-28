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

### Discriminative vs Generative

- We will add a subscript $n$ to denote the $n$-th sample from the dataset. It still is referring to a single sample.
- Discriminative classifiers model the conditional distribution $\mathbb{P}(Y_n = k \mid X_n =  \mathrm{x}_n)$. This means we are modelling the conditional distribution of the target $Y_n$ given the input $\mathrm{x}_n$.
- Generative classifiers model the conditional distribution $\mathbb{P}(X_n = \mathrm{x}_n \mid Y_n = k)$. This means we are modelling the conditional distribution of the input $\mathrm{x}_n$ given the target $Y_n$. Then we can use Bayes' rule to compute the conditional distribution of the target $Y_n$ given the input $\mathrm{x}_n$.
- Both the target $Y_n$ and the input $X_n$ are random variables in the generative model. In the discriminative model, only the target $Y_n$ is a random variable as the input $X_n$ is fixed (we do not need to estimate anything about the input $X$). 
  
## Naive Bayes Setup

Let 

$$
\mathcal{D} = \left \{ \left(\mathrm{X}^{(n)}, Y^{(n)} \right) \right \}_{n=1}^N = \left \{ \left(\mathrm{x}^{(n)}, y^{(n)} \right) \right \}_{n=1}^N
$$

be the dataset with $N$ samples and $D$ predictors.

All samples are assumed to be **independent and identically distributed (i.i.d.)** from the unknown but fixed joint distribution 
$\mathbb{P}(\mathcal{X}, \mathcal{Y} ; \boldsymbol{\theta})$,

$$
\left \{ \left(\mathrm{X}^{(n)}, Y^{(n)} \right) \right \} \overset{\small{\text{i.i.d.}}}{\sim} \mathbb{P}(\mathcal{X}, \mathcal{Y} ; \boldsymbol{\theta}) \quad \text{for } n = 1, 2, \cdots, N
$$

where $\boldsymbol{\theta}$ is the parameter vector of the joint distribution. See {prf:ref}`joint-distribution-example` for an example of such.

(naive-bayes-inference-prediction)=
## Inference/Prediction

Before we look at the fitting/estimating process, let's look at the inference/prediction process.

Suppose the problem at hand has $K$ classes, $k = 1, 2, \cdots, K$, where $k$ is the index of the class.

Then, to find the class of a new test sample $\mathbf{x}^{q} \in \mathbb{R}^{D}$ with $D$ features,
we can compute the conditional probability of each class $Y = k$ given the sample $\mathbf{x}^{q}$:

```{prf:algorithm} Naive Bayes Inference Algorithm
:label: naive-bayes-inference-algorithm

1. Compute the conditional probability of each class $Y = k$ given the sample $\mathbf{x}^{q}$:

    $$
    \mathbb{P}(Y = k \mid \mathbf{X} = \mathbf{x}^{q}) = \dfrac{\mathbb{P}(\mathbf{X} = \mathbf{x}^{q} \mid Y = k) \mathbb{P}(Y = k)}{\mathbb{P}(\mathbf{X} = \mathbf{x}^{q})} \quad \text{for } k = 1, 2, \cdots, K
    $$ (eq:conditional-naive-bayes)

2. Choose the class $k$ that maximizes the conditional probability:

    $$
    \hat{y}^{(q)} = \arg\max_{k=1}^K \mathbb{P}(Y = k \mid \mathbf{X} = \mathbf{x}^{q})
    $$ (eq:argmax-naive-bayes-1)

The observant reader would have noticed that the normalizing constant $\mathbb{P}(X = \mathbf{x}^{(q)})$ is the same for all $k$.
Therefore, we can ignore it and simply choose the class $k$ that maximizes the numerator of the conditional probability. 

$$
\hat{y}_q = \arg\max_{k=1}^K \mathbb{P}(X = \mathbf{x}^{q} \mid Y = k) \mathbb{P}(Y = k)
$$ (eq:argmax-naive-bayes-2)

since 

$$
\mathbb{P}(Y = k \mid \mathbf{X} = \mathbf{x}^{q}) \propto \mathbb{P}(\mathbf{X} = \mathbf{x}^{q} \mid Y = k) \mathbb{P}(Y = k)
$$

by a constant factor $\mathbb{P}(\mathbf{X} = \mathbf{x}^{q})$.
```

Now if we just proceed to estimate the conditional probability $\mathbb{P}(Y = k \mid \mathbf{X} = \mathbf{x}^{q})$, we will need to estimate the joint probability $\mathbb{P}(X = \mathbf{x}^{q}, Y = k)$, which is intractable[^intractable]. 

However, if we can ***estimate*** the conditional probability (likelihood) $\mathbb{P}(\mathbf{X} = \mathbf{x}^{q} \mid Y = k)$ 
and the prior probability $\mathbb{P}(Y = k)$, then we can use Bayes' rule to 
compute the posterior conditional probability $\mathbb{P}(Y = k \mid \mathbf{X} = \mathbf{x}^{q})$. 


## The Naive Bayes Assumptions

In this section, we talk about some implicit and explicit assumptions of the Naive Bayes model.



### Independent and Identically Distributed (i.i.d.)

In supervised learning, implicitly or explicitly, one *always* assumes that the training set

$$
\begin{aligned}
\mathcal{D} &= \left\{\left(\mathbf{x}^{(1)}, y^{(1)}\right), \left(\mathbf{x}^{(2)}, y^{(2)}\right), \cdots, \left(\mathbf{x}^{(N)}, y^{(N)}\right)\right\} \\
\end{aligned}
$$ 

is composed of $N$ input/response tuples 

$$
\left({\mathbf{X}}^{(n)} = \mathbf{x}^{(n)}, Y^{(n)} = y^{(n)}\right)
$$

that are ***independently drawn from the same (identical) joint distribution*** 

$$
\mathbb{P}_{\{\mathcal{X}, \mathcal{Y} ; \boldsymbol{\theta}\}}(\mathbf{x}, y)
$$

with

$$
\mathbb{P}(\mathbf{X} = \mathbf{x}, Y = y ; \boldsymbol{\theta}) = \mathbb{P}(Y = y \mid \mathbf{X} = \mathbf{x}) \mathbb{P}(\mathbf{X} = \mathbf{x})
$$

where $\mathbb{P}(Y = y \mid \mathbf{X} = \mathbf{x})$ is the conditional probability of $Y$ given $\mathbf{X}$,
the relationship that the learner algorithm/concept $c$ is trying to capture.

```{prf:definition} The i.i.d. Assumption
:label: iid-assumption

Mathematically, this i.i.d. assumption writes (also defined in {prf:ref}`def_iid`):

$$
\begin{aligned}
\left({\mathbf{X}}^{(n)}, Y^{(n)}\right) &\sim \mathbb{P}_{\{\mathcal{X}, \mathcal{Y}, \boldsymbol{\theta}\}}(\mathbf{x}, y) \quad \text{and}\\
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

### Conditional Independence

The core assumption of the Naive Bayes model is that the predictors $\mathcal{X}$ are conditionally independent given the class label $Y$.

But how did we arrive at the conditional independence assumption? Let's look at what we wanted to achieve in the first place.

Recall that our goal in {ref}`naive-bayes-inference-prediction` is to find the class $k \in \{1, 2, \cdots, K\}$ that maximizes the posterior probability
$\mathbb{P}(Y = k \mid \mathbf{X} = \mathbf{x}^{(q)} ; \boldsymbol{\theta})$. 

$$
\begin{aligned}
\arg \max_{k} \mathbb{P}(Y = k \mid \mathbf{X} = \mathbf{x}^{(q)} ; \boldsymbol{\theta}) &= \arg \max_{k} \frac{\mathbb{P}(Y = k, \mathbf{X} = \mathbf{x}^{(q)} ; \boldsymbol{\theta})}{\mathbb{P}(\mathbf{X} = \mathbf{x}^{(q)} ; \boldsymbol{\theta})} \\
&= \arg \max_{k} \frac{\mathbb{P}(Y = k ; \boldsymbol{\pi}) \mathbb{P}(\mathbf{X} \mid Y = k ; \boldsymbol{\theta}_{\{\mathbf{X} \mid Y\}})}{\mathbb{P}(\mathbf{X} = \mathbf{x}^{(q)} ; \boldsymbol{\theta})}\\
&\propto \arg \max_{k} \mathbb{P}(Y = k ; \boldsymbol{\pi}) \mathbb{P}\left(\mathbf{X} \mid Y = k ; \boldsymbol{\theta}_{\{\mathbf{X} \mid Y\}}\right)
\end{aligned}
$$ (eq:argmax-naive-bayes-3)

We have seen earlier in {prf:ref}`naive-bayes-inference-algorithm` that since the denominator 
is constant for all $k$, we can ignore it and just maximize the numerator, as shown by the proportional sign.

This suggests we need to find estimates for both the prior and the likelihood. This of course 
involves ur finding the $\boldsymbol{\pi}$ and $\boldsymbol{\theta}_{\{\mathbf{X} \mid Y\}}$ that maximize the likelihood function, which we will talk about later.

In order to meaningfully optimize the expression, we need to decompose the numerator {eq}`eq:argmax-naive-bayes-3`
into its components that contain the parameters we want to estimate.

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
&= \mathbb{P}(Y) \mathbb{P}(X_1 \mid Y) \mathbb{P}(X_2 \mid Y, X_1) \mathbb{P}(X_3 \mid Y, X_1, X_2) \cdots \mathbb{P}(X_D \mid Y, X_1, X_2, \ldots X_{D-1}) \\
&= \mathbb{P}(Y) \prod_{d=1}^D \mathbb{P}(X_d \mid Y, X_1, X_2, \ldots X_{d-1})
\end{aligned}
$$ (eq:joint-distribution-decomposed)

This alone does not get us any further, we still need to estimate roughly $2^{D}$ parameters[^2dparameters], 
which is computationally expensive. Not to forget that we need to estimate for each class $k \in \{1, 2, 3, \ldots, K\}$
which has a complexity of $\sim \mathcal{O}(2^DK)$.

```{prf:remark} Why $2^D$ parameters?
:label: 2dparameters

Let's simplify the problem by assuming each feature $X_d$ and the class label $Y$ are binary random variables, 
i.e. $X_d \in \{0, 1\}$ and $Y \in \{0, 1\}$.

Then $\mathbb{P}(Y, X_1, X_2, \ldots X_D)$ is a joint distribution of $D+1$ random variables, each with $2$ values.

This means the sample space of $\mathbb{P}(Y, X_1, X_2, \ldots X_D)$ is 

$$
\begin{aligned}
\mathcal{S} &= \{(0, 1)\} \times \{(0, 1)\} \times \{(0, 1)\} \times \cdots \times \{(0, 1)\} \\
&= \{(0, 0, 0, \ldots, 0), (0, 0, 0, \ldots, 1), (0, 0, 1, \ldots, 0), \ldots, (1, 1, 1, \ldots, 1)\}
\end{aligned}
$$

which has $2^{D+1}$ elements. 
To really get the exact joint distribution, we need to estimate the probability of each element in the sample space, which is $2^{D+1}$ parameters.

This has two caveats:

1. There are too many parameters to estimate, which is computationally expensive. Imagine if $D$ is 1000, we need to estimate $2^{1000}$ parameters, which is infeasible.
2. Even if we can estimate all the parameters, we are essentially overfitting the data by memorizing the training data. There is no learning involved.
```

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
class conditional density for class $k$ and feature $d$[^kdparameters]. Furthermore,
the boldsymbol $\boldsymbol{\theta}$ is the parameter vector,

$$
\boldsymbol{\theta} = \left(\boldsymbol{\pi}, \{\theta_{dk}\}_{k=1}^K, \{d=1, \ldots, D\}\right) = \left(\boldsymbol{\pi}, \boldsymbol{\theta}_{\{\mathbf{X} \mid Y\}}\right)
$$


```{prf:definition} The Parameter Vector
:label: parameter-vector

There is not much to say about the categorical component $\boldsymbol{\pi}$, since we are
just estimating the prior probabilities of the classes. 

$$
\boldsymbol{\pi} = \begin{bmatrix} \pi_1 & \pi_2 & \dots & \pi_K \end{bmatrix}
$$

The parameter vector (matrix) $\boldsymbol{\theta}_{\{\mathbf{X} \mid Y\}}=\{\theta_{dk}\}_{k=1}^K, \{d=1, \ldots, D\}$ is a bit more complicated.
It resides in the $\mathbb{R}^{K \times D}$ space, where each element $\theta_{dk}$ is the parameter
associated with feature $d$ conditioned on class $k$.

$$
\boldsymbol{\theta}_{\{\mathbf{X} \mid Y\}} = \begin{bmatrix} 
\theta_{11} & \theta_{12} & \dots & \theta_{1D} \\
\theta_{21} & \theta_{22} & \dots & \theta_{2D} \\
\vdots & \vdots & \ddots & \vdots \\
\theta_{K1} & \theta_{K2} & \dots & \theta_{KD}
\end{bmatrix}_{K \times D}
$$

So if $K=3$ and $D=2$, then the parameter vector $\boldsymbol{\theta}$ is a $3 \times 2$ matrix, i.e.

$$
\boldsymbol{\theta}_{\{\mathbf{X} \mid Y\}} = \begin{bmatrix}
\theta_{11} & \theta_{12} \\
\theta_{21} & \theta_{22} \\
\theta_{31} & \theta_{32}
\end{bmatrix}_{3 \times 2}
$$

This means we have effectively reduced our complexity from $\sim \mathcal{O}(2^D)$ to $\sim \mathcal{O}(KD + 1)$.

One big misconception is that the elements in $\boldsymbol{\theta}_{\{\mathbf{X} \mid Y\}}$ are scalar values.
This is not true, for example, let's look at the first entry $\theta_{11}$, corresponding to
the parameter of class $K=1$ and feature $D=1$, i.e. $\theta_{11}$ is the parameter of the class conditional
density $\mathbb{P}(X_1 \mid Y = 1)$. Now $X_1$ can take on any value in $\mathbb{R}$, which is indeed a scalar,
we further assume that $X_1$ takes on a univariate Gaussian distribution, then $\theta_{11}$ is a vector of length 2, i.e.

$$
\theta_{11} = \begin{bmatrix} \mu_{11} & \sigma_{11} \end{bmatrix}
$$

where $\mu_{11}$ is the mean of the Gaussian distribution and $\sigma_{11}$ is the standard deviation of the Gaussian distribution. 
This is something we need to take note of.

**We have also reduced the problem of estimating the joint distribution to just individual conditional distributions.**

Overall, before this assumption, you can think of estimating the joint distribution of $Y$ and $\mathbf{X}$,
and after this assumption, you can simply individually estimate each conditional distribution.
```


```{prf:remark} Notation remark
A note, the notation $\boldsymbol{\theta}_{dk}$ should either be read as $\boldsymbol{\theta}_{kd}$ since
we say $\boldsymbol{\theta}_{\{\mathbf{X} \mid Y\}}$ is a $K \times D$ matrix.

Consider bolding elements of $\boldsymbol{\theta}_{\{\mathbf{X} \mid Y\}}$ to indicate that it can be a vector.
```

### Inductive Bias

We still need to introduce some inductive bias into {eq}`eq:naive-bayes-classifier-1`, more concretely, we need to make some assumptions about the distribution 
of $\mathbb{P}(Y)$ and $\mathbb{P}(X_d \mid Y)$. 

For the target variable, we typically model it as a categorical distribution, 

$$
\mathbb{P}(Y) \sim \mathrm{Categorical}(\boldsymbol{\pi})
$$

For the conditional distribution of the features, we typically model it according to what type of features we have.

For example, if we have binary features, then we can model it as a Bernoulli distribution, 

$$
\mathbb{P}(X_d \mid Y) \sim \mathrm{Bernoulli}(\theta_{dk})
$$

If we have categorical features, then we can model it as a multinomial/catgorical distribution,

$$
\mathbb{P}(X_d \mid Y) \sim \mathrm{Multinomial}(\boldsymbol{\theta}_{dk})
$$

If we have continuous features, then we can model it as a Gaussian distribution,

$$
\mathbb{P}(X_d \mid Y) \sim \mathcal{N}(\mu_{dk}, \sigma_{dk}^2)
$$

To reiterate, we want to make some inductive bias assumptions of $\mathbf{X}$ conditional on $Y$,
as well as with $Y$. Note very carefully that we are not talking about the marginal distribution of
$\mathbf{X}$ here, instead, we are talking about the conditional distribution of $\mathbf{X}$ given $Y$. The distinction is subtle, but important.

#### Discrete Features/Targets (Categorical Distribution)

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

In the case where (all) the features $X_d$ are categorical ($D$ number of features), i.e. $X_d \in \{1, 2, \cdots, C\}$, 
we can use the categorical distribution to model the ($D$-dimensional) conditional distribution of $\mathbf{X} \in \mathbb{R}^{D}$ given $Y = k$. 

$$
\begin{align*}
\mathbf{X} \mid Y = k &\overset{\small{\text{i.i.d.}}}{\sim} \text{Category}(\boldsymbol{\pi}_{\{\mathbf{X} \mid Y\}}) \quad \text{where } \boldsymbol{\pi}_{\{\mathbf{X} \mid Y\}} = \begin{bmatrix} \pi_{1, 1} & \dots & \pi_{1, D} \\ \vdots & \ddots & \vdots \\ \pi_{K, 1} & \dots & \pi_{K, D} \end{bmatrix}
\end{align*}
$$

where each entry $\pi_{k, d}$ is the probability distribution (PDF) of $X_d$ given $Y = k$. 

$$
X_d \mid Y = k \overset{\small{\text{i.i.d.}}}{\sim} \text{Category}(\pi_{k, d})
$$

Furthermore,
each $\pi_{k, d}$ is **not a scalar** but a **vector of size $C$** holding the probability of $X_d = c$ given $Y = k$.

$$
\begin{align*}
\pi_{k, d} = \begin{bmatrix} \pi_{k, d, 1} & \dots & \pi_{k, d, C} \end{bmatrix}
\end{align*}
$$

Then the (chained) multi-dimensional conditional PDF of $\mathbf{X} = \begin{bmatrix} X_1 & \dots & X_D \end{bmatrix}$ given $Y = k$ is

$$
\begin{align*}
\mathbb{P}(\mathbf{X} = \mathbf{x} | Y = k ; \boldsymbol{\theta}_{\{\mathbf{X} \mid Y\}}) &= \prod_{d=1}^D \text{Categorical}(X_d \mid Y = k; \pi_{k, d}) \\
&= \prod_{d=1}^D \prod_{c=1}^C \pi_{k, d, c}^{x_{c, d}} \quad \text{for } c = 1, 2, \cdots, C \text{ and } k = 1, 2, \cdots, K
\end{align*}
$$

As an example, if $C=3$, $D=2$ and $K=4$, then the $\boldsymbol{\theta}_{\{\mathbf{X} \mid Y\}}$ is a $K \times D = 4 \times 2$ matrix, but for
each entry $\pi_{k, d}$, is a $1 \times C$ vector. If one really wants, we can also represent this as a 
$4 \times 2 \times 3$ tensor, especially in the case of implementing it in code.

To be more verbose, when we find

$$
\mathbf{X} \mid Y = k \overset{\small{\text{i.i.d.}}}{\sim} \text{Category}(\boldsymbol{\pi}_{\{\mathbf{X} \mid Y\}})
$$

we are actually finding for all $k = 1, 2, \cdots, K$,

$$
\begin{align*}
\mathbf{X} \mid Y &= 1 &\overset{\small{\text{i.i.d.}}}{\sim} \text{Category}(\boldsymbol{\pi}_{\{\mathbf{X} \mid Y\}}) \\
\mathbf{X} \mid Y &= 2 &\overset{\small{\text{i.i.d.}}}{\sim} \text{Category}(\boldsymbol{\pi}_{\{\mathbf{X} \mid Y\}}) \\
\vdots \\
\mathbf{X} \mid Y &= K &\overset{\small{\text{i.i.d.}}}{\sim} \text{Category}(\boldsymbol{\pi}_{\{\mathbf{X} \mid Y\}})
\end{align*}
$$


#### Continuous Features (Gaussian Distribution)

If all features $X_d$ are continuous, we can use the Gaussian distribution to model the conditional distribution of $\mathbf{X}$ given $Y = k$.



---

See **KEVIN MURPHY pp 358 for more details**.

For simplicity sake, we assume that all features $X_d$ are of the same type, either all binary or all continuous.
In reality, this may not need to be the case.

This section's content is adapted from [Machine Learning from Scratch](https://dafriedman97.github.io/mlbook/content/c4/concept.html).



In general, we assume all samples $X_n$ come from the same *family* of distributions. This means that for samples $n = 1$ to $n = N$, and class $k = 1$ to $k = K$, we have the following:

$$
\begin{align*}
\mathrm{X}_n|(Y_n = 1) &\sim \text{Distribution}(\boldsymbol{\theta}_1) \\
\mathrm{X}_n|(Y_n = 2) &\sim \text{Distribution}(\boldsymbol{\theta}_2) \\
\vdots & \quad \vdots \\
\mathrm{X}_n|(Y_n = K) &\sim \text{Distribution}(\boldsymbol{\theta}_K)
\end{align*}
$$

where $\boldsymbol{\theta}_k$ is the parameter vector of $\mathrm{X}_n$ conditioned on the $k$-th class. For instance, if we are using a Multivariate Gaussian distribution, then $\boldsymbol{\theta}_k = \begin{bmatrix} \boldsymbol{\mu_k} & \boldsymbol{\Sigma_k} \end{bmatrix}$.

However, it is possible for the individual variables within the random vector $\mathrm{X}_n$ to follow different distributions. For instance, if $\mathrm{X}_n = \begin{bmatrix} X_{n1} & X_{n2} \end{bmatrix}^\top$, we might have

$$
\begin{align*}
X_{n1}|(Y_n = k) &\sim \text{Binomial}(n, p_{k}) \\
X_{n2}|(Y_n = k) &\sim \mathcal{N}(\boldsymbol{\mu_k}, \boldsymbol{\Sigma_k})
\end{align*}
$$
 
The machine learning fitting process is then to estimate the parameters of these distributions. More concretely, we need to estimate $\boldsymbol{\pi}_k$ for $k = 1, \dots, K$, as well as $\boldsymbol{\theta}_k$ for $k = 1, \dots, K$ for what might be the possible distributions of $\mathrm{X}_n \mid Y_n$. In this example above, we would need to estimate $p_k$, $\boldsymbol{\mu}_k$ and $\boldsymbol{\Sigma}_k$ for $k = 1, \dots, K$ for the Binomial and Multivariate Gaussian distributions.

Once that's done, we can estimate $\mathbb{P}(Y_n = k)$ and $\mathbb{P}(\mathrm{X}_n \mid Y_n = k)$ for $k = 1, \dots, K$. We can then use these estimates to make predictions about the class of a new sample $\mathrm{X}_n$ using Bayes' rule in equation {eq}`eq:argmax-naive-bayes-2`.

## Model Fitting

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

Then we can maximize 

$$
\prod_{n=1}^N  \mathbb{P}\left(Y^{(n)}=k ; \boldsymbol{\pi}\right)
$$

and

$$
\prod_{n=1}^N  \prod_{d=1}^D \mathbb{P}\left(X_d^{(n)} \mid Y^{(n)} = k ; \boldsymbol{\theta}_{\{\mathbf{X} \mid Y\}} \right)
$$

individually since the above can be decomposed (**CITE MURPHY**).

### Estimating Priors

Before we start the formal estimation process, it is intuitive to think that the prior probabilities $\boldsymbol{\pi}_k$ should be proportional to the number of samples in each class. In other words, if we have $N_1$ samples in class 1, $N_2$ samples in class 2, and so on, then we should have

$$
\begin{align*}
\pi_1 &\propto N_1 \\
\pi_2 &\propto N_2 \\
\vdots & \quad \vdots \\
\pi_K &\propto N_K
\end{align*}
$$

For instance, if we have a dataset with $N=100$ samples with $K=3$ classes, and $N_1 = 10$, $N_2 = 30$ and $N_3 = 60$, then we should have $\pi_1 = \frac{10}{100} = 0.1$, $\pi_2 = \frac{30}{100} = 0.3$ and $\pi_3 = \frac{60}{100} = 0.6$. This is just the relative frequency of each class.

It turns out our intuition matches the formal estimation process. 

### Maximum Likelihood Estimation for Priors (Categorical Distribution)

Let $\mathcal{D} = \left \{ \left(\mathrm{X}^{(n)}, Y^{(n)} \right) \right \}_{n=1}^N = \left \{ \left(\mathrm{x}^{(n)}, y^{(n)} \right) \right \}_{n=1}^N$ be the dataset
with $N$ samples and $D$ predictors. All samples are assumed to be **independent and identically distributed (i.i.d.)** from the unknown but fixed joint distribution 
$\mathbb{D} = \mathbb{P}(\mathcal{X}, \mathcal{Y} ; \boldsymbol{\theta})$ where $\boldsymbol{\theta}$ is the parameter vector of the joint distribution. In other words, $\mathrm{X}^{(1)}, \mathrm{X}^{(2)}, \dots, \mathrm{X}^{(N)}$ are all i.i.d. from $\mathbb{D}$, as well as $\mathrm{Y}^{(1)}, \mathrm{Y}^{(2)}, \dots, \mathrm{Y}^{(N)}$.

We denote the (joint) probability distribution of the observed data $\mathcal{D}$ as $\mathbb{P}(\mathcal{D} ; \boldsymbol{\theta})$.

For example, if $D = 2$, and $X^{(n)}_1$ and $X^{(n)}_2$ are both multivariate Gaussian random variables, 
with $\boldsymbol{\mu}_1$ and $\boldsymbol{\mu}_2$ being the mean vectors of the two distributions,
and $\boldsymbol{\Sigma}_1$ and $\boldsymbol{\Sigma}_2$ being the covariance matrices of the two distributions;
furthermore, $Y^{(n)}$ is a Bernoulli random variable with parameter $\pi$, then we have $\boldsymbol{\theta} = \begin{bmatrix} \boldsymbol{\mu}_1 & \boldsymbol{\Sigma}_1 & \boldsymbol{\mu}_2 & \boldsymbol{\Sigma}_2 & \pi \end{bmatrix}$.

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

Now we are finally ready to solve the estimation (optimization) problem for $\boldsymbol{\pi}$.

$$
\begin{align*}
\hat{\boldsymbol{\pi}} &= \underset{\boldsymbol{\pi}}{\mathrm{argmax}} ~~ \mathcal{L}(\boldsymbol{\pi} ; \mathcal{D}) \\
&= \underset{\boldsymbol{\pi}}{\mathrm{argmax}} ~~ \sum_{k=1}^K N_k \log \pi_k \\
\end{align*}
$$

subject to the constraint that 

$$
\sum_{k=1}^K \pi_k = 1
$$

which is just saying the probabilities must sum up to 1.

We can also write the expression as

$$
\begin{aligned}
\max_{\boldsymbol{\pi}} &~~ \sum_{k=1}^K N_k \log \pi_k \\
\text{subject to} &~~ \sum_{k=1}^K \pi_k = 1
\end{aligned}
$$

This is a constrained optimization problem, and we can solve it using the Lagrangian method.

```{prf:definition} Lagrangian Method
:label: lagrangian-method

The Lagrangian method is a method to solve constrained optimization problems. The idea is to
convert the constrained optimization problem into an unconstrained optimization problem by
introducing a Lagrangian multiplier $\lambda$ and then solve the unconstrained optimization
problem. 

Given a function $f(\mathrm{x})$ and a constraint $g(\mathrm{x}) = 0$, the Lagrangian function, 
$\mathcal{L}(\mathrm{x}, \lambda)$ is defined as

$$
\begin{align*}
\mathcal{L}(\mathrm{x}, \lambda) &= f(\mathrm{x}) - \lambda g(\mathrm{x}) \\
\end{align*}
$$

where $\lambda$ is the Lagrangian multiplier and may be either positive or negative. Then, 
the critical points of the Lagrangian function are the same as the critical points of the
original constrained optimization problem, i.e. setting the gradient vector of the Lagrangian
function $\nabla \mathcal{L}(\mathrm{x}, \lambda) = 0$ with respect to $\mathrm{x}$ and $\lambda$.
```

One note is that the notation of $\mathcal{L}$ seems to be overloaded again with the Lagrangian function, we will have to change it to $\mathcal{L}_\lambda$ to avoid confusion. So, to reiterate, solving the Lagrangian function is equivalent to solving the constrained optimization problem.

In our problem, we can convert it to Lagrangian form as

$$
\begin{align*}
\hat{\boldsymbol{\pi}} &= \underset{\boldsymbol{\pi}}{\mathrm{argmax}} ~~ \mathcal{L}_\lambda(\boldsymbol{\pi}, \lambda ; \mathcal{D}) \\
&= \underset{\boldsymbol{\pi}}{\mathrm{argmax}} ~~ \underbrace{\mathcal{L}(\boldsymbol{\pi} ; \mathcal{D})}_{f(\boldsymbol{\pi})} - \lambda \left(\underbrace{\sum_{k=1}^K \pi_k - 1}_{g(\boldsymbol{\pi})} \right) \\
&= \underset{\boldsymbol{\pi}}{\mathrm{argmax}} ~~ \sum_{k=1}^K N_k \log \pi_k - \lambda \left( \sum_{k=1}^K \pi_k - 1 \right) \\
\end{align*}
$$

which is now an unconstrained optimization problem. We can now solve it by setting the gradient vector of the Lagrangian function $\nabla \mathcal{L}_\lambda(\boldsymbol{\pi}, \lambda ; \mathcal{D}) = 0$ with respect to $\boldsymbol{\pi}$ and $\lambda$, as follows,

$$
\begin{align*}
\nabla \mathcal{L}_\lambda(\boldsymbol{\pi}, \lambda ; \mathcal{D}) &\overset{\mathrm{def}}{=} \frac{\partial \mathcal{L}_\lambda(\boldsymbol{\pi}, \lambda ; \mathcal{D})}{\partial \boldsymbol{\pi}} = 0 \quad \text{and} \quad \frac{\partial \mathcal{L}_\lambda(\boldsymbol{\pi}, \lambda ; \mathcal{D})}{\partial \lambda} = 0 \\
&\iff \frac{\partial}{\partial \boldsymbol{\pi}} \left( \sum_{k=1}^K N_k \log \pi_k - \lambda \left( \sum_{k=1}^K \pi_k - 1 \right) \right) = 0 \quad \text{and} \quad \frac{\partial \mathcal{L}_\lambda(\boldsymbol{\pi}, \lambda ; \mathcal{D})}{\partial \lambda} = 0 \\
\\
&\iff \begin{bmatrix} \frac{\partial \mathcal{L}}{\partial \pi_1} \\ \vdots \\ \frac{\partial \mathcal{L}}{\partial \pi_K} \\ \frac{\partial \mathcal{L}}{\partial \lambda} \end{bmatrix} = \begin{bmatrix} 0 \\ \vdots \\ 0 \\ 0 \end{bmatrix} \\
&\iff \begin{bmatrix} \frac{\partial}{\partial \pi_1} \left( N_1 \log \pi_1 - \lambda \left( \pi_1 - 1 \right) \right) \\ \vdots \\ \frac{\partial}{\partial \pi_K} \left( N_K \log \pi_K - \lambda \left( \pi_K - 1 \right) \right) \\ \frac{\partial \mathcal{L}_\lambda(\boldsymbol{\pi}, \lambda ; \mathcal{D})}{\partial \lambda} \end{bmatrix} = \begin{bmatrix} 0 \\ \vdots \\ 0 \\ 0 \end{bmatrix} \\
&\iff \begin{bmatrix} \frac{N_1}{\pi_1} - \lambda \\ \vdots \\ \frac{N_K}{\pi_K} - \lambda \\ \sum_{k=1}^K \pi_k - 1 \end{bmatrix} = \begin{bmatrix} 0 \\ \vdots \\ 0 \\ 0 \end{bmatrix} \\
\end{align*}
$$

The reason we can unpack $\frac{\partial}{\partial \pi_k}\left( \sum_{k=1}^K N_k \log \pi_k - \lambda \left( \sum_{k=1}^K \pi_k - 1 \right) \right)$ as $\frac{\partial}{\partial \pi_k} \left( N_k \log \pi_k - \lambda \left( \pi_k - 1 \right) \right)$ is because we are dealing with partial derivatives, so other terms other than $\pi_k$ are constant. 

Finally, we have a system of equations for each $\pi_k$ and if we can solve for $\pi_k$ for each $k$, we can then find the best estimate of $\boldsymbol{\pi}$. It turns out we have to find $\lambda$ first, and this can be solved by setting $\sum_{k=1}^K \pi_k - 1 = 0$ and solving for $\lambda$, which is the last equation in the system of equations above. We first express each $\pi_k$ in terms of $\lambda$,

$$
\begin{align*}
\frac{N_1}{\pi_1} - \lambda &= 0 \implies \pi_1 = \frac{N_1}{\lambda} \\
\frac{N_2}{\pi_2} - \lambda &= 0 \implies \pi_2 = \frac{N_2}{\lambda} \\
\vdots \\
\frac{N_K}{\pi_K} - \lambda &= 0 \implies \pi_K = \frac{N_K}{\lambda} \\
\end{align*}
$$

Then we substitute these expressions into the last equation in the system of equations above, and solve for $\lambda$,

$$
\begin{align*}
\sum_{k=1}^K \pi_k - 1 = 0 &\implies \sum_{k=1}^K \frac{N_k}{\lambda} - 1 = 0 \\
&\implies \sum_{k=1}^K \frac{N_k}{\lambda} = 1 \\
&\implies \sum_{k=1}^K N_k = \lambda \\
&\implies \lambda = \sum_{k=1}^K N_k \\
&\implies \lambda = N \\
\end{align*}
$$

and therefore, we can now solve for $\pi_k$,

$$
\boldsymbol{\hat{\pi}} = \left .
  \begin{cases}
    \pi_1 = \frac{N_1}{N} \\
    \pi_2 = \frac{N_2}{N} \\
    \vdots \quad \vdots \quad \vdots \\
    \pi_K = \frac{N_K}{N} \\
  \end{cases}
  \right\} \implies \pi_k = \frac{N_k}{N} \quad \text{for} \quad k = 1, 2, \ldots, K
$$

We conclude that the maximum likelihood estimate of $\boldsymbol{\pi}$ is $\boldsymbol{\pi} = \left( \frac{N_1}{N}, \frac{N_2}{N}, \ldots, \frac{N_K}{N} \right)$, which is the same as the empirical relative frequency of each class in the training data. This coincides with our intuition.

For completeness of expression,

$$
\begin{align*}
\hat{\boldsymbol{\pi}} &= \arg \max_{\boldsymbol{\pi}} \mathcal{L}\left( \boldsymbol{\pi} ; \mathcal{D} \right) \\
&= \begin{bmatrix} \hat{\pi}_1 \\ \vdots \\ \hat{\pi}_K \end{bmatrix} \\
&= \begin{bmatrix} \frac{N_1}{N} \\ \vdots \\ \frac{N_K}{N} \end{bmatrix}
\end{align*}
$$










## References

[^2dparameters]: Dive into Deep Learning, Section 22.9, this is only assuming that each feature $\mathbf{x}_d^{(n)}$ is binary, i.e. $\mathbf{x}_d^{(n)} \in \{0, 1\}$.
[^intractable]: Cite Dive into Deep Learning on this. Also, the joint probability is intractable because the number of parameters to estimate is exponential in the number of features. Use binary bits example, see my notes.
[^categorical-distribution]: [Category Distribution](https://en.wikipedia.org/wiki/Categorical_distribution)
[^joint-distribution]: [Joint Probability Distribution](https://en.wikipedia.org/wiki/Joint_probability_distribution#Discrete_case)
[^chain-rule-of-probability]: [Chain Rule of Probability](https://en.wikipedia.org/wiki/Chain_rule_(probability))
[^conditional-independence]: [Conditional Independence](https://en.wikipedia.org/wiki/Conditional_independence)
[^kdparameters]: Probablistic Machine Learning: An Introduction, Section 9.3, pp 328