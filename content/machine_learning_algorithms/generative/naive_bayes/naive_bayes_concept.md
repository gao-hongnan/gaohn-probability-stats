# Naive Bayes Concept

## Notations

```{prf:definition} Notations
:label: machine-learning-notations

- $\mathcal{X}$: Input space consists of all possible inputs $\mathbf{x} \in \mathcal{X}$.
  
- $\mathcal{Y}$: Label space = $\{1, 2, \cdots, K\}$ where $K$ is the number of classes.
  
- The mapping between $\mathcal{X}$ and $\mathcal{Y}$ is given by $c: \mathcal{X} \rightarrow \mathcal{Y}$ where $c$ is called *concept* according to the PAC learning theory.
  
- $\mathcal{D}$: The fixed but unknown distribution of the data.

Now, consider a dataset $\mathcal{D}_{\{\mathrm{x}, y\}}$ consisting of $N$ samples (observations) and $D$ predictors (features) drawn **jointly** and **indepedently and identically distributed** (i.i.d.) from $\mathcal{D}$. Note we will refer to the dataset with the same notation as the underlying distribution $\mathcal{D}$ from now on. 
  
- $\mathcal{D}$ denotes the training dataset can also be represented compactly as a set:
  
    $$
    \begin{align*}
    \mathcal{D}_{\{\mathrm{x}, y\}} &= \left\{\mathrm{x}^{(n)}, y^{(n)}\right\}_{n=1}^N \\
    &= \left\{(\mathrm{x}_1, y_1), (\mathrm{x}_2, y_2), \cdots, (\mathrm{x}_N, y_N)\right\} \\
    \end{align*}
    $$

- Then $\mathrm{x} = \begin{bmatrix} x_1 & x_2 & \cdots & x_D \end{bmatrix}$ is a sample of size $D$, drawn (jointly) $\textbf{i.i.d.}$ from $\mathcal{D}$. 
  - We often subscript $\mathrm{x}$ with $n$ to denote the $n$-th sample from the dataset, i.e. $\mathrm{x}_n = \begin{bmatrix} x_{n1} & x_{n2} & \cdots & x_{nD} \end{bmatrix}$.
  - We often add an extra feature $x_0 = 1$ to $\mathrm{x}$ to represent the bias term. i.e. $\mathrm{x} = \begin{bmatrix} 1 & x_1 & x_2 & \cdots & x_D \end{bmatrix}$.
  
- Then $y = c(\mathrm{x})$ is the label of the sample $\mathrm{x}$.
  - We often subscript $y$ with $n$ to denote the label of the $n$-th sample, i.e. $y_n = c(\mathrm{x}_n)$.
  
- We denote $\mathbf{X} \in \mathbb{R}^{N \times D} = \begin{bmatrix} \mathrm{x}_1 \\ \mathrm{x}_2 \\ \vdots \\ \mathrm{x}_N \end{bmatrix}= \begin{bmatrix} x_{11} & x_{12} & \cdots & x_{1D} \\ x_{21} & x_{22} & \cdots & x_{2D} \\ \vdots & \vdots & \ddots & \vdots \\ x_{N1} & x_{N2} & \cdots & x_{ND} \end{bmatrix}$ as the matrix of all samples. Note that each row is a sample and each column is a feature. We can append a column of 1's to the first column of $\mathbf{X}$ to represent the bias term.
- We denote $\mathbf{y} = \begin{bmatrix} y_1 \\ y_2 \\ \vdots \\ y_N \end{bmatrix}$ as the vector of all labels corresponding to the samples in $\mathbf{X}$.
  


- When we denote the conditional probability of $y$ given $\mathbf{x}$ as $P(y|\mathbf{x})$, we often use the following notation:
  
  $$
  \begin{aligned}
  \mathbb{P}(Y = k \mid \mathrm{X} = \mathbf{x}) &= \mathbb{P}(Y = k \mid x_1, x_2, \ldots, x_D) \\
                                        &= \mathbb{P}(Y = k \mid x_1, x_2, \ldots, x_D, x_0=1)
  \end{aligned}
  $$
  
  where $x_0=1$ is the bias term. Note that $\mathrm{X}$ and $Y$ are random variables, and not to be confused with the design matrix $\mathbf{X}$.

- Note very carefully here that $\mathrm{X}$ is a random vector, i.e.

  $$
  \mathrm{X} \in \mathbb{R}^{D} = \begin{bmatrix} X_1 \\ X_2 \\ \vdots \\ X_D \end{bmatrix}
  $$

  and therefore, $\mathrm{X}$ can be characterized by an $D$-dimensional PDF

  $$
  f_{\mathrm{X}}(\mathbf{x}) = f_{X_1, X_2, \ldots, X_D}(x_1, x_2, \ldots, x_D)
  $$

  where each $X_d \in \mathrm{X}$ is a random variable which crystallizes into a value $x_d$ when we observe $\mathrm{X}$.
- We denote the likelihood function as $\mathbb{P}(\mathrm{X} = \mathbf{x} \mid Y = k)$, which is the probability of observing $\mathbf{x}$ given that the sample belongs to class $k$. 
- We denote the prior probability of class $k$ as $\mathbb{P}(Y = k)$.
- We denote the normalizing constant as $\mathbb{P}(\mathrm{X} = \mathbf{x}) = \sum_{k=1}^K \mathbb{P}(Y = k) \mathbb{P}(\mathrm{X} = \mathbf{x} \mid Y = k)$.
- We denote the posterior probability of class $k$ given $\mathbf{x}$ as $\mathbb{P}(Y = k \mid \mathrm{X} = \mathbf{x})$.

- To avoid notational confusion with the design matrix $\mathbf{X}$, one should be clear of the context in which they are used. 
```

$$
\begin{align*}
\mathcal{D}_{\{\mathrm{x}, y\}} &= \left\{\mathrm{x}^{(n)}, y^{(n)}\right\}_{n=1}^N \\
&= \left\{(\mathrm{x}_1, y_1), (\mathrm{x}_2, y_2), \cdots, (\mathrm{x}_N, y_N)\right\} \\
\end{align*}
$$
 
 = $\{(\mathrm{x}_1, y_1), (\mathrm{x}_2, y_2), \cdots, (\mathrm{x}_N, y_N)\}$.