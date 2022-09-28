$$
\newcommand{\st}{\mid}
\newcommand{\S}{\Omega}
\newcommand{\P}{\mathbb{P}}
\newcommand{\R}{\mathbb{R}}
\newcommand{\v}{\mathbf{v}}
\newcommand{\a}{\mathbf{a}}
\newcommand{\b}{\mathbf{b}}
\newcommand{\c}{\mathbf{c}}
\newcommand{\u}{\mathbf{u}}
\newcommand{\w}{\mathbf{w}}
\newcommand{\x}{\mathbf{x}}
\newcommand{\y}{\mathbf{y}}
\newcommand{\z}{\mathbf{z}}
\newcommand{\0}{\mathbf{0}}
\newcommand{\1}{\mathbf{1}}
\newcommand{\A}{\mathbf{A}}
\newcommand{\B}{\mathbf{B}}
\newcommand{\C}{\mathbf{C}}
\newcommand{\E}{\mathcal{F}}
\newcommand{\lset}{\left\{}
\newcommand{\rset}{\right\}}
\newcommand{\lsq}{\left[}
\newcommand{\rsq}{\right]}
\newcommand{\pmf}{p_X}
$$

# Discrete Random Variables

## Random Variables

### Definition

````{prf:definition} Random Variables
:label: random_variables

A **random variable** $X$ is a function defined by the mapping

$$
\begin{align}
X: \S &\to \R \\
\xi &\mapsto X(\xi)
\end{align}
$$

which maps an *outcome* $\xi \in \S$ to a real number $X(\xi) \in \R$.

We denote the range of $X$ to be $x$ and shorthand the notation of $X(\xi) = x$ to be $X = x$.
````

````{prf:definition} Pre-image of a Random Variable
:label: pre_image

Given a random variable $X: \S \to \R$, define a singleton set $\{x\} \subset \R$, then by 
the pre-image definition of a function, we have

$$
X^{-1}(\{x\}) = \lset \xi \in \S \st X(\xi) \in \{x\} \rset
$$

which is equivalent to

$$
X^{-1}(x) \overset{\text{def}}{=} \lset \xi \in \S \st X(\xi) = x \rset
$$
````

### Examples

````{prf:example} Coin Toss
:label: example_random_variable_coin_toss
:nonumber: true

Consider a fair coin and define an **experiment** of throwing the coin twice.

Define the **random variable** $X$ to be the total number of heads in an experiment.
(i.e if you throw 1 head and 1 tail the total number of heads in this experiment is 1).

What is the probability of getting 1 head in an experiment, i.e. $\P \lsq X = 1 \rsq$?

**Solution**

We define the **sample space** $\S$ of this experiment to be $\{(HH), (HT), (TH), (TT)\}$.

We enumerate each **outcome** $\xi_i$ in the **sample space** as

- $\xi_1 = HH$
- $\xi_2 = HT$
- $\xi_3 = TH$
- $\xi_4 = TT$

$\color{red} {\textbf{First}}$, recall that $X$ is a ***function*** that map an outcome $\xi$ from the **sample space** $\S$ 
to a number $x$ in the real space $\R$. In this context it means that $X$ maps one of the four outcomes
$\xi_i$ to the total number of heads in the experiment
(i.e $X(\cdot) = \textbf{number of heads}$).

It is important to note that the codomain of $X$ is not any arbitrary number. We can only map our 4 
outcomes $\xi_i$ in the domain to 3 distinct numbers $0$, $1$ or $2$, which we will see by manually
enumerating each case below.

$$
X(\xi_1) = 2, \quad X(\xi_2) = 1, \quad X(\xi_3) = 1, \quad X(\xi_4) = 0
$$ (eq:outcome_to_number_of_heads)


With that, this random variable $X$ is **completely determined**. 

$\color{red} {\textbf{Secondly}}$, we need to examine carefully what is meant by $\P[X(\xi) = 1]$ 
since this will answer the question on what is the probability of getting 1 head. 

However, $X(\xi) = 1$ is an expression and not an event that the probability measure $\P$ expects.
Here we should recall that the probability law $\P(\cdot)$ is always applied to an **event** 
$E \in \E$ where $E$ is a set.

So we need to map this expression to an event $E \in \E$. So you can ask yourself how to establish
this "mapping" of $X(\xi) = 1$ to an event in our event space $\E$. This seems pretty easy since
we already know that $X(\xi) = 1$ has two cases matched in {eq}`eq:outcome_to_number_of_heads`, namely
$X(\xi_2) = 1$ and $X(\xi_3) = 1$. So we can simply define the event $E$ to be 
$\lset \xi_2, \xi_3 \rset = \{(HT), (TH)\}$.

We verify that $E = \lset \xi_2, \xi_3 \rset$ is indeed an event in $\E$:

$$
\E = \{\emptyset, \{\xi_1\}, \{\xi_2\}, \{\xi_3\}, \{\xi_4\}, \{\xi_1\, \xi_2\}, \{\xi_1\, \xi_3\}, \{\xi_1\, \xi_4\}, \{\xi_2\, \xi_3\}, \{\xi_2\, \xi_4\}, \{\xi_3\, \xi_4\}, \S \} 
$$

More concretely, given an expression $X(\xi) = x$, we construct the event set $E$ by enumerating all the outcomes $\xi_i$ in the sample space $\S$ that satisfy $X(\xi) = x$.

$$
E = \lset \xi \in \S \st X(\xi) = x \rset
$$

and this coincides with the pre-image of $x$ in the random variable $X$ as defined in {prf:ref}`pre_image`.

$\color{red} {\textbf{Consequently}}$, we have 

$$
\begin{align}
    \P[X(\xi) = 1] &= \P[\{(HT), (TH)\}] \\
                   &= \dfrac{2}{4} \\
                   &= 0.5
\end{align}
$$
````

### Variable vs Random Variable

````{prf:example} Variable vs Random Variable
:label: example_variable_vs_random_variable

Professor Stanley Chan gave a good example of the difference between a variable and a random variable.

The main difference is that a variable is **deterministic** while a random variable is **non-deterministic**.

Consider solving the following equation:

$$
2X = x
$$

Then, if $x$ is a fixed constant, then $X = \dfrac{x}{2}$ is a variable.

However, if $x$ is not fixed, meaning that it can have multiple states, then $X$ is a random variable
since it is not deterministic.

Tie back to the example in {prf:ref}`example_random_variable_coin_toss`, we note that $X$ is a random variable since the 
total number of heads $x$ in an experiment is not fixed. It can be 0, 1 or 2 depending on your toss.
````

### Summary

1. A random variable $X$ is a function that has the sample space $\S$ as its domain and the real space
   $\R$ as its codomain.

2. $X(\S)$ is the set of all possible values that $X$ can take and the mapping is not necessarily
   a bijective function since $X(\xi)$ can take on the same value for different outcomes $\xi$.

3. The elements in $X(\S)$ are denoted as $x$ (i.e. $x \in X(\S)$). 
   They are often called the **states** of $X$.

4. It is important to not confused $X$ and $x$. $X$ is a function while $x$ are the 
   states of $X$. 

5. When we write $\P\lsq X = x \rsq$, we are describing the probability of the random variable $X$
   taking on a ***particular*** state $x$. This is equivalent to $\P \lsq \lset \xi \in \S \st X(\xi) = x \rset \rsq$.

---

## Probability Mass Function

### Definition

````{prf:definition} State Space
:label: def_state_space

The set of all possible states of $X$ is called the **state space** of $X$ and is denoted as $X(\S)$.
````

````{prf:example} State Space of Coin Toss
:label: example_state_space_coin_toss

Let us revisit the example in {prf:ref}`example_random_variable_coin_toss` and examine the state space of $X$.

The state space of $X$ is the set of all possible values that $X$ can take. As enumerated in the example,
we see that the state space of $X$ is $\{0, 1, 2\}$ (i.e. $X$ takes 3 states 0, 1 and 2).
````


````{prf:definition} Probability Mass Function
:label: def_pmf

The **probability mass function** (PMF) of a random variable $X$ is a function that maps each state 
$x$ in the state space $X(\S)$ to its probability $\pmf(x) = \P\lsq X = x \rsq$.

We denoted the PMF as

$$
\begin{align}
    \pmf: X(\S) &\to [0, 1] \\ 
    X(\S) \ni x &\mapsto \pmf(x)
\end{align}
$$
````

````{prf:example} PMF of Coin Toss
:label: example_pmf_coin_toss

Let us revisit the example in {prf:ref}`example_random_variable_coin_toss` and compute the PMF of $X$.

Recall the sample space is given by $\S = \{(HH), (HT), (TH), (TT)\}$ and the state space is given by
$X(\S) = \{0, 1, 2\}$ as enumerated in {prf:ref}`example_state_space_coin_toss`.

Thus, our domain of $\pmf$ is $X(\S) = \{0, 1, 2\}$ we have 3 mappings to compute:

$$
\begin{align}
    \pmf(0) &= \P\lsq X = 0 \rsq = \P\lsq \lset \xi \in \S \st X(\xi) = 0 \rset \rsq = \P\lsq \{(TT)\} \rsq = \dfrac{1}{4} \\
    \pmf(1) &= \P\lsq X = 1 \rsq = \P\lsq \lset \xi \in \S \st X(\xi) = 1 \rset \rsq = \P\lsq \{(HT), (TH)\} \rsq = \dfrac{2}{4} \\
    \pmf(2) &= \P\lsq X = 2 \rsq = \P\lsq \lset \xi \in \S \st X(\xi) = 2 \rset \rsq = \P\lsq \{(HH)\} \rsq = \dfrac{1}{4}
\end{align}
$$

Here we have enumerated all the possible states of $X$ and computed the probability of each state.
Thus, the PMF of $X$ is completely determined by the 3 mappings above.
````

### Normalization

````{prf:theorem} Normalization Property of PMF
:label: thm_pmf_normalization

A PMF should satisfy the following normalization property:

$$
\sum_{x \in X(\S)} \pmf(x) = 1
$$ (eq:pmf_normalization)
````

````{prf:proof}
**TODO**
````

### Sturges' Rule and Cross Validation

See [Introduction to Probability for Data Science](https://probability4datascience.com/index.html)
section 3.2.5.

<!-- ## Citations

```{bibliography}
``` -->