## CHAPTER 5. JOINT DISTRIBUTIONS

![](https://cdn.mathpix.com/cropped/2022_12_07_2239af0d2825f9a1faceg-01.jpg?height=459&width=434&top_left_y=237&top_left_x=171)

(a) $\widehat{\rho}=-0.0038$

![](https://cdn.mathpix.com/cropped/2022_12_07_2239af0d2825f9a1faceg-01.jpg?height=459&width=430&top_left_y=237&top_left_x=625)

(b) $\widehat{\rho}=0.5321$

![](https://cdn.mathpix.com/cropped/2022_12_07_2239af0d2825f9a1faceg-01.jpg?height=461&width=430&top_left_y=238&top_left_x=1074)

(c) $\widehat{\rho}=0.9656$

Figure 5.9: Visualization of correlated variables. Each of these figures represent a scattered plot of a dataset containing $\left(x_{n}, y_{n}\right)_{n=1}^{N} .(\mathrm{a})$ is uncorrelated. (b) is somewhat correlated. (c) is strongly correlated.

Figure 5.9 shows three example datasets. We plot the $\left(x_{n}, y_{n}\right)$ pairs as coordinates in the $2 \mathrm{D}$ plane. The first dataset contains samples that are almost uncorrelated. We can see that $x_{n}$ does not tell us anything about $y_{n}$. The second dataset is moderately correlated. The third dataset is highly correlated: If we know $x_{n}$, we are almost certain to know the corresponding $y_{n}$, with a small number of perturbations.

On a computer, computing the correlation coefficient can be done using built-in commands such as corrcoef in MATLAB and stats.pearsonr in Python. The codes to generate the results in Figure 5.9 (b) are shown below.
![](https://cdn.mathpix.com/cropped/2022_12_07_2239af0d2825f9a1faceg-01.jpg?height=572&width=1148&top_left_y=1289&top_left_x=152)

\title{
5.3 Conditional PMF and PDF
}

Whenever we have a pair of random variables $X$ and $Y$ that are correlated, we can define their conditional distributions, which quantify the probability of $X=x$ given $Y=y$. In this section, we discuss the concepts of conditional PMF and PDF. 

\subsection{CONDITIONAL PMF AND PDF}

\subsubsection{Conditional PMF}

We start by defining the conditional PMF for a pair of discrete random variables.

Definition 5.14. Let $X$ and $Y$ be two discrete random variables. The conditional PMF of $X$ given $Y$ is

$$
p_{X \mid Y}(x \mid y)=\frac{p_{X, Y}(x, y)}{p_{Y}(y)} .
$$

The simplest way to understand this is to view $p_{X \mid Y}(x \mid y)$ as $\mathbb{P}[X=x \mid Y=y]$. That is, given that $Y=y$, what is the probability for $X=x$ ? To see why this perspective makes sense, let us recall the definition of a conditional probability:

$$
\begin{aligned}
p_{X \mid Y}(x \mid y) & =\frac{p_{X, Y}(x, y)}{p_{Y}(y)} \\
& =\frac{\mathbb{P}[X=x \cap Y=y]}{\mathbb{P}[Y=y]}=\mathbb{P}[X=x \mid Y=y]
\end{aligned}
$$

As we can see, the last two equalities are essentially the definitions of conditional probability and the joint PMF.

How should we understand the notation $p_{X \mid Y}(x \mid y)$ ? Is it a one-variable function in $x$ or a two-variable function in $(x, y)$ ? What does $p_{X \mid Y}(x \mid y)$ tell us? To answer these questions, let us first try to understand the randomness exhibited in a conditional PMF. In $p_{X \mid Y}(x \mid y)$, the random variable $Y$ is fixed to a specific value $Y=y$. Therefore there is nothing random about $Y$. All the possibilities of $Y$ have already been taken care of by the denominator $p_{Y}(y)$. Only the variable $x$ in $p_{X \mid Y}(x \mid y)$ has randomness. What do we mean by "fixed at a value $Y=y$ "? Consider the following example.

Example 5.16. Suppose there are two coins. Let

$$
\begin{aligned}
& X=\text { the sum of the values of two coins, } \\
& Y=\text { the value of the first coin. }
\end{aligned}
$$

Clearly, $X$ has 3 states: $0,1,2$, and $Y$ has two states: either 0 or 1 . When we say $p_{X \mid Y}(x \mid 1)$, we refer to the probability mass function of $X$ when fixing $Y=1$. If we do not impose this condition, the probability mass of $X$ is simple:

$$
p_{X}(x)=\left[\frac{1}{4}, \frac{1}{2}, \frac{1}{4}\right] .
$$

However, if we include the conditioning, then

$$
\begin{aligned}
p_{X \mid Y}(x \mid 1) & =\frac{p_{X, Y}(x, 1)}{p_{Y}(1)} \\
& =\frac{\left[0, \frac{2}{4}, \frac{1}{4}\right]}{\frac{1}{6}}=\left[0, \frac{2}{3}, \frac{1}{3}\right] .
\end{aligned}
$$



\section{CHAPTER 5. JOINT DISTRIBUTIONS}

To put this in plain words, when $Y=1$, there is no way for $X$ to take the state 0 . The chance for $X$ to take the state 1 is $2 / 3$ because either $(0,1)$ or $(1,0)$ can give $X=1$. The chance for $X$ to take the state 2 is $1 / 3$ because it has to be $(1,1)$ in order to give $X=2$. Therefore, when we say "conditioned on $Y=1$ ", we mean that we limit our observations to cases where $Y=1$. Since $Y$ is already fixed at $Y=1$, there is nothing random about $Y$. The only variable is $X$. This example is illustrated in Figure 5.10.
![](https://cdn.mathpix.com/cropped/2022_12_07_2239af0d2825f9a1faceg-03.jpg?height=330&width=1294&top_left_y=509&top_left_x=188)

Figure 5.10: Suppose $X$ is the sum of two coins with PMF $0.25,0.5,0.25$. Let $Y$ be the first coin. When $X$ is unconditioned, the PMF is just $[0.25,0.5,0.25]$. When $X$ is conditioned on $Y=1$, then " $X=0$ " cannot happen. Therefore, the resulting PMF $p_{X \mid Y}(x \mid 1)$ only has two states. After normalization we obtain the conditional PMF [0, $0.66,0.33]$.

Since $Y$ is already fixed at a particular value $Y=y, p_{X \mid Y}(x \mid y)$ is a probability mass function of $x$ (we want to emphasize again that it is $x$ and not $y$ ). So $p_{X \mid Y}(x \mid y)$ is a onevariable function in $x$. It is not the same as the usual $\operatorname{PMF} p_{X}(x) \cdot p_{X \mid Y}(x \mid y)$ is conditioned on $Y=y$. For example, $p_{X \mid Y}(x \mid 1)$ is the PMF of $X$ restricted to the condition that $Y=1$. In fact, it follows that

$$
\sum_{x \in \Omega_{X}} p_{X \mid Y}(x \mid y)=\sum_{x \in \Omega_{X}} \frac{p_{X, Y}(x, y)}{p_{Y}(y)}=\frac{\sum_{x \in \Omega_{X}} p_{X, Y}(x, y)}{p_{Y}(y)}=\frac{p_{Y}(y)}{p_{Y}(y)}=1,
$$

but this tells us that $p_{X \mid Y}(x \mid y)$ is a legitimate probability mass of $X$. If we sum over the $y$ 's instead, then we will hit a bump:

$$
\sum_{y \in \Omega_{Y}} p_{X \mid Y}(x \mid y)=\sum_{y \in \Omega_{Y}} \frac{p_{X, Y}(x, y)}{p_{Y}(y)} \neq 1 .
$$

Therefore, while $p_{X \mid Y}(x \mid y)$ is a legitimate probability mass function of $X$, it is not a probability mass function of $Y$.

Example 5.17. Consider a joint PMF given in the following table. Find the conditional $\operatorname{PMF~}_{X \mid Y}(x \mid 1)$ and the marginal $\operatorname{PMF} p_{X}(x)$.

\begin{tabular}{r|cccc} 
& \multicolumn{5}{|c}{$\mathrm{Y}=$} \\
& 1 & 2 & 3 & 4 \\
\hline $\mathrm{X}=1$ & $\frac{1}{20}$ & $\frac{1}{20}$ & $\frac{1}{20}$ & $\frac{0}{20}$ \\
2 & $\frac{1}{20}$ & $\frac{2}{20}$ & $\frac{3}{20}$ & $\frac{1}{20}$ \\
3 & $\frac{1}{20}$ & $\frac{2}{20}$ & $\frac{3}{20}$ & $\frac{1}{20}$ \\
4 & $\frac{0}{20}$ & $\frac{1}{20}$ & $\frac{1}{20}$ & $\frac{1}{20}$
\end{tabular}



\subsection{CONDITIONAL PMF AND PDF}

Solution. To find the marginal PMF, we sum over all the $y$ 's for every $x$ :

$$
\begin{array}{ll}
x=1: & p_{X}(1)=\sum_{y=1}^{4} p_{X, Y}(1, y)=\frac{1}{20}+\frac{1}{20}+\frac{1}{20}+\frac{0}{20}=\frac{3}{20}, \\
x=2: & p_{X}(2)=\sum_{y=1}^{4} p_{X, Y}(2, y)=\frac{1}{20}+\frac{2}{20}+\frac{2}{20}+\frac{1}{20}=\frac{6}{20}, \\
x=3: \quad p_{X}(3)=\sum_{y=1}^{4} p_{X, Y}(3, y)=\frac{1}{20}+\frac{3}{20}+\frac{3}{20}+\frac{1}{20}=\frac{8}{20}, \\
x=4: \quad p_{X}(4)=\sum_{y=1}^{4} p_{X, Y}(4, y)=\frac{0}{20}+\frac{1}{20}+\frac{1}{20}+\frac{1}{20}=\frac{3}{20} .
\end{array}
$$

Hence, the marginal PMF is

$$
p_{X}(x)=\left[\begin{array}{llll}
\frac{3}{20} & \frac{6}{20} & \frac{8}{20} & \frac{3}{20}
\end{array}\right] .
$$

The conditional $\operatorname{PMF} p_{X \mid Y}(x \mid 1)$ is

$$
p_{X \mid Y}(x \mid 1)=\frac{p_{X, Y}(x, 1)}{p_{Y}(1)}=\frac{\left[\begin{array}{llll}
\frac{1}{20} & \frac{1}{20} & \frac{1}{20} & \frac{0}{20}
\end{array}\right]}{\frac{3}{20}}=\left[\begin{array}{llll}
\frac{1}{3} & \frac{1}{3} & \frac{1}{3} & 0
\end{array}\right] .
$$

Practice Exercise 5.7. Consider two random variables $X$ and $Y$ defined as follows.

$$
Y=\left\{\begin{array}{ll}
10^{2}, & \text { with prob } 5 / 6, \\
10^{4}, & \text { with prob } 1 / 6 .
\end{array} \quad X= \begin{cases}10^{-4} Y, & \text { with prob } 1 / 2, \\
10^{-3} Y, & \text { with prob } 1 / 3, \\
10^{-2} Y, & \text { with prob } 1 / 6\end{cases}\right.
$$

Find $p_{X \mid Y}(x \mid y), p_{X}(x)$ and $p_{X, Y}(x, y)$

Solution. Since $Y$ takes two different states, we can enumerate $Y=10^{2}$ and $Y=10^{4}$. This gives us

$$
\begin{aligned}
& p_{X \mid Y}\left(x \mid 10^{2}\right)= \begin{cases}1 / 2, & \text { if } x=0.01, \\
1 / 3, & \text { if } x=0.1 \\
1 / 6, & \text { if } x=1\end{cases} \\
& p_{X \mid Y}\left(x \mid 10^{4}\right)= \begin{cases}1 / 2, & \text { if } x=1 \\
1 / 3, & \text { if } x=10 \\
1 / 6, & \text { if } x=100\end{cases}
\end{aligned}
$$



\section{CHAPTER 5. JOINT DISTRIBUTIONS}

The joint $\operatorname{PMF} p_{X, Y}(x, y)$ is

$$
\begin{aligned}
& p_{X, Y}\left(x, 10^{2}\right)=p_{X \mid Y}\left(x \mid 10^{2}\right) p_{Y}\left(10^{2}\right)= \begin{cases}\left(\frac{1}{2}\right)\left(\frac{5}{6}\right), & x=0.01, \\
\left(\frac{1}{3}\right)\left(\frac{5}{6}\right), & x=0.1 \\
\left(\frac{1}{6}\right)\left(\frac{5}{6}\right), & x=1 .\end{cases} \\
& p_{X, Y}\left(x, 10^{4}\right)=p_{X \mid Y}\left(x \mid 10^{4}\right) p_{Y}\left(10^{4}\right)= \begin{cases}\left(\frac{1}{2}\right)\left(\frac{1}{6}\right), & x=1, \\
\left(\frac{1}{3}\right)\left(\frac{1}{6}\right), & x=10 \\
\left(\frac{1}{6}\right)\left(\frac{1}{6}\right), & x=100\end{cases}
\end{aligned}
$$

Therefore, the joint PMF is given by the following table.

\begin{tabular}{c|ccccc}
$10^{4}$ & 0 & 0 & $\frac{1}{12}$ & $\frac{1}{18}$ & $\frac{1}{36}$ \\
$10^{2}$ & $\frac{5}{12}$ & $\frac{5}{18}$ & $\frac{5}{36}$ & 0 & 0 \\
\hline & $0.01$ & $0.1$ & 1 & 10 & 100
\end{tabular}

The marginal PMF $p_{X}(x)$ is thus

$$
p_{X}(x)=\sum_{y} p_{X, Y}(x, y)=\left[\begin{array}{lllll}
\frac{5}{12} & \frac{5}{18} & \frac{2}{9} & \frac{1}{18} & \frac{1}{36}
\end{array}\right] .
$$

In the previous two examples, what is the probability $\mathbb{P}[X \in A \mid Y=y]$ or the probability $\mathbb{P}[X \in A]$ for some events $A$ ? The answers are giving by the following theorem.

Theorem 5.7. Let $X$ and $Y$ be two discrete random variables, and let $A$ be an event. Then

$$
\begin{array}{ll}
\text { (i) } & \mathbb{P}[X \in A \mid Y=y]=\sum_{x \in A} p_{X \mid Y}(x \mid y) \\
\text { (ii) } & \mathbb{P}[X \in A]=\sum_{x \in A} \sum_{y \in \Omega_{Y}} p_{X \mid Y}(x \mid y) p_{Y}(y)=\sum_{y \in \Omega_{Y}} \mathbb{P}[X \in A \mid Y=y] p_{Y}(y) .
\end{array}
$$

Proof. The first statement is based on the fact that if $A$ contains a finite number of elements, then $\mathbb{P}[X \in A]$ is equivalent to the sum $\sum_{x \in A} \mathbb{P}[X=x]$. Thus,

$$
\begin{aligned}
\mathbb{P}[X \in A \mid Y=y] & =\frac{\mathbb{P}[X \in A \cap Y=y]}{\mathbb{P}[Y=y]} \\
& =\frac{\sum_{x \in A} \mathbb{P}[X=x \cap Y=y]}{\mathbb{P}[Y=y]} \\
& =\sum_{x \in A} p_{X \mid Y}(x \mid y)
\end{aligned}
$$

The second statement holds because the inner summation $\sum_{y \in \Omega_{Y}} p_{X \mid Y}(x \mid y) p_{Y}(y)$ is just the marginal PMF $p_{X}(x)$. Thus the outer summation yields the probability. Example 5.18. Let us follow up on Example 5.17. What is the probability that $\mathbb{P}[X>2 \mid Y=1]$ ? What is the probability that $\mathbb{P}[X>2]$ ?

Solution. Since the problem asks about the conditional probability, we know that it can be computed by using the conditional PMF. This gives us

$$
\begin{aligned}
\mathbb{P}[X>2 \mid Y=1] & =\sum_{x>2} p_{X \mid Y}(x \mid 1) \\
& =\underline{p}_{X \mid Y}(1 \mid 1)+\underline{p}_{X \mid Y}(2 \mid 1)+\underbrace{p_{X \mid Y}(3 \mid 1)}_{\frac{1}{3}}+\underbrace{p_{X \mid Y}(4 \mid 1)}_{0}=\frac{1}{3} .
\end{aligned}
$$

The other probability is

$$
\begin{aligned}
\mathbb{P}[X>2] & =\sum_{x>2} p_{X}(x) \\
& =p_{X}(1)+p_{X}(2)+\underbrace{p_{X}(3)}_{\frac{8}{20}}+\underbrace{p_{X}(4)}_{\frac{3}{20}}=\frac{11}{20} .
\end{aligned}
$$

\section{What is the rule of thumb for conditional distribution?}

- The PMF/PDF should match with the probability you are finding.

- If you want to find the conditional probability $\mathbb{P}[X \in A \mid Y=y]$, use the conditional PMF $p_{X \mid Y}(x \mid y)$.

- If you want to find the probability $\mathbb{P}[X \in A]$, use the marginal $\operatorname{PMF} p_{X}(x)$.

Finally, we define the conditional CDF for discrete random variables.

Definition 5.15. Let $X$ and $Y$ be discrete random variables. Then the conditional CDF of $X$ given $Y=y$ is

$$
F_{X \mid Y}(x \mid y)=\mathbb{P}[X \leq x \mid Y=y]=\sum_{x^{\prime} \leq x} p_{X \mid Y}\left(x^{\prime} \mid y\right) .
$$

\subsubsection{Conditional PDF}

We now discuss the conditioning of a continuous random variable.

Definition 5.16. Let $X$ and $Y$ be two continuous random variables. The conditional PDF of $X$ given $Y$ is

$$
f_{X \mid Y}(x \mid y)=\frac{f_{X, Y}(x, y)}{f_{Y}(y)} .
$$

Example 5.19. Let $X$ and $Y$ be two continuous random variables with a joint PDF

$$
f_{X, Y}(x, y)= \begin{cases}2 e^{-x} e^{-y}, & 0 \leq y \leq x<\infty \\ 0, & \text { otherwise. }\end{cases}
$$

Find the conditional PDFs $f_{X \mid Y}(x \mid y)$ and $f_{Y \mid X}(y \mid x)$.

Solution. We first find the marginal PDFs.

$$
\begin{aligned}
f_{X}(x) & =\int_{-\infty}^{\infty} f_{X, Y}(x, y) d y=\int_{0}^{x} 2 e^{-x} e^{-y} d y=2 e^{-x}\left(1-e^{-x}\right) \\
f_{Y}(y) & =\int_{-\infty}^{\infty} f_{X, Y}(x, y) d x=\int_{y}^{\infty} 2 e^{-x} e^{-y} d x=2 e^{-2 y}
\end{aligned}
$$

Thus, the conditional PDFs are

$$
\begin{aligned}
f_{X \mid Y}(x \mid y) & =\frac{f_{X, Y}(x, y)}{f_{Y}(y)} \\
& =\frac{2 e^{-x} e^{-y}}{2 e^{-2 y}}=e^{-(x+y)}, \quad x \geq y \\
f_{Y \mid X}(y \mid x) & =\frac{f_{X, Y}(x, y)}{f_{X}(x)} \\
& =\frac{2 e^{-x} e^{-y}}{2 e^{-x}\left(1-e^{-x}\right)}=\frac{e^{-y}}{1-e^{-x}}, \quad 0 \leq y<x
\end{aligned}
$$

Where does the conditional PDF come from? We cannot duplicate the argument we used for the discrete case because the denominator of a conditional PMF becomes $\mathbb{P}[Y=y]=0$ when $Y$ is continuous. To answer this question, we first define the conditional $\mathrm{CDF}$ for continuous random variables.

Definition 5.17. Let $X$ and $Y$ be continuous random variables. Then the conditional CDF of $X$ given $Y=y$ is

$$
F_{X \mid Y}(x \mid y)=\frac{\int_{-\infty}^{x} f_{X, Y}\left(x^{\prime}, y\right) d x^{\prime}}{f_{Y}(y)} .
$$

Why should the conditional CDF of continuous random variable be defined in this way? One way to interpret $F_{X \mid Y}(x \mid y)$ is as the limiting perspective. We can define the conditional CDF as

$$
\begin{aligned}
F_{X \mid Y}(x \mid y) & =\lim _{h \rightarrow 0} \mathbb{P}(X \leq x \mid y \leq Y \leq y+h) \\
& =\lim _{h \rightarrow 0} \frac{\mathbb{P}(X \leq x \cap y \leq Y \leq y+h)}{\mathbb{P}[y \leq Y \leq y+h]}
\end{aligned}
$$



\subsection{CONDITIONAL PMF AND PDF}

With some calculations, we have that

$$
\begin{aligned}
\lim _{h \rightarrow 0} \frac{\mathbb{P}(X \leq x \cap y \leq Y \leq y+h)}{\mathbb{P}[y \leq Y \leq y+h]} & =\lim _{h \rightarrow 0} \frac{\int_{-\infty}^{x} \int_{y}^{y+h} f_{X, Y}\left(x^{\prime}, y^{\prime}\right) d y^{\prime} d x^{\prime}}{\int_{y}^{y+h} f_{Y}\left(y^{\prime}\right) d y^{\prime}} \\
& =\lim _{h \rightarrow 0} \frac{\int_{-\infty}^{x} f_{X, Y}\left(x^{\prime}, y^{\prime}\right) d x^{\prime} \cdot h}{f_{Y}(y) \cdot h} \\
& =\frac{\int_{-\infty}^{x} f_{X, Y}\left(x^{\prime}, y^{\prime}\right) d x^{\prime}}{f_{Y}(y)} .
\end{aligned}
$$

The key here is that the small step size $h$ in the numerator and the denominator will cancel each other out. Now, given the conditional CDF, we can verify the definition of the conditional PDF. It holds that

$$
\begin{aligned}
f_{X \mid Y}(x \mid y) & =\frac{d}{d x} F_{X \mid Y}(x \mid y) \\
& =\frac{d}{d x}\left\{\frac{\int_{-\infty}^{x} f_{X, Y}\left(x^{\prime}, y\right) d x^{\prime}}{f_{Y}(y)}\right\} \stackrel{(a)}{=} \frac{f_{X, Y}(x, y)}{f_{Y}(y)},
\end{aligned}
$$

where (a) follows from the fundamental theorem of calculus.

Just like the conditional PMF, we can calculate the probabilities using the conditional PDFs. In particular, if we evaluate the probability where $X \in A$ given that $Y$ takes a particular value $Y=y$, then we can integrate the conditional PDF $f_{X \mid Y}(x \mid y)$, with respect to $x$.

Theorem 5.8. Let $X$ and $Y$ be continuous random variables, and let $A$ be an event.

(i) $\mathbb{P}[X \in A \mid Y=y]=\int_{A} f_{X \mid Y}(x \mid y) d x$

(ii) $\mathbb{P}[X \in A]=\int_{\Omega_{Y}} \mathbb{P}[X \in A \mid Y=y] f_{Y}(y) d y$.

Example 5.20. Let $X$ be a random bit such that

$$
X= \begin{cases}+1, & \text { with prob } 1 / 2 \\ -1, & \text { with prob } 1 / 2\end{cases}
$$

Suppose that $X$ is transmitted over a noisy channel so that the observed signal is

$$
Y=X+N,
$$

where $N \sim \operatorname{Gaussian}(0,1)$ is the noise, which is independent of the signal $X$. Find the probabilities $\mathbb{P}[X=+1 \mid Y>0]$ and $\mathbb{P}[X=-1 \mid Y>0]$.

Solution. First, we know that

$$
f_{Y \mid X}(y \mid+1)=\frac{1}{\sqrt{2 \pi}} e^{-\frac{(y-1)^{2}}{2}} \quad \text { and } \quad f_{Y \mid X}(y \mid-1)=\frac{1}{\sqrt{2 \pi}} e^{-\frac{(y+1)^{2}}{2}}
$$



\section{CHAPTER 5. JOINT DISTRIBUTIONS}

Therefore, integrating $y$ from 0 to $\infty$ gives us

$$
\begin{aligned}
\mathbb{P}[Y>0 \mid X=+1] & =\int_{0}^{\infty} \frac{1}{\sqrt{2 \pi}} e^{-\frac{(y-1)^{2}}{2}} d y \\
& =1-\int_{-\infty}^{0} \frac{1}{\sqrt{2 \pi}} e^{-\frac{(y-1)^{2}}{2}} d y \\
& =1-\Phi\left(\frac{0-1}{1}\right)=1-\Phi(-1) .
\end{aligned}
$$

Similarly, we have $\mathbb{P}[Y>0 \mid X=-1]=1-\Phi(+1)$. The probability we want to find is $\mathbb{P}[X=+1 \mid Y>0]$, which can be determined using Bayes' theorem.

$$
\mathbb{P}[X=+1 \mid Y>0]=\frac{\mathbb{P}[Y>0 \mid X=+1] \mathbb{P}[X=+1]}{\mathbb{P}[Y>0]} .
$$

The denominator can be found by using the law of total probability:

$$
\begin{aligned}
\mathbb{P}[Y>0]= & \mathbb{P}[Y>0 \mid X=+1] \mathbb{P}[X=+1] \\
& +\mathbb{P}[Y>0 \mid X=-1] \mathbb{P}[X=-1] \\
= & 1-\frac{1}{2}(\Phi(+1)+\Phi(-1)) \\
= & \frac{1}{2},
\end{aligned}
$$

since $\Phi(+1)+\Phi(-1)=\Phi(+1)+1-\Phi(+1)=1$. Therefore,

$$
\begin{aligned}
\mathbb{P}[X=+1 \mid Y>0] & =1-\Phi(-1) \\
& =0.8413 .
\end{aligned}
$$

The implication is that if $Y>0$, the probability $\mathbb{P}[X=+1 \mid Y>0]=0.8413$. The complement of this result gives $\mathbb{P}[X=-1 \mid Y>0]=1-0.8413=0.1587$.

Practice Exercise 5.8. Find $\mathbb{P}[Y>y]$, where

$$
X \sim \operatorname{Uniform}[1,2], \quad Y \mid X \sim \operatorname{Exponential}(x) .
$$

Solution. The tricky part of this problem is the tendency to confuse the two variables $X$ and $Y$. Once you understand their roles the problem becomes easy. First notice that $Y \mid X \sim \operatorname{Exponential}(x)$ is a conditional distribution. It says that given $X=x$, the probability distribution of $Y$ is exponential, with the parameter $x$. Thus, we have that

$$
f_{Y \mid X}(y \mid x)=x e^{-x y} .
$$

Why? Recall that if $Y \sim \operatorname{Exponential}(\lambda)$ then $f_{Y}(y)=\lambda e^{-\lambda y}$. Now if we replace $\lambda$ with $x$, we have $x e^{-x y}$. So the role of $x$ in this conditional density function is as a parameter. 

\subsection{CONDITIONAL EXPECTATION}

Given this property, we can compute the conditional probability:

$$
\begin{aligned}
\mathbb{P}[Y>y \mid X=x] & =\int_{y}^{\infty} f_{Y \mid X}\left(y^{\prime} \mid x\right) d y^{\prime} \\
& =\int_{y}^{\infty} x e^{-x y^{\prime}} d y^{\prime}=\left[-e^{-x y^{\prime}}\right]_{y^{\prime}=y}^{\infty}=e^{-x y} .
\end{aligned}
$$

Finally, we can compute the marginal probability:

$$
\begin{aligned}
\mathbb{P}[Y>y] & =\int_{\Omega_{X}} \mathbb{P}\left[Y>0 \mid X=x^{\prime}\right] f_{X}\left(x^{\prime}\right) d x^{\prime} \\
& =\int_{0}^{1} e^{-x^{\prime} y} d x^{\prime} \\
& =\left[\frac{1}{y} e^{-x^{\prime} y}\right]_{x^{\prime}=0}^{x^{\prime}=1}=\frac{1}{y}\left(1-e^{-y}\right) .
\end{aligned}
$$

We can double-check this result by noting that the problem asks about the probability $\mathbb{P}[Y>y]$. Thus, the answer must be a function of $y$ but not of $x$.

\subsection{Conditional Expectation}

\subsubsection{Definition}

When dealing with two dependent random variables, at times we would like to determine the expectation of a random variable when the second random variable takes a particular state. The conditional expectation is a formal way of doing so.

Definition 5.18. The conditional expectation of $X$ given $Y=y$ is

$$
\mathbb{E}[X \mid Y=y]=\sum_{x} x p_{X \mid Y}(x \mid y)
$$

for discrete random variables, and

$$
\mathbb{E}[X \mid Y=y]=\int_{-\infty}^{\infty} x f_{X \mid Y}(x \mid y) d x
$$

for continuous random variables.

There are two points to note here. First, the expectation of $\mathbb{E}[X \mid Y=y]$ is taken with respect to $f_{X \mid Y}(x \mid y)$. We assume that the random variable $Y$ is already fixed at the state $Y=y$. Thus, the only source of randomness is $X$. Secondly, since the expectation $\mathbb{E}[X \mid Y=y]$ has eliminated the randomness of $X$, the resulting function is in $y$.