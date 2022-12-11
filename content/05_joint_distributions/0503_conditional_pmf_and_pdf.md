# Conditional PMF and PDF

1. $Y=X+N$ such that $N$ is a normal random variable with mean 0 and variance 1.
   1. Note that $X$ and $N$ are independent by definition. This means that $N$ happening does
        not change the probability of $X$ happening. Ask why $N = Y - X$ does not imply that $X$
        is dependent on $N$.
   2. Note that $X$ and $Y$ are **not** independent. Can we justify or our intuition is wrong.
   
2. To find $P[X=1 | Y>0]$, we need the following:
   1. We first recall that to find the conditional probability, we need to find the conditional PDF $f_{X|Y}(x|y)$ first, or more concretely, $f_{X|Y}(x=1|y>0)$.
   2. We first note $f_{X|Y}(x|y) = \frac{f_{X,Y}(x,y)}{f_Y(y)}$.
   3. This is by definition of conditional probability. We can see that in  section 5.3.1 and also in the chapter on conditional probability.
   4. In particular, note that $f_{X, Y}(x, y) = f_X(x) \cap f_Y(y)$, so it is indeed the numerator of the conditional probability.
   5. Overall, it is not clear how to find $f_Y(y)$, though we can find $f_X(x)$.
   6. We will leave finding the denominator for later.
   7. Note that $P[X=1|Y>0]$ is equivalent to integrating $f_{X|Y}(x=+1|y>0)$ for all $y>0$. But we will soon hit a wall when we try to find an expression form for this PDF, furthermore, we could make use of the fact that the marginal PDF of $X$ is given to solve this problem.
   8. Now we instead use Bayes to say that

    $$
    P[X=+1|Y>0] = \dfrac{P[Y>0|X=+1]P[X=+1]}{P[Y>0]}
    $$

    which translates to finding the RHS of the equation. Note the numerator is a consequence of $P(X = +1, Y > 0) = P(Y > 0 | X = +1)P(X = +1)$, which is the definition of conditional probability. The denominator is the marginal probability of $Y>0$, which we will find later.

   9. Note $P[X=+1]$ is trivially equals to $\frac{1}{2}$, since $X$ is a Bernoulli random variable with $p=0.5$. Even though it is not mentioned explicitly, we can assume that $X$ is a Bernoulli random variable with $p=0.5$ since it does seem to fulfil the definition of a Bernoulli random variable provided it is independent trials.

   10. Now to find $P[Y>0|X=+1]$, we need to find $f_{Y|X}(y>0|x=+1)$.

3. To find the conditional distribution $f_{Y|X}(y>0|x=+1)$, we first must be clear that this is a conditional PDF and not a probability yet, i.e. $P[Y>0|X=1]$ is found by integrating this PDF! We must also be clear that this probability is all about $y$ and therefore we will integrate over $dy$ only instead of the usual double integral. Why? Because we are given $X=+1$, this means $X$ is fixed and there is nothing ***random*** about it, you can imagine in the 2D (3D) space PDF where the axis $X$ is fixed at 1, and we are integrating over the curve under $Y>0$ with $X=1$, i.e. $(x=1, y=0.1), (x=1, y=0.2), \ldots$
   1. Now the difficult question is what is $f_{Y|X}(y>0|x=1)$? We can find clues by looking at the equation $Y=X+N$. In laymen terms, $Y=X+N$ means what is $Y$ given $X=1$? So we can simplify $Y=X+N$ to $Y=1+N$. We emphasise that this PDF is a function of $y$ only, and not $x$. But this does not mean $f_{Y|X} = f_Y$, which we will soon see.
   2. Next, by the definition of shifting (linear transformation), if $N$ is a normal random variable of mean $\mu$ and $\sigma$, then shifting it by $1$ merely shifts the mean by $1$ and the variance remains the same [^1]. This shows that $Y$ is actually still a gaussian family, same as $N$, but with a different mean and same variance.
   3. Therefore, $Y=1+N$ is a normal random variable with mean $1+\mu$ and variance $\sigma^2$, $Y \sim \mathcal{N}(1+\mu, \sigma^2)$. With $\mu=0$ and $\sigma=1$, we have $Y \sim \mathcal{N}(1, 1)$.
   4. Now we can find $f_{Y|X}(y>0|x=1)$ by plugging in $y>0$ into the PDF of $\mathcal{N}(1, 1)$, which is $f_{Y|X}(y>0|x=1) = \frac{1}{\sqrt{2\pi}}e^{-\frac{1}{2}(y-1)^2}$. Note that this is a PDF, not a probability yet.
   5. To recover the probability, we must integrate over $dy$.

    $$
    P[Y>0|X=1] = \int_{y>0} \frac{1}{\sqrt{2\pi}}e^{-\frac{1}{2}(y-1)^2} dy = \int_{0}^{\infty} \frac{1}{\sqrt{2\pi}}e^{-\frac{1}{2}(y-1)^2} dy
    $$

    this is because $y>0$ is equivalent to $0<y<\infty$.

   6. We can now use the standard normal table to find the probability, which is $0.8413$. See chan's solution which is $1 - \Phi(-1) = 0.8413$.
   7. Similarly, we can find $P[Y>0|X=-1]$ by plugging in $y>0$ into the PDF of $\mathcal{N}(-1, 1)$, which is $f_{Y|X}(y>0|x=-1) = \frac{1}{\sqrt{2\pi}}e^{-\frac{1}{2}(y+1)^2}$. We can then integrate over $y>0$ to find the probability, $1-\Phi(1)$.
   
4. As of now, we have recovered $P[X=+1]$ and $P[Y>0|X=+1]$, what is left is the denominator $P[Y>0]$. By the law of total probability, we have

    $$
    \begin{aligned}
        P[Y>0] &= P[Y>0|X=+1]P[X=+1] \\
        &+ P[Y>0|X=-1]P[X=-1]
    \end{aligned}
    $$

    which is $0.8413 \times 0.5 + 0.1587 \times 0.5 = 0.5$.

5. Finally, we can now recover $P[X=+1|Y>0]$ by plugging in the values we have found.

    $$
    P[X=+1|Y>0] = \dfrac{P[Y>0|X=+1]P[X=+1]}{P[Y>0]} = \dfrac{0.8413 \times 0.5}{0.5} = 0.8413
    $$

    which is the same as the answer given in the question.

6. Last but not least, to find $P[X=-1|Y>0]$, it is simply the complement of $P[X=+1|Y>0]$, which is $1 - 0.8413 = 0.1587$.

    $$
    P[X=-1|Y>0] = 1 - P[X=+1|Y>0] = 1 - 0.8413 = 0.1587
    $$

    which is the same as the answer given in the question.

[^1]: You can easily plot it out to see that the bell curve shifting 1 on the x axis merely shifts the curve right by 1, and since mean is the center of the bell curve, the mean is shifted by 1. The variance remains the same because the bell curve is symmetric about the mean, and the variance is the width of the bell curve, which remains unchanged.