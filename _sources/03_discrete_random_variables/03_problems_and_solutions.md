# Problems and Solutions

## Problem 1

[Question 11 of Chapter 3 in Introduction to Probability, Statistics, and Random Processes](https://www.probabilitycourse.com/chapter3/3_3_0_chapter3_problems.php)

The number of emails that I get in a weekday (Monday through Friday) can be modeled by a 
Poisson distribution with an average of $\frac{1}{6}$ emails per minute. 
The number of emails that I receive on weekends (Saturday and Sunday) can be modeled by a 
Poisson distribution with an average of $\frac{1}{30}$ emails per minute.

a) What is the probability that I get no emails in an interval of length 4 hours on a Sunday?

b) A random day is chosen (all days of the week are equally likely to be selected), and a random 
interval of length one hour is selected on the chosen day.
It is observed that I did not receive any emails in that interval. 
What is the probability that the chosen day is a weekday?

a) Let $X$ be the number of emails received on a weekend, in a time interval of length $1$ minute. 
Then $X \sim \text{Poisson}(\frac{1}{30})$ has a Poisson distribution with parameter $\lambda = \frac{1}{30}$.

In Poisson's {ref}`poisson_assumptions`, we noted that the **linearity assumption** states 
that the probability of an event occurring is proportional to the length of the time period.
As a consequence, the value of $\lambda$ is proportional to the length of the time period.

And since the problem asked for a time period of length $4$ hours, we have that the 
$\lambda$ is now,

$$
\lambda = 60 \times 4 \times \frac{1}{30} = 8
$$

This should be intuitive because $\lambda$ is the average number of occurences of an event in a time period $T$.
Thus, if in $1$ minute, there is $\frac{1}{30}$ email, then in $240$ minutes (4 hours), there should be $8$ emails.

We can rephrase our initial statement as follows:

Let $X$ be the number of emails received on a weekend, in a time interval of length $4$ hours. 
Then $X \sim \text{Poisson}(8)$ has a Poisson distribution with parameter $\lambda = 8$.

Subsequently, the probability of getting no emails in a time interval of length $4$ hours is given by

$$
\P \lsq X = 0 \rsq = \frac{e^{-8} 8^0}{0!} = e^{-8} \approx 3.4 \times 10^{-4}
$$
