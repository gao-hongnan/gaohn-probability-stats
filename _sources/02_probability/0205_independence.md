# Independence

## Definition (Independent Events)

Two events $A, B \in \E$ are **statiscally independent** if

$$
\P(A \cap B) = \P(A)P(B)
$$


## Intuition (Independence) \[\^intuition_independence\] \[\^**Stanley H. Chan: Introduction to Probability for Data Science, 2021. (pp. 85-88)**\] {#intuition-independence-intuition_independence-stanley-h-chan-introduction-to-probability-for-data-science-2021-pp-85-88}

The formula above is not at all intuitive.

A more intuitive way to think about it is to think that two events $A$
and $B$ are **independent** if the occurrence of $B$ does not affect the
probability of the occurrence of $A$. This can be illustrated further
with the following narrative.

Let us assume the scenario: given that $B$ occurred (with or without
event $A$ occurring), what is the probability of event $A$ occurring? In
other words, we are now in the universe in which $B$ occurred - which is
the full right circle. In that right hand side circle ($B$), the
probability of $A$ is the area of $A$ intersect $B$ divided by the area
of the circle - or in other words, the probability of $A$ is the number
of outcomes of $A$ in the right circle (which is $n(A \cap B)$, over the
number of outcomes of the reduced sample space $B$. Therefore, if we
think of independence as event $B$ occuring not affecting event $A$
occurring, then it means that the probability of $A$ occurring is still
the probability of $A$ occurring. i.e $\P(A|B) = \P(A)$

It follows immediately that
$$P(A) = P(A|B) = \dfrac{P(A \cap B)}{P(B)} \Longrightarrow P(A) P(B) = P(A \cap B)$$

```{figure} https://storage.googleapis.com/reighns/reighns_ml_projects/docs/probability_and_statistics/02_introduction_to_probability/conditional.png
---
height: 300px
name: conditional
---
```

> So the intuition can be understood by using conditional, say
> $\P(A ~|~ B)$, if $A$ and $B$ are truly independent, then even if $B$
> happened, the probability of $A$ should remain unchanged, which means
> that $\P(A ~|~ B) = \P(A)$, but also recall the definition of
> conditionals, $\P(A ~|~ B) = \dfrac{\P(A \cap B)}{B}$, so equating
> them we have the nice equation of $\P(A)\P(B) = \P(A \cap B)$.

\[\^intuition_independence\]: [The definition of independence is not
intuitive](https://math.stackexchange.com/questions/123192/the-definition-of-independence-is-not-intuitive)
\[\^**Stanley H. Chan: Introduction to Probability for Data Science,
1.    (pp. 85-88)**\]: **Stanley H. Chan: Introduction to Probability
for Data Science, 2021. (pp. 85-88)**

## Definition (Equivalent Independence Definition)

We have stated this in the previous intuition section. Now we formally
state it:

Let $A$ nad $B$ be two events such that $\P(A)$ and $\P(B)$ are more
than $0$, then $A$ and $B$ are **independent** if

$$
\P(A|B) = \P(A) \quad \P(B|A) = \P(B)
$$

## Definition (Disjoint vs Independence)

Given two events $A$ and $B$. The only condition when
$\textbf{Disjoint} \iff \textbf{Independence}$ if that if $\P(A) = 0$ or
$\P(B) = 0$.

## Exercise (Independence) \[\^**Stanley H. Chan: Introduction to Probability for Data Science, 2021. (pp. 87-88)**\] {#exercise-independence-stanley-h-chan-introduction-to-probability-for-data-science-2021-pp-87-88}

Consider the experiment of throwing a die twice. One should be clear
from the context that the outcomes are in the form of a tuple
$(\textbf{dice_1}, \textbf{dice_2})$ and the sample space is:

$$
\S = \left\{(1, 1), (1, 2), \ldots, (6, 6)\right\}
$$

Define the three events below:

$$
A = \{\textbf{1st dice is 3}\} \quad B = \{\textbf{sum of two die is 7}\} \quad C = \{\textbf{sum of two die is 8}\}
$$

We want to find out if events $A$ and $B$ are independent? How about $A$
and $C$?

We focus on the independence of $A$ and $C$ first. The author said that
intuitively, given that event $C$ has happened, will this affect the
**probability of $A$** happening? I assume that this means we do have to
know the probability of event $A$ without $C$ first.

We can enumerate and see that event $A$ has the following set
representation:

$$
A = \{(3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6)\}
$$

which amounts to $\P(A) = \frac{6}{36} = \frac{1}{6}$. Now if $C$
happened, we know that the two rolls have a sum of $8$, and we cannot
construct a sum of $8$ with a roll of $1$. To me, I immediately know
that event $A$ cannot have the outcome that has a $1$ in the second
roll, and thus the outcomes should only be limited to $5$ instead of $6$
and hence dependence is established.

I believe somewhere my intuition is flawed, the author mentioned that:

> If you like a more intuitive argument, you can imagine that C has
> happened, i.e., the sum is 8. Then the probability for the first die
> to be 1 is 0 because there is no way to construct 8 when the first die
> is 1. As a result, we have eliminated one choice for the first die,
> leaving only five options. Therefore, since C has influenced the
> probability of A, they are dependent.

I think I cannot understand why the author mentioned about \"first die\"
when in event $A$, the first die is already a $3$.

One can find more explanation here\[\^mathstack1\] and
here\[\^mathstack2\].

\[\^**Stanley H. Chan: Introduction to Probability for Data Science,
1.    (pp. 87-88)**\]: **Stanley H. Chan: Introduction to Probability
for Data Science, 2021. (pp. 87-88)** \[\^mathstack1\]: [Intuition on
independence of two
events](https://math.stackexchange.com/questions/4403684/intuition-on-independence-of-two-events/)
\[\^mathstack2\]: [Dartboard paradox and understanding
independence](https://math.stackexchange.com/questions/3897127/dartboard-paradox-and-understanding-independence)