# Conditional Probability

## Definition (Conditional Probability)

Let $\P$ be a probability function and $A, B \in \E$ be events. Then the
**conditional probability** of $A$ given that event $B$ has occurred is
denoted

$$
\P(A|B)
$$

and is defined by

$$
\P(A|B) = \dfrac{\P(A\cap B)}{\P(B)}
$$

## Intuition (Conditional Probability)

The intuition of the conditional probability might not be immediate for
those not inclined in statistical ideas. We will follow this
[link](https://stats.stackexchange.com/questions/326253/what-is-the-intuition-behind-the-formula-for-conditional-probability)
illustrate the
intuition[\^intuition_conditional_probability](https://stats.stackexchange.com/questions/326253/what-is-the-intuition-behind-the-formula-for-conditional-probability).

Informally, the below figure 1 gives you an idea: the shaded area belong
to both $A$ and $B$, So given $B$ has happened, what then, is the
probability of event $A$ occurring? In particular, in the sample space
$B$ now, there is only a portion of $A$ there, and one sees that portion
is $P(A \cap B) = P(A)$.

A good intuition is given that $B$ occurred---with or without $A$---what
is the probability of $A$? I.e, we are now in the universe in which $B$
occurred - which is the full right circle. In that circle, the
probability of A is the area of A intersect B divided by the area of the
circle - or in other words, the number of outcomes of $A$ in the right
circle (which is $n(A \cap B)$, over the number of outcomes of the
reduced sample space $B$.

```{figure} https://storage.googleapis.com/reighns/reighns_ml_projects/docs/probability_and_statistics/02_introduction_to_probability/conditional.png
---
name: fig_conditional_probability
---
Figure 1: Conditional Probability
```

> Therefore, after the intuition, one should not be constantly checking
> what the formula represents, if we have $\P(A ~|~ B)$, then it just
> means given $B$ has happened, what is the probability of $A$
> happening? The logic becomes apparent when we reduce the whole sample
> space $\S$ to be only $B$ now, and that whatever $A$ overlaps with $B$
> will be the probability of this conditional.

## Proposition (Conditional Probability)

-   If $\P(A) < \P(B)$, then $\P(A|B) < \P(B|A)$

-   If $\P(A) > \P(B)$, then $\P(A|B) > \P(B|A)$

-   If $\P(A) = \P(B)$, then $\P(A|B) = \P(B|A)$

