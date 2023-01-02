# Bayes Optimal Classifier

## Definition

For a classification problem on $\mathcal X\times\mathcal Y$ with feature space $\mathcal X = \mathbb R^d$ and $K$ classes $\mathcal Y = \{1,\ldots,K\}$. The Bayes classifier $\eta : \mathcal X \to \mathcal Y$ is defined as

$$
\eta(x) :=\arg\max_{k\in\mathcal Y}\ \mathbb P(Y=k\ |\ X=x) 
$$

With this definition, if we define the [risk](https://en.wikipedia.org/wiki/Statistical_risk) of a classifier (i.e. any function $f : \mathcal X \to \mathcal Y$) as the quantity

$$
\mathcal R(f):=\mathbb P(f(X)\ne Y)
$$

which simply represents the probability that the classifier $f$ makes a mistake, it is straightforward to prove that $\eta$ minimizes $\mathcal R$ (see the proof on the [wiki article](https://en.wikipedia.org/wiki/Bayes_classifier#Proof_of_Optimality)). Which means that for any classifier $f$, you **always** have

$$
\mathcal R(f) \ge \mathcal R(\eta) 
$$

Note that this definition merely says that the Bayes classifier achieves minimal zero-one loss over any other
deterministic classifier, it does not say anything about it achieving zero error.

## References

- pp 19 of A Course in Machine Learning by Hal Daume III
- https://stats.stackexchange.com/questions/567299/what-does-it-mean-for-the-bayes-classifier-to-be-optimal
- https://en.wikipedia.org/wiki/Bayes_classifier