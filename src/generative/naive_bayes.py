"""Naive Bayes Classifier.

High level module with business logic.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from src.base.hyperparams import BaseHyperParams
from src.base.estimator import BaseEstimator, T
import numpy as np
import math
from typing import Tuple, Dict, List


@dataclass
class NaivesBayesHyperParams(BaseHyperParams):
    random_state: int
    num_classes: int

    # one of ["gaussian", "bernoulli", "multinomial", "poisson"]
    distributions: Tuple[str, ...]


class NaivesBayesClassifier(BaseEstimator):
    def __init__(self, hyperparameters: BaseHyperParams = None) -> None:
        super().__init__(hyperparameters=hyperparameters)
        self.random_state = hyperparameters.random_state
        self.num_classes = hyperparameters.num_classes
        self.distributions = hyperparameters.distributions
        self.num_features: int
        self.num_samples: int
        self.estimated_parameters: List[List[Dict[str, float]]]
        self.prior: T
        self.likelihood: T
        self.posterior: T

    def fit(self, X, y):
        self.X, self.y = X, y
        self.num_samples, self.num_features = X.shape
        # Calculate the mean and variance of each feature for each class
        self.estimated_parameters = self._estimate_class_parameters(X, y)
        self.prior = self._calculate_prior(y)

        # fit done
        # 1. prior
        # 2. estimated_parameters
        return self

    def _estimate_class_parameters(self, X, y):
        """Estimate the mean and variance of each feature for each class."""
        parameters = []
        # for each feature (column) conditional on the class k
        # here must be careful, X_k is a 2D array consisting the rows
        # where the label equals the given class
        for i, c in enumerate(range(self.num_classes)):
            # Only select the rows where the label equals the given class
            X_where_c = X[np.where(y == c)]  # shape = (num_samples, num_features)
            parameters.append([])
            for col in X_where_c.T:  # we want the column now so loop over transpose.
                param_dict = {"mean": col.mean(), "var": col.var()}
                parameters[i].append(param_dict)
        return parameters

    # def _calculate_likelihood(self, X, y, x, mean=None, var=None):
    #     """Gaussian likelihood of the data x given mean and var."""
    #     likelihood = np.zeros(self.num_classes)
    #     eps = 1e-4  # Added in denominator to prevent division by zero
    #     # coeff = 1.0 / math.sqrt(2.0 * math.pi * var + eps)
    #     # exponent = math.exp(-(math.pow(x - mean, 2) / (2 * var + eps)))
    #     # 1 row: P(X=x1|Y=0), P(X=x2|Y=0), ..., P(X=xn|Y=0)
    #     # 2 row: P(X=x1|Y=1), P(X=x2|Y=1), ..., P(X=xn|Y=1)
    #     for i, c in enumerate(range(self.num_classes)):
    #         X_where_c = X[np.where(y == c)]

    #         for j, row in enumerate(X_where_c):
    #             mean, var = (
    #                 self.estimated_parameters[i][j]["mean"],
    #                 self.estimated_parameters[i][j]["var"],
    #             )

    #             coeff = 1.0 / math.sqrt(2.0 * math.pi * var + eps)
    #             exponent = math.exp(-(math.pow(x - mean, 2) / (2 * var + eps)))
    #             gaussian = coeff * exponent
    #             likelihood[i] *= gaussian
    #     return likelihood
    def _calculate_likelihood(self, x, mean=None, var=None):
        """Gaussian likelihood of the data x given mean and var"""
        eps = 1e-4  # Added in denominator to prevent division by zero
        coeff = 1.0 / math.sqrt(2.0 * math.pi * var + eps)
        exponent = math.exp(-(math.pow(x - mean, 2) / (2 * var + eps)))
        return coeff * exponent

    def _calculate_prior(self, y: T) -> T:
        """Calculate the prior probability of each class.

        Returns a vector of prior probabilities for each class.
        prior = [P(Y = 0), P(Y = 1), ..., P(Y = c)]
        """
        prior = np.zeros(self.num_classes)  # shape: (num_classes,)
        for i in range(self.num_classes):
            prior[i] = np.sum(y == i) / len(y)
        return prior

    def _calculate_posterior(self, x):  # x is vector
        self.likelihood = np.ones(
            self.num_classes
        )  # TODO: must be ones if not will be zero as I did *= later
        for i, c in enumerate(range(self.num_classes)):
            for j, feature in enumerate(x):
                mean, var = (
                    self.estimated_parameters[i][j]["mean"],
                    self.estimated_parameters[i][j]["var"],
                )
                print(feature, mean, var)
                self.likelihood[i] *= self._calculate_likelihood(feature, mean, var)
        self.posterior = self.likelihood * self.prior
        print(self.posterior)
        return np.argmax(self.posterior)

    def predict(self, X):
        """Predict the class labels of the samples in X"""
        y_pred = [self._calculate_posterior(sample) for sample in X]
        print(len(y_pred))
        return y_pred

    def _classify(self, sample):
        """Classification using Bayes Rule P(Y|X) = P(X|Y)*P(Y)/P(X),
            or Posterior = Likelihood * Prior / Scaling Factor
        P(Y|X) - The posterior is the probability that sample x is of class y given the
                 feature values of x being distributed according to distribution of y and the prior.
        P(X|Y) - Likelihood of data X given class distribution Y.
                 Gaussian distribution (given by _calculate_likelihood)
        P(Y)   - Prior (given by _calculate_prior)
        P(X)   - Scales the posterior to make it a proper probability distribution.
                 This term is ignored in this implementation since it doesn't affect
                 which class distribution the sample is most likely to belong to.
        Classifies the sample as the class that results in the largest P(Y|X) (posterior)
        """
        posteriors = []
        likelihood = self._calculate_likelihood(self.X, self.y, x)
        print(likelihood)
        # # Go through list of classes
        # for i, c in enumerate(self.num_classes):
        #     # Initialize posterior as prior
        #     posterior = self._calculate_prior(c)
        #     # Naive assumption (independence):
        #     # P(x1,x2,x3|Y) = P(x1|Y)*P(x2|Y)*P(x3|Y)
        #     # Posterior is product of prior and likelihoods (ignoring scaling factor)
        #     for feature_value, params in zip(sample, self.parameters[i]):
        #         # Likelihood of feature value given distribution of feature values given y
        #         likelihood = self._calculate_likelihood(
        #             params["mean"], params["var"], feature_value
        #         )
        #         posterior *= likelihood
        #     posteriors.append(posterior)
        # # Return the class with the largest posterior probability
        # return np.argmax(posteriors)


# class NaivesBayesClassifier(BaseEstimator):
#     def __init__(self, hyperparameters: BaseHyperParams = None) -> None:
#         super().__init__(hyperparameters=hyperparameters)
#         self.random_state = hyperparameters.random_state
#         self.num_classes = hyperparameters.num_classes
#         self.distributions = hyperparameters.distributions
#         self.num_features: int
#         self.num_samples: int
#         self.parameters: List[List[Dict[str, float]]]
#         self.prior: T
#         self.likelihood: T
#         self.posterior: T

#     def _calculate_prior(self, y: T) -> T:
#         """Calculate the prior probability of each class.

#         Note
#         ____
#         prior = P(Y = c) for c in classes
#         and is estimated to be the relative frequency of class c in the training set.
#         This estimation can also be derived from performing MLE on the Bernoulli/Categorical
#         distribution.

#         Parameters
#         ----------
#         y : T
#             The target values.

#         Returns
#         -------
#         prior : T
#             The prior probability of each class.
#         """
#         prior = np.zeros(self.num_classes)  # shape: (num_classes,)
#         for i in range(self.num_classes):
#             prior[i] = np.sum(y == i) / len(y)
#         return prior

#     def _calculate_likelihood(self, x: T) -> T:
#         """Calculate the likelihood of each feature given each class.

#         Parameters
#         ----------
#         x: T
#             One feature of the training set.
#         y : T
#             The target values.

#         Returns
#         -------
#         likelihood : T
#             The likelihood of each feature given each class.
#         """
#         likelihood = np.zeros((self.num_features, self.num_classes))

#         for i in range(self.num_features):

#             for j in range(self.num_classes):
#                 if self.distributions[i] == "gaussian":
#                     l = self._calculate_conditional_gaussian_pdf(
#                         x=x,
#                         mean=self.parameters[i][j]["mean"],
#                         var=self.parameters[i][j]["var"],
#                     )
#                 likelihood[:, j] = l

#             # elif self.distributions[i] == "bernoulli":
#             #     likelihood[i, j] = self._calculate_conditional_bernoulli_pdf(
#             #         x=X[:, i], mean=self.parameters[i][j]["mean"]
#             #     )
#             # elif self.distributions[i] == "multinomial":
#             #     likelihood[i, j] = self._calculate_conditional_multinomial_pdf(
#             #         x=X[:, i], mean=self.parameters[i][j]["mean"]
#             #     )
#             # elif self.distributions[i] == "poisson":
#             #     likelihood[i, j] = self._calculate_conditional_poisson_pdf(
#             #         x=X[:, i], mean=self.parameters[i][j]["mean"]
#             #     )
#             # else:
#             #     raise ValueError(f"Invalid distribution: {self.distributions[i]}")

#         return likelihood

#     def _calculate_posterior(self, X: T, prior: T, likelihood: T) -> T:
#         """Calculate the posterior probability of each class given each feature.

#         Parameters
#         ----------
#         X : T
#             The input values.
#         prior : T
#             The prior probability of each class.
#         likelihood : T
#             The likelihood of each feature given each class.

#         Returns
#         -------
#         posterior : T
#             The posterior probability of each class given each feature.
#         """
#         posterior = np.zeros(self.num_classes)

#         for i in range(self.num_classes):
#             posterior[i] = prior[i] * np.prod(likelihood[:, i])
#         return posterior

#     def _calculate_conditional_gaussian_pdf(self, x: T, mean: T, var: T) -> T:
#         """Calculate the conditional Gaussian probability density function.

#         Parameters
#         ----------
#         x : T
#             The input values.
#         mean : T
#             The mean of the Gaussian distribution.
#         var : T
#             The variance of the Gaussian distribution.

#         Returns
#         -------
#         pdf : T
#             The probability density function.
#         """
#         print(f"mean: {mean}, var: {var}, x: {x}")
#         print(
#             f"pdf: {1 / (math.sqrt(2 * math.pi * var)) * np.exp(-((x - mean) ** 2) / (2 * var))}"
#         )
#         return (1 / (math.sqrt(2 * math.pi * var))) * np.exp(
#             -((x - mean) ** 2) / (2 * var)
#         )

#     def fit(self, X: T, y: T) -> NaivesBayesClassifier:
#         """Fit the model according to the given training data.

#         Parameters
#         ----------
#         X : T
#             The input values.
#         y : T
#             The target values.

#         Returns
#         -------
#         self : NaivesBayesClassifier
#             The fitted model.
#         """
#         print(f"X: {X.shape}, y: {y.shape}")
#         self.num_samples, self.num_features = X.shape
#         self.parameters = []  # shape (num_features, num_classes)
#         # for each class, get the mean and variance of each feature if gaussian
#         if (
#             self.distributions[0] == "gaussian"
#         ):  # FIXME: only works for one distribution
#             for i in range(self.num_features):
#                 mean = np.mean(X[:, i])
#                 var = np.var(X[:, i])
#                 self.parameters.append(
#                     [{"mean": mean, "var": var} for i in range(self.num_classes)]
#                 )
#         # NOTE: during fit, we should know the prior and the parameters estimated.
#         # no need to know likelihood as this depends on the given feature value in test
#         self.prior = self._calculate_prior(y)

#         return self

#     def predict(self, X: T) -> T:
#         """Predict the target values given the input values.

#         Parameters
#         ----------
#         X : T
#             The input values.

#         Returns
#         -------
#         y_pred : T
#             The predicted target values.
#         """
#         predictions = np.zeros(self.num_samples)

#         for i in range(self.num_samples):
#             self.likelihood = self._calculate_likelihood(X[i])
#             self.posterior = self._calculate_posterior(
#                 X[i], self.prior, self.likelihood
#             )
#             predictions[i] = np.argmax(self.posterior)

#         print(f"prior: {self.prior.shape}, likelihood: {self.likelihood.shape}")
#         print(f"parameters: {self.parameters}")

#         print(f"posterior: {self.posterior}")
#         return predictions

#     def score(self, X: T, y: T) -> float:
#         """Return the mean accuracy on the given test data and labels.

#         Parameters
#         ----------
#         X : T
#             The input values.
#         y : T
#             The target values.

#         Returns
#         -------
#         score : float
#             The mean accuracy of self.predict(X) wrt. y.
#         """
#         return np.mean(self.predict(X) == y)


if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    from sklearn.model_selection import train_test_split

    features, labels = load_iris(return_X_y=True)
    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, test_size=0.5, random_state=0
    )

    hparams = {
        "random_state": 1992,
        "num_classes": 3,
        "distributions": ("gaussian", "gaussian", "gaussian", "gaussian"),
    }
    naives_bayes_hyperparams = NaivesBayesHyperParams.from_dict(hparams)

    gnb = NaivesBayesClassifier(naives_bayes_hyperparams)
    gnb.fit(train_features, train_labels)  # type: ignore
    predictions = gnb.predict(test_features)  # type: ignore
    print(predictions)

    accuracy = accuracy_score(test_labels, predictions)
    precision, recall, fscore, _ = precision_recall_fscore_support(
        test_labels, predictions, average="macro"
    )

    print(f"Accuracy:  {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F-score:   {fscore:.3f}")
    print()
    print(f"Mislabeled points: {(predictions != test_labels).sum()}/{test_features.shape[0]}")  # type: ignore
