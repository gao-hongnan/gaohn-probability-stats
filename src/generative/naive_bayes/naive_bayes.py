"""Naive Bayes Classifier.

High level module with business logic.

Classification using Bayes Rule P(Y|X) = P(X|Y)*P(Y)/P(X),
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

NOTE:
    1. This does not implement the log likelihood to avoid underflow.
"""
from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from src.base.estimator import BaseEstimator, T
from src.base.hyperparams import BaseHyperParams
from src.generative.naive_bayes.config import NaivesBayesHyperParams


class NaiveBayesGaussian(BaseEstimator):
    num_samples: int
    num_features: int

    theta: List[List[Dict[str, float]]]
    pi: T

    prior: T
    likelihood: T
    posterior: T

    def __init__(self, hyperparameters: BaseHyperParams = None) -> None:
        super().__init__(hyperparameters=hyperparameters)
        self.random_state = hyperparameters.random_state
        self.num_classes = hyperparameters.num_classes

    def _set_num_samples_and_features(self, X: T) -> None:
        # num_samples unused since we vectorized when estimating parameters
        self.num_samples, self.num_features = X.shape

    def fit(self, X: T, y: T) -> NaiveBayesGaussian:
        """Fit Naive Bayes classifier according to X, y.

        Note:
            Fitting Naive Bayes involves us finding the theta and pi vector.

        Args:
            X (T): N x D matrix
            y (T): N x 1 vector
        """
        self._set_num_samples_and_features(X)

        # Calculate the mean and variance of each feature for each class
        self.theta = self._estimate_likelihood_parameters(X, y)  # this is theta_{X|Y}
        self.pi = self._estimate_prior_parameters(y)  # this is \boldsymbol{\pi}
        return self

    def _estimate_prior_parameters(self, y: T) -> T:
        """Calculate the prior probability of each class.

        Returns a vector of prior probabilities for each class.
        prior = [P(Y = 0), P(Y = 1), ..., P(Y = k)]
        """
        pi = np.zeros(self.num_classes)
        # use for loop for readability or np.bincount(y) / len(y)
        for k in range(self.num_classes):
            pi[k] = np.sum(y == k) / len(y)
        return pi

    def _estimate_likelihood_parameters(
        self, X: T, y: T
    ) -> List[List[Dict[str, float]]]:
        """Estimate the mean and variance of each feature for each class.

        The final theta should have shape K \times D.
        """
        # corresponds to theta_{X|Y} matrix but the last two dimensions
        # is the mean and variance of the feature d given class k
        parameters = np.zeros((self.num_classes, self.num_features, 2))

        for k in range(self.num_classes):
            # Only select the rows where the label equals the given class
            X_where_k = X[np.where(y == k)]  # shape = (num_samples, num_features)
            for d in range(self.num_features):
                mean = X_where_k[:, d].mean()
                var = X_where_k[:, d].var()
                # encode mean as first element and var as second
                parameters[k, d, :] = [mean, var]
        return parameters

    @staticmethod
    def _calculate_conditional_gaussian_pdf(
        x: T, mean: float, var: float, eps: float = 1e-4
    ) -> float:
        """Univariate Gaussian likelihood of the data x given mean and var.

        \mathbb{P}(X_d = x_d | Y = k)

        Args:
            eps (float): Added in denominator to prevent division by zero.
        """
        coeff = 1.0 / math.sqrt(2.0 * math.pi * var + eps)
        exponent = math.exp(-(math.pow(x - mean, 2) / (2 * var + eps)))
        return coeff * exponent

    def _calculate_prior(self) -> T:
        """Calculate the prior probability of each class.

        Returns a vector of prior probabilities for each class.
        prior = [P(Y = 0), P(Y = 1), ..., P(Y = K)].T
        This is our matrix M1 in the notes, and M1 = pi
        due to the construction of the Catagorical distribution.
        """
        prior = self.pi
        return prior

    def _calculate_joint_likelihood(self, x: T) -> T:
        """Calculate the joint likelihood of the data x given the parameters.

        $P(X|Y) = \prod_{d=1}^{D} P(X_d|Y)$

        This is our matrix M2 (M3) in the notes.

        Args:
            x (T): A vector of shape (num_features,).

        Returns:
            T: A vector of shape (num_classes,).
        """
        likelihood = np.ones(self.num_classes)  # M2 matrix in notes
        M3 = np.ones((self.num_classes, self.num_features))  # M3 matrix in notes
        for k in range(self.num_classes):
            for d in range(self.num_features):
                mean = self.theta[k, d, 0]
                var = self.theta[k, d, 1]
                M3[k, d] = self._calculate_conditional_gaussian_pdf(x[d], mean, var)

        likelihood = np.prod(M3, axis=1)
        return likelihood

    def _calculate_posterior(self, x: T) -> T:
        # x: (num_features,) 1 sample
        self.prior = self._calculate_prior()
        self.likelihood = self._calculate_joint_likelihood(x)
        # M3 * M1
        self.posterior = self.likelihood * self.prior
        return np.argmax(self.posterior)

    def predict_one_sample(self, x: T) -> T:
        """Predict the class label of one sample x."""
        return self._calculate_posterior(x)

    def predict(self, X: T) -> T:
        """Predict the class labels of all the samples in X."""
        num_samples = X.shape[0]
        y_pred = np.ones(num_samples)
        for sample_i, x in enumerate(X):
            y_pred[sample_i] = self.predict_one_sample(x)
        return y_pred


if __name__ == "__main__":

    X, y = load_iris(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    hparams = {
        "random_state": 1992,
        "num_classes": 3,
    }
    naives_bayes_hyperparams = NaivesBayesHyperParams.from_dict(hparams)

    gnb = NaiveBayesGaussian(naives_bayes_hyperparams)
    gnb.fit(X_train, y_train)
    y_preds = gnb.predict(X_test)

    report = classification_report(
        y_test,
        y_preds,
        labels=[0, 1, 2],
        target_names=["setosa", "versicolor", "virginica"],
    )
    print(f"Classification report: \n{report}")
    print()
    print(f"Mislabeled points: {(y_preds != y_test).sum()}/{X_test.shape[0]}")

    sk_gnb = GaussianNB(var_smoothing=0)  # to get exact same results as sklearn
    sk_gnb.fit(X_train, y_train)
    sk_y_preds = sk_gnb.predict(X_test)

    report = classification_report(
        y_test,
        sk_y_preds,
        labels=[0, 1, 2],
        target_names=["setosa", "versicolor", "virginica"],
    )
    print(f"Classification report: \n{report}")
    print()
    print(f"Mislabeled points: {(sk_y_preds != y_test).sum()}/{X_test.shape[0]}")
