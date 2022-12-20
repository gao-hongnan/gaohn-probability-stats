"""Base Abstract Class for Estimators.

Design Pattern: Template/Strategy/Learner Pattern
For a more sophisicated design, refer to scikit-learn's OOP paradigm
https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/base.py




References:
    - https://towardsdatascience.com/how-the-strategy-design-pattern-can-help-you-quickly-evaluate-alternative-models-66e0f625016f
    - Scikit-learn's OOP paradigm
"""
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
import torch
from typing import TypeVar, Any, Dict, Union
from src.base.hyperparams import BaseHyperParams

# predict(self, X: T) -> T:
# indicates that the input type and the output type are the same
# i.e. torch.Tensor -> torch.Tensor
T = TypeVar("T", np.ndarray, torch.Tensor)


class BaseEstimator(ABC):
    """Base Abstract Class for Estimators.

    All estimators should specify all the parameters that can be set
    at the class level in their ``__init__`` as explicit keyword
    arguments (no ``*args`` or ``**kwargs``).

    Parameters
    ----------
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Attributes
    ----------
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    """

    def __init__(self, hyperparameters: BaseHyperParams = None) -> None:
        self.hyperparameters = hyperparameters

    def __repr__(self) -> str:
        """Return the string representation of the estimator.

        Returns
        -------
        repr : str
            The string representation of the estimator.
        """
        return f"{self.__class__.__name__}()"

    def __str__(self) -> str:
        """Return the string representation of the estimator.

        Returns
        -------
        str : str
            The string representation of the estimator.
        """
        return self.__repr__()

    def __eq__(self, other: BaseEstimator) -> bool:
        """Check if two estimators are equal.

        Parameters
        ----------
        other : BaseEstimator
            The other estimator.

        Returns
        -------
        eq : bool
            True if the estimators are equal, False otherwise.
        """
        return self.__dict__ == other.__dict__

    def __hash__(self) -> int:
        """Get the hash of the estimator.

        Returns
        -------
        hash : int
            The hash of the estimator.
        """
        return hash(tuple(sorted(self.__dict__.items())))

    @abstractmethod
    def fit(self, X: T, y: T = None) -> BaseEstimator:
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples,) or (n_samples, n_outputs), optional
            Target relative to X for classification or regression;
            None for unsupervised learning.

        Returns
        -------
        self : object
            Returns self.
        """

    @abstractmethod
    def predict(self, X: T) -> T:
        """Predict class labels for samples in X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Samples.

        Returns
        -------
        C : array, shape (n_samples,)
            Predicted class label per sample.
        """

    def score(self, X: T, y: T) -> float:
        """Returns the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Test samples.

        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            True labels for X.

        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) wrt. y.
        """

    @property
    def hyperparameters(self) -> int:
        """Get the random state.

        Returns
        -------
        random_state : int
            The random state.
        """
        return self._hyperparameters

    @hyperparameters.setter
    def hyperparameters(self, hyperparameters: BaseHyperParams) -> None:
        """Set the random state.

        Parameters
        ----------
        random_state : int
            The random state.
        """
        if not isinstance(hyperparameters, BaseHyperParams):
            raise TypeError(
                f"hyperparameters must be a BaseHyperParams, not {type(hyperparameters)}"
            )
        self._hyperparameters = hyperparameters


class TestBaseEstimator(BaseEstimator):
    def fit(self, X: T, y: T = None) -> BaseEstimator:
        return self

    def predict(self, X: T) -> T:
        return X

    def score(self, X: T, y: T) -> float:
        return 1.0


if __name__ == "__main__":
    hparams = {"random_state": 42, "num_trees": 100}
    test = TestBaseEstimator(hparams)

    print(test)
    print(test.hyperparameters)
