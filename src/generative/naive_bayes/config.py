"""Concrete Implementation of Naive Bayes HyperParams."""
from __future__ import annotations

from dataclasses import dataclass

from src.base.hyperparams import BaseHyperParams


@dataclass
class NaivesBayesHyperParams(BaseHyperParams):
    random_state: int
    num_classes: int
