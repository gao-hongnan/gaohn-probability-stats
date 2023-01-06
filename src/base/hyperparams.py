"""Base Abstract Class for Hyperparameters.

Another way of handling this is using PyTorch Lightning's Mixin class,
which saves all the parameters passed to the class constructor as attributes
https://github.com/Lightning-AI/lightning/blob/master/src/pytorch_lightning/core/mixins/hparams_mixin.py
def save_hyperparameters(self, ignore: Optional[List[Any]] = None):
    if ignore is None:
        ignore = []

    frame = inspect.currentframe().f_back
    _, _, _, local_vars = inspect.getargvalues(frame)
    self.hparams = {
        k: v
        for k, v in local_vars.items()
        if k not in set(ignore + ["self"]) and not k.startswith("_")
    }
    for k, v in self.hparams.items():
        setattr(self, k, v)
"""
from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Any, Mapping, Type

from src.utils.general_utils import dataclass_from_dict


@dataclass
class BaseHyperParams(ABC):
    """Base Abstract Class for Hyperparameters."""

    @classmethod
    def from_dict(
        cls: Type[BaseHyperParams], src: Mapping[str, Any]
    ) -> BaseHyperParams:
        """Create a new instance of the class from a dictionary.

        Reference: https://stackoverflow.com/questions/53376099/python-dataclass-from-a-nested-dict
        """
        return dataclass_from_dict(cls, src)
