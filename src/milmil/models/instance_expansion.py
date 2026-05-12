from typing import Self

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression

from ..types import ProbabilisticEstimator
from .base import BaseMILModel


class InstanceExpansionMIL(BaseMILModel, BaseEstimator, ClassifierMixin):
    def __init__(self, estimator: ProbabilisticEstimator | None = None):
        self.estimator = estimator

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        *args: object,
        **kwargs: object,
    ) -> Self:
        est: ProbabilisticEstimator = (
            self.estimator if self.estimator is not None else LogisticRegression()
        )
        self.estimator_ = est.fit(x, y[z])
        return self

    def predict_proba(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        proba = self.estimator_.predict_proba(x)[:, 1]
        unique_z = np.unique(z)
        return np.array([proba[z == i].max() for i in unique_z])
