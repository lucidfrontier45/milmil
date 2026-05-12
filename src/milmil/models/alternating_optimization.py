from typing import Self

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from ..types import ProbabilisticEstimator
from .base import BaseMILModel


class AlternatingOptimizationMIL(BaseMILModel, BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        estimator: ProbabilisticEstimator,
        max_iter: int = 100,
        tol: float = 1e-4,
        k: int = 1,
    ) -> None:
        self.estimator = estimator
        self.max_iter = max_iter
        self.tol = tol
        self.k = k

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        *args: object,
        **kwargs: object,
    ) -> Self:
        pseudo_y = y[z].copy()

        for iteration in range(self.max_iter):
            old_pseudo_y = pseudo_y.copy()
            self.estimator_ = self.estimator.fit(x, pseudo_y)
            proba = self.estimator_.predict_proba(x)[:, 1]

            positive_bags = np.where(y == 1)[0]
            for bag_idx in positive_bags:
                instance_indices = np.where(z == bag_idx)[0]
                instance_proba = proba[instance_indices]
                n_instances = len(instance_proba)
                k_actual = min(self.k, n_instances)
                top_k_indices = instance_indices[np.argsort(instance_proba)[-k_actual:]]
                pseudo_y[instance_indices] = 0
                pseudo_y[top_k_indices] = 1

            if np.array_equal(old_pseudo_y, pseudo_y):
                self.n_iter_ = iteration + 1
                self.converged_ = True
                break
        else:
            self.n_iter_ = self.max_iter
            self.converged_ = False

        return self

    def predict_proba(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        proba = self.estimator_.predict_proba(x)[:, 1]
        unique_z = np.unique(z)
        return np.array([proba[z == i].max() for i in unique_z])
