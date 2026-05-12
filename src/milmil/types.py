from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, Self

import numpy as np


class ProbabilisticEstimator(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray, *args: Any, **kwargs: Any) -> Self: ...  # noqa: N803
    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...  # noqa: N803
    def predict(self, X: np.ndarray) -> np.ndarray: ...  # noqa: N803


@dataclass(slots=True)
class Bag:
    features: np.ndarray


@dataclass(slots=True)
class LabeledBag(Bag):
    label: int


def arrays_to_bags(x: np.ndarray, z: np.ndarray) -> list[Bag]:
    unique_z = np.unique(z)
    return [Bag(features=x[z == i]) for i in unique_z]


def bags_to_arrays(bags: Sequence[Bag]) -> tuple[np.ndarray, np.ndarray]:
    if not bags:
        return np.empty((0, 0)), np.empty(0, dtype=np.intp)
    features = np.vstack([b.features for b in bags])
    bag_sizes = [b.features.shape[0] for b in bags]
    z = np.repeat(np.arange(len(bags), dtype=np.intp), bag_sizes)
    return features, z


def arrays_to_labeled_bags(
    x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> list[LabeledBag]:
    unique_z = np.unique(z)
    return [LabeledBag(features=x[z == i], label=int(y[i])) for i in unique_z]


def labeled_bags_to_arrays(
    bags: Sequence[LabeledBag],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not bags:
        return np.empty((0, 0)), np.empty(0, dtype=np.intp), np.empty(0, dtype=np.intp)
    features = np.vstack([b.features for b in bags])
    bag_sizes = [b.features.shape[0] for b in bags]
    z = np.repeat(np.arange(len(bags), dtype=np.intp), bag_sizes)
    y = np.array([b.label for b in bags], dtype=np.intp)
    return features, y, z
