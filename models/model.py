from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from pandas import DataFrame, Series

from .utils import augment_with_ones

FeatureMatrix = np.ndarray[tuple[Any, Any], np.dtype[np.float64]]
FeatureVector = np.ndarray[tuple[Any], np.dtype[np.float64]]
YVector = np.ndarray[tuple[Any], np.dtype[np.float64]]
ClassificiationVector = np.ndarray[tuple[Any], np.dtype[np.bool]]
WeightVector = np.ndarray[tuple[Any], np.dtype[np.float64]]


class Model(ABC):
    def __init__(self, x: DataFrame, y: Series) -> None:
        self._x: FeatureMatrix = augment_with_ones(x.to_numpy())
        self._y: YVector = y.to_numpy()
        self._weights: WeightVector = np.ones(self._x.shape[1])

    @abstractmethod
    def train(self) -> None: ...

    def predict(self, x: DataFrame) -> Series:
        predictions = augment_with_ones(x.to_numpy()) @ self._weights
        return Series(predictions, index=x.index)

    @property
    def _model_type(self) -> str:
        return self.__class__.__name__

    def __str__(self) -> str:
        rhs = " + ".join(
            [
                f"{weight:.2f}{f"x{i}" if i != 0 else ""}"
                for i, weight in enumerate(self._weights)
            ]
        )
        return f"y = {rhs}"

    def __repr__(self) -> str:
        return self.__str__()


class Classifier(Model):
    def __init__(self, x: DataFrame, y: Series[bool]) -> None:
        y_labels = y.map(lambda l: 1 if l else -1)
        super().__init__(x, y_labels)

    def classify(self, x: DataFrame) -> Series[bool]:
        return self.predict(x) > 0
