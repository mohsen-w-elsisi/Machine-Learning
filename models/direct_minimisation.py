import numpy as np
from pandas import DataFrame, Series

from .model import Classifier, Model
from .multiple_classifier import MultipleClassifier


class DMRegresser(Model):
    def __init__(self, x: DataFrame, y: Series) -> None:
        super().__init__(x, y)

    def train(self) -> None:
        self._weights = np.linalg.inv(self._x.T @ self._x) @ self._x.T @ self._y


class DMClassifier(DMRegresser, Classifier):
    def __init__(self, x: DataFrame, y: Series[bool]) -> None:
        super().__init__(x, y)


class DMMultipleClassifier(MultipleClassifier[DMClassifier]):
    def construct_classifier(
        self, features: DataFrame, labels: Series[bool]
    ) -> DMClassifier:
        return DMClassifier(features, labels)
