from typing import Any, Literal

import numpy as np

FeatureMatrix = np.ndarray[tuple[Any, Any], np.dtype[np.float64]]
FeatureVector = np.ndarray[tuple[Any], np.dtype[np.float64]]
YVector = np.ndarray[tuple[Any], np.dtype[np.float64]]
ClassificiationVector = np.ndarray[tuple[Any], np.dtype[np.bool]]
WeightVector = np.ndarray[tuple[Any], np.dtype[np.float64]]


def calculate_weights(x: FeatureMatrix, y: YVector) -> WeightVector:
    xt = np.transpose(x)
    return np.linalg.inv(xt @ x) @ xt @ y


def augment_with_ones(x: np.ndarray) -> np.ndarray:
    shape = []
    for dim in x.shape:
        shape.append(1)
    shape[0] = x.shape[0]
    return np.hstack([np.ones(shape), x])


class _LinearModel:
    def __init__(self, x: FeatureMatrix, y: YVector) -> None:
        self._weights = calculate_weights(augment_with_ones(x), y)

    @property
    def weights(self):
        return self._weights.copy()

    def _predict(self, x: FeatureMatrix):
        return augment_with_ones(x) @ self._weights

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


class LinearRegresser(_LinearModel):
    def __init__(self, x: FeatureMatrix, y: YVector) -> None:
        super().__init__(x, y)

    def predict(self, x: FeatureMatrix):
        return self._predict(x)

    def __repr__(self) -> str:
        return f"Linear Regression model: {super().__repr__()}"


class BinaryLinearClassifier(_LinearModel):
    def __init__(self, x: FeatureMatrix, y: ClassificiationVector) -> None:
        y_as_ints: YVector = np.array([1 if y_val else -1 for y_val in y])
        super().__init__(x, y_as_ints)

    def predict(self, x: FeatureMatrix):
        return self._predict(x) > 0

    def confidence(self, x: FeatureMatrix):
        return self._predict(x)

    def __repr__(self) -> str:
        return f"Binary Linear Classification model: {super().__repr__()}"
