import numpy as np
from pandas import DataFrame, Series

from .model import Classifier, Model, WeightVector


class GDRegresser(Model):
    """
    A model that uses *Gradient Descent* to in its internal classifications
    """

    def __init__(
        self,
        x: DataFrame,
        y: Series,
        lambda_: float,
        learning_rate: float,
        error_tolerance: float,
        t_max: int | None = None,
    ) -> None:
        """
        Args:
            lambda_ (float): anti-overfitting coeffecient
            learning_rate (float): learning rate
            error_tolerance (float): acceptable gradient of loss function at which to stop training (should approach 0)
            t_max (int | None): maximum number of training epochs, if None then will train until error_tolerance is met
        """

        super().__init__(x, y)

        self._lambda = lambda_
        self._learning_rate = learning_rate
        self._error_tolerance = error_tolerance
        self._t_max = t_max
        self._training_epochs = 0

    def train(self) -> None:
        loss_grad: WeightVector = self._calc_loss_grad()
        while not (
            self._epoch_limit_reached() or self._grad_mag_within_tolerance(loss_grad)
        ):
            loss_grad = self._calc_loss_grad()
            self._weights -= loss_grad * self._learning_rate
            self._training_epochs += 1

    def _calc_loss_grad(self) -> WeightVector:
        return (
            (self._x.T @ self._x @ self._weights)
            - (self._x.T @ self._y)
            + self._lambda * self._weights
        )

    def _epoch_limit_reached(self) -> bool:
        if self._t_max is not None:
            return self._training_epochs >= self._t_max
        else:
            return False

    def _grad_mag_within_tolerance(self, loss_grad: WeightVector | None = None) -> bool:
        loss_grad = loss_grad if loss_grad is not None else self._calc_loss_grad()
        return bool(np.linalg.norm(loss_grad) <= self._error_tolerance)


class GDClassifier(GDRegresser, Classifier): ...


class SGDClassifier(GDClassifier):
    """
    A model that uses *Stochastic Gradient Descent*
    """

    def train(self) -> None:
        no_misclassifications = False
        while not (
            no_misclassifications
            or self._epoch_limit_reached()
            or self._grad_mag_within_tolerance()
        ):
            no_misclassifications = True
            for i, (xi, yi) in enumerate(zip(self._x, self._y)):
                is_classified_correctly: bool = (yi * np.dot(xi, self._weights)) > 0
                if not is_classified_correctly:
                    self._update_weight(i)
                    no_misclassifications = False
            self._training_epochs += 1

    def _update_weight(self, sample_i: int):
        loss_grad = self._calc_sample_loss_grad(sample_i)
        self._weights -= loss_grad * self._learning_rate

    def _calc_sample_loss_grad(self, i: int) -> WeightVector:
        xi: np.ndarray = self._x[np.newaxis, i]
        yi: np.float64 = self._y[i]
        loss_grad = (
            (xi.T @ xi @ self._weights) - (xi * yi) + self._lambda * self._weights
        )
        return loss_grad.reshape(self._weights.shape[0])
