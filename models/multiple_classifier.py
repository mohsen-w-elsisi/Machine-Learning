from abc import ABC, abstractmethod

from pandas import DataFrame, Series

from .model import Classifier


class MultipleClassifier[T: Classifier](ABC):
    """
    Classifier works by using One VS All method
    """

    def __init__(self, features: DataFrame, labels: Series[str]) -> None:
        self._classes = labels.unique()

        y = DataFrame({"__labels__": labels})
        for _class in self._classes:
            y[_class] = y["__labels__"] == _class

        self._classifiers: dict[str, T] = {
            _class: self.construct_classifier(features, y[_class])
            for _class in self._classes
        }

    @abstractmethod
    def construct_classifier(self, features: DataFrame, labels: Series[bool]) -> T: ...

    def train(self) -> None:
        for classifier in self._classifiers.values():
            classifier.train()

    def classify(self, x: DataFrame) -> Series[str]:
        predictions = DataFrame(index=x.index)
        for _class in self._classes:
            predictions[_class] = self._classifiers[_class].predict(x)
        predictions["__labels__"] = predictions.idxmax(axis=1)
        return predictions["__labels__"]

    def __repr__(self) -> str:
        equations = "\n".join(
            [f"\t{_class}: {model}" for _class, model in self._classifiers.items()]
        )
        return f"""Linear Classifier:\n{equations}
        """
