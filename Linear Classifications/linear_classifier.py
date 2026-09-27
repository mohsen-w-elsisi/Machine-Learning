from pandas import DataFrame, Series

from linear_models import BinaryLinearClassifier, FeatureMatrix


class LinearClassifier:
    def __init__(self, features: FeatureMatrix, labels: Series[str]) -> None:
        self._classes = labels.unique()

        y = DataFrame({"__labels__": labels})
        for _class in self._classes:
            y[_class] = y["__labels__"] == _class

        self._classifiers: dict[str, BinaryLinearClassifier] = {
            _class: BinaryLinearClassifier(features, y[_class].to_numpy())
            for _class in self._classes
        }

    def classify(self, x: DataFrame) -> Series[str]:
        predictions = DataFrame(index=x.index)
        for _class in self._classes:
            predictions[_class] = self._classifiers[_class].confidence(x.to_numpy())
        predictions["__labels__"] = predictions.idxmax(axis=1)
        return predictions["__labels__"]

    def __repr__(self) -> str:
        equations = "\n".join(
            [f"\t{_class}: {model}" for _class, model in self._classifiers.items()]
        )
        return f"""Linear Classifier:\n{equations}
        """
