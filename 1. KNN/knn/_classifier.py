from pandas import DataFrame, Series

from ._exception import KNNException
from ._majority_voter import _MajorityVoter
from ._standardiser import Standardiser


class KNNClassifier:
    def __init__(
        self,
        k: int,
        data: DataFrame,
        classes_column: object,
        adaptive_k=True,
        normalise=True,
    ) -> None:
        # flags
        self._adaptive_k = adaptive_k
        self._normalise = normalise

        # data
        self._classes_sorted = data[classes_column]
        data = data.drop(columns=[classes_column])
        self._standardiser = Standardiser(data)
        self._data = self._standardiser.standardied_data() if normalise else data

        # k
        self._k = k
        if k % 2 == 0:
            raise KNNException("K must be odd number")

    def classify(self, sample: Series):
        if self._normalise:
            sample = self._standardiser.standardise_sample(sample)
        labeled_distances = DataFrame(
            {"distance": self._find_distances(sample), "class": self._classes_sorted}
        )
        return _MajorityVoter(labeled_distances, self._k, self._adaptive_k).vote()

    def _find_distances(self, sample: Series) -> Series[float]:
        distances = self._data.copy()
        for label, value in sample.items():
            distances[label] = distances[label].sub(value).abs()
        distances["distance"] = distances.sum(axis=1)
        return distances["distance"]
