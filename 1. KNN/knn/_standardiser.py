from collections.abc import Hashable
from dataclasses import dataclass

from pandas import DataFrame, Series

from knn._exception import KNNException


class Standardiser:
    def __init__(self, data: DataFrame) -> None:
        self._data = data
        self._features: list[Feature] = [
            Feature(name, data.mean(), data.std()) for name, data in self._data.items()
        ]

    def standardied_data(self):
        standardized_data = self._data.copy()
        for feature in self._features:
            standardized_data[feature.name] = standardized_data[feature.name].map(
                feature.standardise_value
            )
        return standardized_data

    def standardise_sample(self, sample: Series) -> Series:
        self._assert_sample_is_valid(sample)
        for feature in self._features:
            sample.at[feature.name] = feature.standardise_value(
                sample.get(feature.name)
            )
        return sample

    def _assert_sample_is_valid(self, sample: Series):
        if len(sample) != len(self._features):
            raise KNNException("shape of sample given does not match data shape")


@dataclass
class Feature:
    name: Hashable
    mean: float
    std: float

    def standardise_value(self, value) -> float:
        return (value - self.mean) / self.std
