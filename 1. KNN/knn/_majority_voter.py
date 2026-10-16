from pandas import DataFrame


class _MajorityVoter:
    def __init__(self, distances: DataFrame, inital_k: int, adaptive_k: bool) -> None:
        self._all_distances = distances.sort_values("distance")
        self._k = inital_k
        self._adaptive_k = adaptive_k

    def vote(self) -> object:
        if self._adaptive_k:
            while not self._is_tie() or self._k_at_max():
                self._increase_k()
        return self._count_votes()

    @property
    def _nearest_neighbours(self) -> DataFrame:
        return self._all_distances[0 : self._k]

    def _increase_k(self):
        self._k += 2

    def _count_votes(self):
        all_classes = self._nearest_neighbours["class"]
        majority_class = all_classes.value_counts().index[0]
        return majority_class

    def _k_at_max(self) -> bool:
        return self._k >= len(self._all_distances) - 1

    def _is_tie(self) -> bool:
        last_neighbour = self._all_distances.iloc[self._k - 1]
        peeked_neighbour = self._all_distances.iloc[self._k]
        return last_neighbour["distance"] == peeked_neighbour["distance"]
