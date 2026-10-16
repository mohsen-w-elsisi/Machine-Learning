import numpy as np


def augment_with_ones(x: np.ndarray) -> np.ndarray:
    shape = []
    for dim in x.shape:
        shape.append(1)
    shape[0] = x.shape[0]
    return np.hstack([np.ones(shape), x])
