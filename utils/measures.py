import numpy as np
from enum import Enum

def euclidean_dist(v1: np.ndarray, v2: np.ndarray) -> float:
    return np.sqrt(np.sum((v2 - v1) ** 2))


def l1_dist(v1: np.ndarray, v2: np.ndarray) -> float:
    return np.sum(np.abs(v2 - v1))


def x2_dist(v1: np.ndarray, v2: np.ndarray, eps: float = 1e-10) -> float:
    return 0.5 * np.sum(((v1 - v2) ** 2) / (v1 + v2 + eps))


def hist_intersect(v1: np.ndarray, v2: np.ndarray) -> float:
    return np.sum(np.minimum(v1, v2))


def hellinger_kernel(v1: np.ndarray, v2: np.ndarray) -> float:
    return np.sum(np.sqrt(v1 * v2))

class MeasureType(Enum):
    DISTANCE = "distance"
    SIMILARITY = "similarity"

class SimilarityMeasure(Enum):
    EUCLIDEAN = ("euclidean", euclidean_dist, MeasureType.DISTANCE)
    L1 = ("l1", l1_dist, MeasureType.DISTANCE)
    CHI2 = ("chi2", x2_dist, MeasureType.DISTANCE)
    HIST_INTERSECT = ("hist_intersect", hist_intersect, MeasureType.SIMILARITY)
    HELLINGER = ("hellinger", hellinger_kernel, MeasureType.SIMILARITY)

    def __init__(self, label, func, measure_type):
        self.label = label
        self.func = func
        self.measure_type = measure_type