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

class SimilarityMeasure(Enum):
    EUCLIDEAN = "euclidean"
    L1 = "l1"
    CHI2 = "chi2"
    HIST_INTERSECT = "hist_intersect"
    HELLINGER = "hellinger"
    
    def compute(self, v1: np.ndarray, v2: np.ndarray) -> float:
        if self == SimilarityMeasure.EUCLIDEAN:
            return euclidean_dist(v1, v2)
        elif self == SimilarityMeasure.L1:
            return l1_dist(v1, v2)
        elif self == SimilarityMeasure.CHI2:
            return x2_dist(v1, v2)
        elif self == SimilarityMeasure.HIST_INTERSECT:
            return hist_intersect(v1, v2)
        elif self == SimilarityMeasure.HELLINGER:
            return hellinger_kernel(v1, v2)
        else:
            raise ValueError(f"Unknown similarity measure: {self}")
    
    @property
    def is_similarity(self) -> bool:
        return self in [SimilarityMeasure.HIST_INTERSECT, SimilarityMeasure.HELLINGER]
    
    @property
    def is_distance(self) -> bool:
        return not self.is_similarity
    
    @property
    def name(self) -> str:
        return self.value.upper().replace("_", " ")