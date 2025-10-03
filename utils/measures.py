import numpy as np


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
