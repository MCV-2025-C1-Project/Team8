import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
from utils.measures import (
    euclidean_dist,
    l1_dist,
    x2_dist,
    hist_intersect,
    hellinger_kernel,
)


class TestEuclideanDistance:

    def test_euclidean_distance_identical_vectors(self) -> None:
        # Given two identical vectors
        v1: np.ndarray = np.array([1.0, 2.0, 3.0])
        v2: np.ndarray = np.array([1.0, 2.0, 3.0])

        # When computing euclidean distance
        result: float = euclidean_dist(v1, v2)

        # Then distance should be zero
        assert abs(result) < 1e-10

    def test_euclidean_distance_orthogonal_vectors(self) -> None:
        # Given two orthogonal unit vectors
        v1: np.ndarray = np.array([1.0, 0.0])
        v2: np.ndarray = np.array([0.0, 1.0])

        # When computing euclidean distance
        result: float = euclidean_dist(v1, v2)

        # Then distance should be sqrt(2)
        expected: float = np.sqrt(2.0)
        assert abs(result - expected) < 1e-10

    def test_euclidean_distance_simple_case(self) -> None:
        # Given two simple vectors
        v1: np.ndarray = np.array([0.0, 0.0])
        v2: np.ndarray = np.array([3.0, 4.0])

        # When computing euclidean distance
        result: float = euclidean_dist(v1, v2)

        # Then distance should be 5 (3-4-5 triangle)
        assert abs(result - 5.0) < 1e-10


class TestL1Distance:

    def test_l1_distance_identical_vectors(self) -> None:
        # Given two identical vectors
        v1: np.ndarray = np.array([1.0, 2.0, 3.0])
        v2: np.ndarray = np.array([1.0, 2.0, 3.0])

        # When computing L1 distance
        result: float = l1_dist(v1, v2)

        # Then distance should be zero
        assert abs(result) < 1e-10

    def test_l1_distance_simple_case(self) -> None:
        # Given two vectors with known differences
        v1: np.ndarray = np.array([1.0, 2.0, 3.0])
        v2: np.ndarray = np.array([2.0, 4.0, 1.0])

        # When computing L1 distance
        result: float = l1_dist(v1, v2)

        # Then distance should be |1-2| + |2-4| + |3-1| = 1 + 2 + 2 = 5
        assert abs(result - 5.0) < 1e-10

    def test_l1_distance_negative_values(self) -> None:
        # Given vectors with negative values
        v1: np.ndarray = np.array([-1.0, -2.0])
        v2: np.ndarray = np.array([1.0, 2.0])

        # When computing L1 distance
        result: float = l1_dist(v1, v2)

        # Then distance should be |-1-1| + |-2-2| = 2 + 4 = 6
        assert abs(result - 6.0) < 1e-10


class TestChiSquaredDistance:

    def test_chi_squared_identical_vectors(self) -> None:
        # Given two identical positive vectors
        v1: np.ndarray = np.array([1.0, 2.0, 3.0])
        v2: np.ndarray = np.array([1.0, 2.0, 3.0])

        # When computing chi-squared distance
        result: float = x2_dist(v1, v2)

        # Then distance should be zero
        assert abs(result) < 1e-10

    def test_chi_squared_simple_case(self) -> None:
        # Given two positive vectors
        v1: np.ndarray = np.array([2.0, 4.0])
        v2: np.ndarray = np.array([4.0, 2.0])

        # When computing chi-squared distance
        result: float = x2_dist(v1, v2)

        # Then distance should be 0.5 * ((2-4)^2/(2+4) + (4-2)^2/(4+2)) = 0.5 * (4/6 + 4/6) = 0.5 * 8/6 = 2/3
        expected: float = 2.0 / 3.0
        assert abs(result - expected) < 1e-10

    def test_chi_squared_with_eps(self) -> None:
        # Given vectors where one has zero values
        v1: np.ndarray = np.array([0.0, 1.0])
        v2: np.ndarray = np.array([1.0, 0.0])
        eps: float = 1e-5

        # When computing chi-squared distance with epsilon
        result: float = x2_dist(v1, v2, eps=eps)

        # Then result should be finite (not divide by zero)
        assert np.isfinite(result)
        assert result > 0


class TestHistogramIntersection:

    def test_histogram_intersection_identical_vectors(self) -> None:
        # Given two identical normalized histograms
        v1: np.ndarray = np.array([0.3, 0.4, 0.3])
        v2: np.ndarray = np.array([0.3, 0.4, 0.3])

        # When computing histogram intersection
        result: float = hist_intersect(v1, v2)

        # Then intersection should be 1.0 (sum of minimums equals sum of identical values)
        assert abs(result - 1.0) < 1e-10

    def test_histogram_intersection_disjoint_vectors(self) -> None:
        # Given two disjoint histograms
        v1: np.ndarray = np.array([1.0, 0.0, 0.0])
        v2: np.ndarray = np.array([0.0, 0.0, 1.0])

        # When computing histogram intersection
        result: float = hist_intersect(v1, v2)

        # Then intersection should be 0.0 (no overlap)
        assert abs(result) < 1e-10

    def test_histogram_intersection_partial_overlap(self) -> None:
        # Given two partially overlapping histograms
        v1: np.ndarray = np.array([0.5, 0.3, 0.2])
        v2: np.ndarray = np.array([0.2, 0.3, 0.5])

        # When computing histogram intersection
        result: float = hist_intersect(v1, v2)

        # Then intersection should be min(0.5,0.2) + min(0.3,0.3) + min(0.2,0.5) = 0.2 + 0.3 + 0.2 = 0.7
        expected: float = 0.7
        assert abs(result - expected) < 1e-10


class TestHellingerKernel:

    def test_hellinger_kernel_identical_vectors(self) -> None:
        # Given two identical normalized histograms
        v1: np.ndarray = np.array([0.25, 0.25, 0.25, 0.25])
        v2: np.ndarray = np.array([0.25, 0.25, 0.25, 0.25])

        # When computing Hellinger kernel
        result: float = hellinger_kernel(v1, v2)

        # Then kernel should be 1.0 (sum of sqrt(0.25*0.25) * 4 = 0.25 * 4 = 1.0)
        assert abs(result - 1.0) < 1e-10

    def test_hellinger_kernel_simple_case(self) -> None:
        # Given two simple normalized histograms
        v1: np.ndarray = np.array([0.64, 0.36])  # sqrt values: 0.8, 0.6
        v2: np.ndarray = np.array([0.36, 0.64])  # sqrt values: 0.6, 0.8

        # When computing Hellinger kernel
        result: float = hellinger_kernel(v1, v2)

        # Then kernel should be sqrt(0.64*0.36) + sqrt(0.36*0.64) = 2 * sqrt(0.2304) = 2 * 0.48 = 0.96
        expected: float = 0.96
        assert abs(result - expected) < 1e-10
