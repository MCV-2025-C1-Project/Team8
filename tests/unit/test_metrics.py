import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
from typing import List, Union
from utils.metrics import apk, mapk


class TestAPK:

    def test_apk_perfect_prediction(self) -> None:
        # Given actual items and perfect prediction in correct order
        actual: List[int] = [1, 2, 3]
        predicted: List[int] = [1, 2, 3]
        k: int = 3

        # When computing average precision at k
        result: float = apk(actual, predicted, k)

        # Then precision should be 1.0 (perfect)
        assert abs(result - 1.0) < 1e-10

    def test_apk_no_matches(self) -> None:
        # Given actual items and predictions with no matches
        actual: List[int] = [1, 2, 3]
        predicted: List[int] = [4, 5, 6]
        k: int = 3

        # When computing average precision at k
        result: float = apk(actual, predicted, k)

        # Then precision should be 0.0 (no matches)
        assert abs(result) < 1e-10

    def test_apk_partial_matches_correct_order(self) -> None:
        # Given actual items and partial matches in correct order
        actual: List[int] = [1, 2, 3]
        predicted: List[int] = [1, 4, 2]
        k: int = 3

        # When computing average precision at k
        result: float = apk(actual, predicted, k)

        # Then precision should be (1/1 + 2/3) / 3 = (1 + 0.667) / 3 = 0.556
        expected: float = (1.0 / 1.0 + 2.0 / 3.0) / 3.0
        assert abs(result - expected) < 1e-10

    def test_apk_partial_matches_wrong_order(self) -> None:
        # Given actual items and partial matches in wrong order
        actual: List[int] = [1, 2, 3]
        predicted: List[int] = [2, 1, 4]
        k: int = 3

        # When computing average precision at k
        result: float = apk(actual, predicted, k)

        # Then precision should be (1/1 + 2/2) / 3 = (1 + 1) / 3 = 0.667
        expected: float = (1.0 / 1.0 + 2.0 / 2.0) / 3.0
        assert abs(result - expected) < 1e-10

    def test_apk_with_k_limit(self) -> None:
        # Given actual items and more predictions than k
        actual: List[int] = [1, 2]
        predicted: List[int] = [3, 4, 1, 2, 5]
        k: int = 2

        # When computing average precision at k (only first 2 predictions considered)
        result: float = apk(actual, predicted, k)

        # Then precision should be 0.0 (no matches in first 2 predictions)
        assert abs(result) < 1e-10

    def test_apk_empty_actual(self) -> None:
        # Given empty actual list and some predictions
        actual: List[int] = []
        predicted: List[int] = [1, 2, 3]
        k: int = 3

        # When computing average precision at k
        result: float = apk(actual, predicted, k)

        # Then precision should be 0.0 (no actual items to match)
        assert abs(result) < 1e-10

    def test_apk_with_strings(self) -> None:
        # Given actual items and predictions as strings
        actual: List[str] = ["cat", "dog", "bird"]
        predicted: List[str] = ["cat", "fish", "dog"]
        k: int = 3

        # When computing average precision at k
        result: float = apk(actual, predicted, k)

        # Then precision should be (1/1 + 2/3) / 3 = (1 + 0.667) / 3 = 0.556
        expected: float = (1.0 / 1.0 + 2.0 / 3.0) / 3.0
        assert abs(result - expected) < 1e-10

    def test_apk_duplicate_predictions(self) -> None:
        # Given actual items and predictions with duplicates
        actual: List[int] = [1, 2, 3]
        predicted: List[int] = [1, 1, 2]
        k: int = 3

        # When computing average precision at k
        result: float = apk(actual, predicted, k)

        # Then precision should be (1/1 + 2/3) / 3 = 0.556 (duplicates ignored)
        expected: float = (1.0 / 1.0 + 2.0 / 3.0) / 3.0
        assert abs(result - expected) < 1e-10


class TestMAPK:

    def test_mapk_perfect_predictions(self) -> None:
        # Given multiple queries with perfect predictions
        actual: List[List[int]] = [[1, 2], [3, 4]]
        predicted: List[List[int]] = [[1, 2], [3, 4]]
        k: int = 2

        # When computing mean average precision at k
        result: float = mapk(actual, predicted, k)

        # Then mean precision should be 1.0 (all perfect)
        assert abs(result - 1.0) < 1e-10

    def test_mapk_no_matches(self) -> None:
        # Given multiple queries with no matches
        actual: List[List[int]] = [[1, 2], [3, 4]]
        predicted: List[List[int]] = [[5, 6], [7, 8]]
        k: int = 2

        # When computing mean average precision at k
        result: float = mapk(actual, predicted, k)

        # Then mean precision should be 0.0 (no matches)
        assert abs(result) < 1e-10

    def test_mapk_mixed_performance(self) -> None:
        # Given multiple queries with mixed performance
        actual: List[List[int]] = [[1, 2], [3, 4]]
        predicted: List[List[int]] = [[1, 2], [5, 6]]
        k: int = 2

        # When computing mean average precision at k
        result: float = mapk(actual, predicted, k)

        # Then mean precision should be (1.0 + 0.0) / 2 = 0.5
        expected: float = 0.5
        assert abs(result - expected) < 1e-10

    def test_mapk_partial_matches(self) -> None:
        # Given multiple queries with partial matches
        actual: List[List[int]] = [[1, 2, 3], [4, 5]]
        predicted: List[List[int]] = [[1, 6, 2], [4, 7]]
        k: int = 3

        # When computing mean average precision at k
        result: float = mapk(actual, predicted, k)

        # Then mean precision should be average of individual APK scores
        apk1: float = apk([1, 2, 3], [1, 6, 2], 3)  # (1/1 + 2/3) / 3
        apk2: float = apk([4, 5], [4, 7], 3)  # (1/1) / 2
        expected: float = (apk1 + apk2) / 2.0
        assert abs(result - expected) < 1e-10

    def test_mapk_different_k_values(self) -> None:
        # Given multiple queries and small k value
        actual: List[List[int]] = [[1, 2, 3], [4, 5, 6]]
        predicted: List[List[int]] = [[3, 1, 2], [6, 4, 5]]
        k: int = 1

        # When computing mean average precision at k=1
        result: float = mapk(actual, predicted, k)

        # Then mean precision should consider only first prediction per query
        apk1: float = apk([1, 2, 3], [3], 1)  # 1/3 (3 is in actual)
        apk2: float = apk([4, 5, 6], [6], 1)  # 1/3 (6 is in actual)
        expected: float = (apk1 + apk2) / 2.0
        assert abs(result - expected) < 1e-10

    def test_mapk_empty_queries(self) -> None:
        # Given queries with empty actual lists
        actual: List[List[int]] = [[], [1, 2]]
        predicted: List[List[int]] = [[1, 2], [1, 3]]
        k: int = 2

        # When computing mean average precision at k
        result: float = mapk(actual, predicted, k)

        # Then mean precision should be (0.0 + apk_score) / 2
        apk1: float = 0.0  # Empty actual list
        apk2: float = apk([1, 2], [1, 3], 2)  # (1/1) / 2
        expected: float = (apk1 + apk2) / 2.0
        assert abs(result - expected) < 1e-10

    def test_mapk_with_strings(self) -> None:
        # Given multiple queries with string labels
        actual: List[List[str]] = [["cat", "dog"], ["bird", "fish"]]
        predicted: List[List[str]] = [["cat", "mouse"], ["bird", "snake"]]
        k: int = 2

        # When computing mean average precision at k
        result: float = mapk(actual, predicted, k)

        # Then mean precision should be average of string-based APK scores
        apk1: float = apk(["cat", "dog"], ["cat", "mouse"], 2)  # (1/1) / 2
        apk2: float = apk(["bird", "fish"], ["bird", "snake"], 2)  # (1/1) / 2
        expected: float = (apk1 + apk2) / 2.0
        assert abs(result - expected) < 1e-10

    def test_mapk_single_query(self) -> None:
        # Given single query (edge case)
        actual: List[List[int]] = [[1, 2, 3]]
        predicted: List[List[int]] = [[1, 4, 2]]
        k: int = 3

        # When computing mean average precision at k
        result: float = mapk(actual, predicted, k)

        # Then mean precision should equal single APK score
        expected: float = apk([1, 2, 3], [1, 4, 2], 3)
        assert abs(result - expected) < 1e-10
