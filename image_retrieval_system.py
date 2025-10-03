import numpy as np
from PIL import Image
from typing import List, Tuple, Dict, Any
from enum import Enum
from dataloader.dataloader import DataLoader, DatasetType
from utils.descriptors import histogram_1_channel, histogram_3_channels
from utils.measures import hist_intersect
from utils.metrics import mapk


class DescriptorMethod(Enum):
    GRAYSCALE = "grayscale"
    RGB = "rgb"


class ImageRetrievalSystem:

    def __init__(self):
        self.dataloader = DataLoader()
        self.bbdd_descriptors: Dict[int, np.ndarray] = {}
        self.qsd1_descriptors: Dict[int, np.ndarray] = {}
        self.ground_truth: List[List[int]] = []

    def load_datasets(self) -> None:
        print("Loading BBDD dataset...")
        self.dataloader.load_dataset(DatasetType.BBDD)
        bbdd_info = self.dataloader.get_dataset_info()
        print(f"Loaded {bbdd_info['num_images']} BBDD images")

        print("Loading QSD1 dataset...")
        self.dataloader.load_dataset(DatasetType.QSD1_W1)
        qsd1_info = self.dataloader.get_dataset_info()
        print(f"Loaded {qsd1_info['num_images']} QSD1 images")

    def compute_bbdd_descriptors(self, method: DescriptorMethod) -> None:
        print(f"Computing BBDD descriptors using {method.value}...")
        self.dataloader.load_dataset(DatasetType.BBDD)

        for image_id, image, info, relationship in self.dataloader.iterate_images():
            if method == DescriptorMethod.GRAYSCALE:
                descriptor = histogram_1_channel(image)
            elif method == DescriptorMethod.RGB:
                descriptor = histogram_3_channels(image)
            else:
                raise ValueError(f"Unknown method: {method}")

            self.bbdd_descriptors[image_id] = descriptor

        print(f"Computed {len(self.bbdd_descriptors)} BBDD descriptors")

    def compute_qsd1_descriptors(self, method: DescriptorMethod) -> None:
        print(f"Computing QSD1 descriptors using {method.value}...")
        self.dataloader.load_dataset(DatasetType.QSD1_W1)

        self.ground_truth = []
        for image_id, image, info, relationship in self.dataloader.iterate_images():
            if method == DescriptorMethod.GRAYSCALE:
                descriptor = histogram_1_channel(image)
            elif method == DescriptorMethod.RGB:
                descriptor = histogram_3_channels(image)
            else:
                raise ValueError(f"Unknown method: {method}")

            self.qsd1_descriptors[image_id] = descriptor

            if relationship is not None:
                if isinstance(relationship, list):
                    self.ground_truth.append(relationship)
                else:
                    self.ground_truth.append([relationship])
            else:
                self.ground_truth.append([])

        print(f"Computed {len(self.qsd1_descriptors)} QSD1 descriptors")

    def retrieve_similar_images(self, k: int = 5) -> List[List[int]]:
        predictions = []

        for qsd1_id in sorted(self.qsd1_descriptors.keys()):
            qsd1_desc = self.qsd1_descriptors[qsd1_id]
            similarities = []

            for bbdd_id, bbdd_desc in self.bbdd_descriptors.items():
                similarity = hist_intersect(qsd1_desc, bbdd_desc)
                similarities.append(
                    (-similarity, bbdd_id)
                )  # Negative for sorting (highest similarity first)

            similarities.sort(key=lambda x: x[0])
            top_k_ids = [bbdd_id for _, bbdd_id in similarities[:k]]
            predictions.append(top_k_ids)

        return predictions

    def evaluate_map_at_k(self, predictions: List[List[int]], k: int) -> float:
        return mapk(self.ground_truth, predictions, k)

    def run_evaluation(self, method: DescriptorMethod) -> Dict[str, float]:
        print(f"\n{'='*60}")
        print(f"EVALUATION: {method.value.upper()} METHOD")
        print(f"{'='*60}")

        self.bbdd_descriptors.clear()
        self.qsd1_descriptors.clear()

        self.compute_bbdd_descriptors(method)
        self.compute_qsd1_descriptors(method)

        predictions_k5 = self.retrieve_similar_images(k=5)

        map_at_1 = self.evaluate_map_at_k(predictions_k5, k=1)
        map_at_5 = self.evaluate_map_at_k(predictions_k5, k=5)

        results = {"mAP@1": map_at_1, "mAP@5": map_at_5}

        print(f"mAP@1: {map_at_1:.4f}")
        print(f"mAP@5: {map_at_5:.4f}")

        return results


def main() -> None:
    print("Image Retrieval System - QSD1 vs BBDD")
    print("=" * 60)

    retrieval_system = ImageRetrievalSystem()
    retrieval_system.load_datasets()

    method1_results = retrieval_system.run_evaluation(DescriptorMethod.GRAYSCALE)
    method2_results = retrieval_system.run_evaluation(DescriptorMethod.RGB)

    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Method 1 (Grayscale Histogram):")
    print(f"  mAP@1: {method1_results['mAP@1']:.4f}")
    print(f"  mAP@5: {method1_results['mAP@5']:.4f}")
    print(f"\nMethod 2 (RGB Histogram):")
    print(f"  mAP@1: {method2_results['mAP@1']:.4f}")
    print(f"  mAP@5: {method2_results['mAP@5']:.4f}")


if __name__ == "__main__":
    main()
