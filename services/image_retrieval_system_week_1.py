import numpy as np
import os
import pickle
import json
from PIL import Image
from typing import List, Tuple, Dict, Any
from enum import Enum
from dataloader.dataloader import DataLoader, DatasetType
from utils.descriptors import histogram_lab, histogram_hsv
from utils.measures import hist_intersect
from utils.metrics import mapk


class DescriptorMethod(Enum):
    LAB = "lab"
    HSV = "hsv"


class ImageRetrievalSystem:

    def __init__(self):
        self.bbdd_loader = DataLoader()
        self.qsd1_loader = DataLoader()
        self.bbdd_descriptors: Dict[int, np.ndarray] = {}
        self.qsd1_descriptors: Dict[int, np.ndarray] = {}
        self.ground_truth: List[List[int]] = []

    def compute_bbdd_descriptors(self, method: DescriptorMethod) -> None:
        print(f"Computing BBDD descriptors using {method.value}...")
        self.bbdd_loader.load_dataset(DatasetType.BBDD)

        for image_id, image, info, relationship in self.bbdd_loader.iterate_images():
            if method == DescriptorMethod.LAB:
                descriptor = histogram_lab(image)
            elif method == DescriptorMethod.HSV:
                descriptor = histogram_hsv(image)
            else:
                raise ValueError(f"Unknown method: {method}")

            self.bbdd_descriptors[image_id] = descriptor

        print(f"Computed {len(self.bbdd_descriptors)} BBDD descriptors")

    def compute_qsd1_descriptors(self, method: DescriptorMethod) -> None:
        print(f"Computing QSD1 descriptors using {method.value}...")
        self.qsd1_loader.load_dataset(DatasetType.QSD1_W1)

        self.ground_truth = []
        for image_id, image, info, relationship in self.qsd1_loader.iterate_images():
            if method == DescriptorMethod.LAB:
                descriptor = histogram_lab(image)
            elif method == DescriptorMethod.HSV:
                descriptor = histogram_hsv(image)
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

    def save_results(self, predictions: List[List[int]], method: DescriptorMethod, metrics: Dict[str, float] = None) -> str:
        if method == DescriptorMethod.LAB:
            method_name = "method1"
        elif method == DescriptorMethod.HSV:
            method_name = "method2"
        else:
            raise ValueError(f"Unknown method: {method}")
            
        results_dir = os.path.join("results", "week1", "QST1", method_name)
        os.makedirs(results_dir, exist_ok=True)
        
        # Save predictions as .pkl file (list of lists with integer image IDs)
        pkl_filepath = os.path.join(results_dir, "result.pkl")
        with open(pkl_filepath, 'wb') as f:
            pickle.dump(predictions, f)
        
        # Save metrics as JSON file (human-readable)
        if metrics:
            json_filepath = os.path.join(results_dir, "metrics.json")
            with open(json_filepath, 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f"Metrics saved to: {json_filepath}")
        
        print(f"Results saved to: {pkl_filepath}")
        print(f"Format: List of {len(predictions)} queries, each with K=10 best results")
        return pkl_filepath

    def run_evaluation(self, method: DescriptorMethod, save_results: bool = True) -> Dict[str, float]:
        """Run complete evaluation for a method."""
        print(f"\n{'='*60}")
        print(f"EVALUATION: {method.value.upper()} METHOD")
        print(f"{'='*60}")

        self.bbdd_descriptors.clear()
        self.qsd1_descriptors.clear()

        self.compute_bbdd_descriptors(method)
        self.compute_qsd1_descriptors(method)

        # Generate predictions with K=10 for competition format
        predictions_k10 = self.retrieve_similar_images(k=10)
        
        # Also compute K=5 for evaluation metrics
        predictions_k5 = [pred[:5] for pred in predictions_k10]

        map_at_1 = self.evaluate_map_at_k(predictions_k5, k=1)
        map_at_5 = self.evaluate_map_at_k(predictions_k5, k=5)

        results = {"mAP@1": map_at_1, "mAP@5": map_at_5}

        print(f"mAP@1: {map_at_1:.4f}")
        print(f"mAP@5: {map_at_5:.4f}")
        
        if save_results:
            self.save_results(predictions_k10, method, results)

        return results

