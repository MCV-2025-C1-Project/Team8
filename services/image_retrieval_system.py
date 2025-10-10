import numpy as np
import os
import pickle
import json
from PIL import Image
from typing import List, Tuple, Dict, Any
from enum import Enum
from dataloader.dataloader import DataLoader, DatasetType, WeekFolder
from descriptors.descriptors import DescriptorMethod
from utils.measures import SimilarityMeasure
from utils.metrics import mapk
from preprocessing.color_adjustments import PreprocessingMethod


class DatasetRole(Enum):
    """Enum for dataset roles in the retrieval system."""
    INDEX = "index" # BBDD
    QUERY = "query" # QSD1_W1, QST1_W1, QSD2_W2, QST1_W2, QST2_W2


class ImageRetrievalSystem:

    def __init__(self):
        self.index_dataset = DataLoader()
        self.query_dataset = DataLoader()
        self.index_descriptors: Dict[int, np.ndarray] = {}
        self.query_descriptors: Dict[int, np.ndarray] = {}
        self.ground_truth: List[List[int]] = []

    def compute_descriptors(
        self, 
        role: DatasetRole, 
        method: DescriptorMethod, 
        bins: int = 256, 
        preprocessing: PreprocessingMethod = PreprocessingMethod.NONE,
        # Spatial parameters
        ns_blocks: List[int] = None,
        max_level: int = 2,
        **preprocessing_kwargs
    ) -> None:
        
        if role == DatasetRole.INDEX:
            loader = self.index_dataset
            target_dict = self.index_descriptors
        elif role == DatasetRole.QUERY:
            loader = self.query_dataset
            target_dict = self.query_descriptors
        else:
            raise ValueError(f"Unknown dataset role: {role}")

        target_dict.clear()
        for image_id, image, _, _ in loader.iterate_images():
            desc = method.compute(
                image, 
                bins=bins, 
                preprocessing=preprocessing,
                ns_blocks=ns_blocks,
                max_level=max_level,
                **preprocessing_kwargs
            )
            target_dict[image_id] = desc

        print(f"Computed descriptors for {len(target_dict)} images in {loader.dataset_type.name} dataset")

    def load_ground_truth(self) -> None:
        self.ground_truth = []
        for _, _, _, relationship in self.query_dataset.iterate_images():
            if relationship is not None:
                if isinstance(relationship, list):
                    self.ground_truth.append(relationship)
                else:
                    self.ground_truth.append([relationship])
            else:
                self.ground_truth.append([])
        print(f"Loaded ground truth for dataset {self.query_dataset.dataset_type.name}")

    def retrieve_similar_images(self, measure: SimilarityMeasure, k: int = 5) -> List[List[int]]:
        predictions = []

        # For each query image, find the most similar index images
        for query_id in sorted(self.query_descriptors.keys()):
            query_desc = self.query_descriptors[query_id]
            similarities = []

            for index_id, index_desc in self.index_descriptors.items():
                similarity = measure.compute(query_desc, index_desc)
                if measure.is_similarity:
                    similarities.append((-similarity, index_id))  # Negative for sorting (highest similarity first)
                elif measure.is_distance:
                    similarities.append((similarity, index_id))   # Lower distance is better
    
            similarities.sort(key=lambda x: x[0])
            top_k_ids = [index_id for _, index_id in similarities[:k]]
            predictions.append(top_k_ids)
            
        print(f"Retrieved top-{k} similar images for {len(predictions)} query images")

        return predictions

    def evaluate_map_at_k(self, predictions: List[List[int]], k: int) -> float:
        return mapk(self.ground_truth, predictions, k)

    def save_results(self, predictions: List[List[int]], method: DescriptorMethod, week_folder: WeekFolder, metrics: Dict[str, float] = None, dataset_name: str = None) -> str:
        # Determine method name based on descriptor
        if method == DescriptorMethod.LAB:
            method_name = "method1"
        elif method == DescriptorMethod.HSV:
            method_name = "method2"
        else:
            method_name = f"method_{method.value}"
        
        # Determine dataset name - use provided name or fall back to loaded dataset
        if dataset_name is None:
            if self.query_dataset.dataset_type is not None:
                dataset_name = self.query_dataset.dataset_type.value.upper()
            else:
                dataset_name = "UNKNOWN"
            
        results_dir = os.path.join("results", week_folder.value, dataset_name, method_name)
        os.makedirs(results_dir, exist_ok=True)
        
        # Save predictions as .pkl file (list of lists with integer image IDs)
        pkl_filepath = os.path.join(results_dir, "result.pkl")
        with open(pkl_filepath, 'wb') as f:
            pickle.dump(predictions, f)
        
        # Save metrics as JSON file (human-readable) only if metrics are provided
        if metrics:
            json_filepath = os.path.join(results_dir, "metrics.json")
            with open(json_filepath, 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f"Metrics saved to: {json_filepath}")
        
        print(f"Results saved to: {pkl_filepath}")
        print(f"Format: List of {len(predictions)} queries, each with K=10 best results")
        return pkl_filepath

    def run(
        self, 
        method: DescriptorMethod, 
        measure: SimilarityMeasure, 
        index_dataset: DatasetType, 
        query_dataset: DatasetType, 
        week_folder: WeekFolder,
        save_results: bool = True, 
        bins: int = 256, 
        preprocessing: PreprocessingMethod = PreprocessingMethod.NONE,
        # Spatial parameters
        ns_blocks: List[int] = None,
        max_level: int = 2,
        **preprocessing_kwargs
    ) -> Dict[str, float]:
        
        """Run retrieval with given descriptor and measure."""

        # Load datasets
        self.index_dataset.load_dataset(index_dataset)
        self.query_dataset.load_dataset(query_dataset)

        # Load ground truth
        if self.query_dataset.has_ground_truth():
            self.load_ground_truth()

        # Compute descriptors
        self.compute_descriptors(
            DatasetRole.QUERY, 
            method, 
            bins, 
            preprocessing, 
            ns_blocks=ns_blocks,
            max_level=max_level,
            **preprocessing_kwargs
        )
        self.compute_descriptors(
            DatasetRole.INDEX, 
            method, 
            bins, 
            preprocessing,
            ns_blocks=ns_blocks,
            max_level=max_level,
            **preprocessing_kwargs
        )

        # Generate predictions with K=10 for competition format
        predictions_k10 = self.retrieve_similar_images(measure, k=10)
        
        # Also compute K=5 for evaluation metrics
        predictions_k5 = [pred[:5] for pred in predictions_k10]

        # Evaluate
        if self.query_dataset.has_ground_truth():
            map_at_1 = self.evaluate_map_at_k(predictions_k5, k=1)
            map_at_5 = self.evaluate_map_at_k(predictions_k5, k=5)

            metrics = {"mAP@1": map_at_1, "mAP@5": map_at_5}
        
            if save_results:
                self.save_results(predictions_k10, method, week_folder, metrics)

            return metrics
        else:
            print("No ground truth available for evaluation")
            if save_results:
                self.save_results(predictions_k10, method, week_folder, None)
            return {}
