"""
Background Removal + Image Retrieval System for QSD2_W2.

This service combines background removal preprocessing with image retrieval,
specifically designed for the QSD2_W2 dataset workflow.
"""

import numpy as np
import os
import pickle
import json
from typing import List, Dict, Callable, Optional
from dataloader.dataloader import DataLoader, DatasetType, WeekFolder
from descriptors.descriptors import DescriptorMethod
from utils.measures import SimilarityMeasure
from utils.metrics import mapk, evaluate_background_removal
from preprocessing.preprocessors import PreprocessingMethod


class BackgroundRemovalImageRetrievalSystem:
    """
    Combined system for background removal + image retrieval on QSD2_W2 dataset.
    
    Workflow:
    1. Load QSD2_W2 (query) and BBDD (index) datasets
    2. Apply background removal to QSD2_W2 images
    3. Compute descriptors on both datasets
    4. Perform image retrieval
    5. Evaluate retrieval performance
    6. Optionally evaluate background removal quality
    """

    def __init__(self):
        self.index_dataset = DataLoader()  # BBDD
        self.query_dataset = DataLoader()  # QSD2_W2
        self.index_descriptors: Dict[int, np.ndarray] = {}
        self.query_descriptors: Dict[int, np.ndarray] = {}
        self.ground_truth: List[List[int]] = []
        self.background_removal_function: Optional[Callable] = None
        self.background_removal_kwargs: Dict = {}
        self.predicted_masks: Dict[int, np.ndarray] = {}  # Store predicted masks

    def set_background_removal_method(
        self, 
        preprocessing: PreprocessingMethod, 
        **kwargs
    ) -> None:
        """
        Set the background removal method and its parameters.
        """
        if not preprocessing.is_background_removal:
            raise ValueError(f"Preprocessing method {preprocessing} is not a background removal method")
        
        self.background_removal_function = preprocessing.apply
        self.background_removal_kwargs = kwargs
        print(f"Background removal method set: {preprocessing.value}")

    def background_removal_preprocess_images(self, dataset_type: str = "query") -> None:
        """
        Apply background removal to images in the specified dataset.
        This stores the predicted masks for evaluation and creates background-removed images for descriptors.
        """
        if self.background_removal_function is None:
            raise ValueError("Background removal method not set. Call set_background_removal_method() first.")

        if dataset_type == "query":
            dataset = self.query_dataset
            print("Applying background removal to query images...")
        else:
            dataset = self.index_dataset
            print("Applying background removal to index images...")
        
        processed_count = 0
        for image_id, image, info, relationship in dataset.iterate_images():
            # Apply background removal to get the mask
            predicted_mask = self.background_removal_function(image, **self.background_removal_kwargs)
            
            # Store the predicted mask for evaluation (query only)
            if dataset_type == "query":
                self.predicted_masks[image_id] = predicted_mask
            
            # Apply the mask to the original image to create background-removed image
            if len(predicted_mask.shape) == 2:  # Binary mask
                # Convert mask to 3-channel for broadcasting
                mask_3d = np.stack([predicted_mask] * 3, axis=2)
                # Normalize mask to 0-1 range
                mask_normalized = mask_3d.astype(np.float32) / 255.0
                # Apply mask to original image
                background_removed_image = (image.astype(np.float32) * mask_normalized).astype(np.uint8)
            else:
                # If mask is already 3-channel, use it directly
                background_removed_image = predicted_mask
            
            # Update the dataset with background-removed image
            if image_id in dataset.data:
                dataset.data[image_id]["image"] = background_removed_image
                processed_count += 1
        
        print(f"Background removal applied to {processed_count} {dataset_type} images")

    def compute_descriptors(
        self, 
        method: DescriptorMethod, 
        bins: int = 256, 
        preprocessing: PreprocessingMethod = PreprocessingMethod.NONE,
        ns_blocks: List[int] = None,
        max_level: int = 2,
        **preprocessing_kwargs
    ) -> None:
        """Compute descriptors for both index and query datasets."""
        
        self.index_descriptors.clear()
        self.query_descriptors.clear()
        
        print("Computing query descriptors...")
        for image_id, image, _, _ in self.query_dataset.iterate_images():
            desc = method.compute(
                image, 
                bins=bins, 
                preprocessing=preprocessing,
                ns_blocks=ns_blocks,
                max_level=max_level,
                **preprocessing_kwargs
            )
            self.query_descriptors[image_id] = desc

        print("Computing index descriptors...")
        for image_id, image, _, _ in self.index_dataset.iterate_images():
            desc = method.compute(
                image, 
                bins=bins, 
                preprocessing=preprocessing,
                ns_blocks=ns_blocks,
                max_level=max_level,
                **preprocessing_kwargs
            )
            self.index_descriptors[image_id] = desc

        print(f"Computed descriptors for {len(self.query_descriptors)} query and {len(self.index_descriptors)} index images")

    def load_ground_truth(self) -> None:
        """Load ground truth correspondences for QSD2_W2."""
        self.ground_truth = []
        for _, _, _, relationship in self.query_dataset.iterate_images():
            if relationship is not None:
                if isinstance(relationship, list):
                    self.ground_truth.append(relationship)
                else:
                    self.ground_truth.append([relationship])
            else:
                self.ground_truth.append([])
        print(f"Loaded ground truth for {len(self.ground_truth)} query images")

    def retrieve_similar_images(self, measure: SimilarityMeasure, k: int = 5) -> List[List[int]]:
        """Retrieve similar images using the computed descriptors."""
        predictions = []

        for query_id in sorted(self.query_descriptors.keys()):
            query_desc = self.query_descriptors[query_id]
            similarities = []

            for index_id, index_desc in self.index_descriptors.items():
                similarity = measure.compute(query_desc, index_desc)
                if measure.is_similarity:
                    similarities.append((-similarity, index_id))
                elif measure.is_distance:
                    similarities.append((similarity, index_id))
    
            similarities.sort(key=lambda x: x[0])
            top_k_ids = [index_id for _, index_id in similarities[:k]]
            predictions.append(top_k_ids)
            
        print(f"Retrieved top-{k} similar images for {len(predictions)} query images")
        return predictions

    def evaluate_retrieval_performance(self, predictions: List[List[int]]) -> Dict[str, float]:
        """Evaluate retrieval performance using mAP metrics."""
        if not self.ground_truth:
            print("No ground truth available for evaluation")
            return {}
        
        # Evaluate with K=5 for metrics
        predictions_k5 = [pred[:5] for pred in predictions]
        map_at_1 = mapk(self.ground_truth, predictions_k5, k=1)
        map_at_5 = mapk(self.ground_truth, predictions_k5, k=5)
        
        return {"mAP@1": map_at_1, "mAP@5": map_at_5}

    def evaluate_background_removal_quality(self) -> Dict[str, float]:
        """Evaluate background removal quality using already computed predictions."""
        if not self.predicted_masks:
            print("No predicted masks available for evaluation")
            return {}
        
        try:
            # Get ground truth masks and use already computed predicted masks
            ground_truth_masks = []
            predicted_masks = []
            
            for image_id, image, info, relationship in self.query_dataset.iterate_images():
                # Get ground truth mask (PNG file)
                image_data = self.query_dataset.get_image_by_id(image_id)
                if image_data and "background_removed" in image_data:
                    gt_mask = image_data["background_removed"]
                    if gt_mask is not None and image_id in self.predicted_masks:
                        ground_truth_masks.append(gt_mask)
                        predicted_masks.append(self.predicted_masks[image_id])
            
            if not ground_truth_masks:
                print("No ground truth masks found for evaluation")
                return {}
            
            return evaluate_background_removal(ground_truth_masks, predicted_masks)
            
        except Exception as e:
            print(f"Error evaluating background removal: {e}")
            return {}

    def save_results(
        self, 
        predictions: List[List[int]], 
        method: DescriptorMethod, 
        week_folder: WeekFolder,
        retrieval_metrics: Dict[str, float] = None,
        bg_removal_metrics: Dict[str, float] = None,
        dataset_name: str = "QSD2_W2"
    ) -> str:
        """Save results to files."""
        
        method_name = f"method_{method.value}_bg_removal"
        results_dir = os.path.join("results", week_folder.value, dataset_name, method_name)
        os.makedirs(results_dir, exist_ok=True)
        
        # Save predictions
        pkl_filepath = os.path.join(results_dir, "result.pkl")
        with open(pkl_filepath, 'wb') as f:
            pickle.dump(predictions, f)
        
        if retrieval_metrics:
            retrieval_filepath = os.path.join(results_dir, "retrieval_metrics.json")
            with open(retrieval_filepath, 'w') as f:
                json.dump(retrieval_metrics, f, indent=2)
            print(f"Retrieval metrics saved to: {retrieval_filepath}")
        
        if bg_removal_metrics:
            bg_filepath = os.path.join(results_dir, "background_removal_metrics.json")
            with open(bg_filepath, 'w') as f:
                json.dump(bg_removal_metrics, f, indent=2)
            print(f"Background removal metrics saved to: {bg_filepath}")
        
        print(f"Results saved to: {pkl_filepath}")
        return pkl_filepath

    def run(
        self,
        method: DescriptorMethod,
        measure: SimilarityMeasure,
        index_dataset: DatasetType,
        query_dataset: DatasetType,
        week_folder: WeekFolder,
        save_results: bool = True,
        evaluate_bg_removal: bool = True,
        bins: int = 256,
        preprocessing: PreprocessingMethod = PreprocessingMethod.BG_RECTANGLES,
        ns_blocks: List[int] = None,
        max_level: int = 2,
        **preprocessing_kwargs
    ) -> Dict[str, float]:
        """
        Run the complete background removal + image retrieval pipeline.
        """
                
        # Set background removal method from preprocessing
        self.set_background_removal_method(preprocessing, **preprocessing_kwargs)
        
        self.index_dataset.load_dataset(index_dataset)
        self.query_dataset.load_dataset(query_dataset)
        
        self.load_ground_truth()
        
        # Apply background removal to both query and index datasets
        self.background_removal_preprocess_images("query")
        self.background_removal_preprocess_images("index")
        
        self.compute_descriptors(
            method, 
            bins, 
            PreprocessingMethod.NONE,  # No additional preprocessing since we already did background removal
            ns_blocks=ns_blocks,
            max_level=max_level
        )
        
        predictions = self.retrieve_similar_images(measure, k=10)
        
        print("\n📊 Evaluating retrieval performance...")
        retrieval_metrics = self.evaluate_retrieval_performance(predictions)
        
        bg_removal_metrics = {}
        if evaluate_bg_removal:
            print("\n🎯 Evaluating background removal quality...")
            bg_removal_metrics = self.evaluate_background_removal_quality()
        
        # Save results
        if save_results:
            self.save_results(
                predictions, 
                method, 
                week_folder,
                retrieval_metrics,
                bg_removal_metrics
            )
        
        # Combine all metrics
        all_metrics = {**retrieval_metrics, **bg_removal_metrics}
        
        print("\n✅ Pipeline completed successfully!")
        return all_metrics
