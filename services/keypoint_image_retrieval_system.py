from dataloader.dataloader import DataLoader, DatasetType, WeekFolder
from preprocessing.preprocessors import PreprocessingMethod
from descriptors.descriptors import DescriptorMethod
from utils.metrics import kp_mapk

import cv2 as cv
from tqdm import tqdm as TQDM
from typing import Dict, List
import numpy as np
import os
import pickle
import json


class KeyPointImageRetrievalSystem:
    def __init__(self):
        self.index_dataset = DataLoader()
        self.query_dataset = DataLoader()

        self.index_descriptors: Dict[int, tuple] = {} # image_id -> (keypoints, descriptors)
        self.query_descriptors: Dict[int, tuple] = {} # image_id -> (keypoints, descriptors)

        self.ground_truth = []

    def load_ground_truth(self) -> None:
        """Load ground truth correspondences for QSD2_W2."""
        self.ground_truth = []
        for *_, relationship in self.query_dataset.iterate_images():
            if relationship is not None:
                if isinstance(relationship, list):
                    self.ground_truth.append(relationship)
                else:
                    self.ground_truth.append([relationship])
            else:
                self.ground_truth.append([])
        print(f"Loaded ground truth for {len(self.ground_truth)} query images")

    def compute_keypoint_descriptors(
        self, 
        method: DescriptorMethod,
        preprocessing: PreprocessingMethod = PreprocessingMethod.NONE,
        **preprocessing_kwargs
    ):
        self.index_descriptors.clear()
        self.query_descriptors.clear()

        # Query descriptors
        progress_bar = TQDM(
            self.query_dataset.iterate_images(),
            total=len(self.query_dataset.data),
            desc="Computing query descriptors"
        )

        for image_id, image, *_ in progress_bar:
            keypoints, descriptors = method.compute(
                image,
                preprocessing=preprocessing,
                **preprocessing_kwargs
            )
            self.query_descriptors[image_id] = (keypoints, descriptors)

        # Index descriptors
        progress_bar = TQDM(
            self.index_dataset.iterate_images(),
            total=len(self.index_dataset.data),
            desc="Computing index descriptors"
        )

        for image_id, image, *_ in progress_bar:
            keypoints, descriptors = method.compute(
                image,
                preprocessing=preprocessing,
                **preprocessing_kwargs
            )
            self.index_descriptors[image_id] = (keypoints, descriptors)
    
    def retrieve(
            self,
            query_keypoints,
            query_descriptors,
            n=5, # images to retrieve
            norm_type=cv.NORM_HAMMING,
            ratio_threshold=0.75,
            min_matches=10,
        ):
        matcher = cv.BFMatcher(normType=norm_type)

        # Early return if query descriptors are None or empty (should be handled in run, but safety check)
        if query_descriptors is None or len(query_descriptors) == 0:
            return [-1]

        results = []
        for image_id, *_ in self.index_dataset.iterate_images():
            keypoints, descriptors = self.index_descriptors[image_id]

            # Skip this index image if its descriptors are None or empty
            if descriptors is None or len(descriptors) == 0:
                continue
            
            # Handle dtype mismatch (ORB produces uint8, other methods produce float32)
            # Create local copies to avoid modifying the original query_descriptors
            current_descriptors = descriptors
            current_query_descriptors = query_descriptors
            if current_descriptors.dtype != current_query_descriptors.dtype:
                # ORB produces uint8, other methods produce float32
                current_descriptors = current_descriptors.astype(np.float32)
                current_query_descriptors = current_query_descriptors.astype(np.float32)
            
            # Skip if descriptor dimensions don't match
            if current_descriptors.shape[1] != current_query_descriptors.shape[1]:
                continue
            
            # Match
            knn_matches = matcher.knnMatch(current_query_descriptors, current_descriptors, k=2)
            if len(knn_matches) == 0:
                continue

            # Apply Lowe's ratio test to filter out poor matches
            # https://stackoverflow.com/questions/51197091/how-does-the-lowes-ratio-test-work
            good_matches = []
            for match_pair in knn_matches:
                if len(match_pair) < 2:
                    continue
                m, n_match = match_pair[0], match_pair[1]
                if m.distance < ratio_threshold * n_match.distance:
                    good_matches.append(m)
            if len(good_matches) < min_matches:
                continue

            # Verify spatial consistency using homography
            src_pts = np.float32(
                [query_keypoints[m.queryIdx].pt for m in good_matches]
            ).reshape(-1, 1, 2)
            dst_pts = np.float32(
                [keypoints[m.trainIdx].pt for m in good_matches]
            ).reshape(-1, 1, 2)
            H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
            if H is None:
                continue
            n_inliers = int(mask.sum())
            matches_mask = mask.ravel().tolist()
            inlier_ratio = sum(matches_mask) / len(good_matches) if len(good_matches) > 0 else 0
            
            # Score: Weighted combination of quantity (n_inliers) and quality (inlier_ratio)
            # This balances both the number of matches and their geometric consistency
            # - n_inliers: Primary metric (more matches = better)
            # - (1 + inlier_ratio): Boosts score for high-quality matches (ratio 0.0-1.0 becomes 1.0-2.0 multiplier)
            score = n_inliers * (1 + inlier_ratio)

            # Keep results
            results.append({
                "image_id": image_id,
                "num_good_matches": len(good_matches),
                "n_inliers": n_inliers,
                "inlier_ratio": inlier_ratio,
                "score": score,
                "homography": H,
                "good_matches": good_matches,
                "centroid": np.mean(src_pts, axis=0).flatten().tolist(),
            })

        # Sort results by score (descending) - higher score = better match
        results.sort(key=lambda x: x["score"], reverse=True)
        results = results[:n]  # Keep top n results
        
        # Return top n image IDs, or [-1] if no matches found
        if len(results) == 0:
            return [-1]
        
        # If 2 images, sort them by position of centroid (left to right / top to bottom)
        if len(results) == 2:
            x_diff = abs(results[0]["centroid"][0] - results[1]["centroid"][0])
            y_diff = abs(results[0]["centroid"][1] - results[1]["centroid"][1])
            if x_diff >= y_diff:
                # Sort by x coordinate
                results.sort(key=lambda x: x["centroid"][0])
            else:
                # Sort by y coordinate
                results.sort(key=lambda x: x["centroid"][1])
        
        # Return top n image IDs
        top_n_ids = [result["image_id"] for result in results]

        return top_n_ids
    
    def evaluate_map_at_k(self, predictions: List[List[int]], k: int) -> float:
        """Evaluate mean Average Precision at k."""
        return kp_mapk(self.ground_truth, predictions, k)
    
    def save_results(
        self,
        predictions: List[List[int]],
        method: DescriptorMethod,
        week_folder: WeekFolder,
        metrics: Dict[str, float] = None,
        dataset_name: str = None
    ) -> str:
        """Save retrieval results to files."""
        # Determine method name
        method_name = f"method_{method.value}"
        
        # Determine dataset name
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
        local_descriptor_method: DescriptorMethod,
        index_dataset: DatasetType,
        query_dataset: DatasetType,
        week_folder: WeekFolder,
        preprocessing: PreprocessingMethod = PreprocessingMethod.NONE,
        save_results: bool = True,
        similarity_metric=cv.NORM_HAMMING,
        ratio_threshold=0.75,
        min_matches=10,
        **preprocessing_kwargs
    ):
        
        # Load datasets
        self.index_dataset.load_dataset(index_dataset)
        self.query_dataset.load_dataset(query_dataset)

        # Only load ground truth if the dataset has it
        if self.query_dataset.has_ground_truth():
            self.load_ground_truth()
        
        # Descriptors
        self.compute_keypoint_descriptors(
            local_descriptor_method,
            preprocessing=preprocessing,
            **preprocessing_kwargs
        )

        # Retrieve for all query images
        predictions = []
        progress_bar = TQDM(
            self.query_dataset.iterate_images(),
            total=len(self.query_dataset.data),
            desc="Retrieving matches"
        )
        
        for image_id, *_ in progress_bar:
            image_kp, image_dsc = self.query_descriptors[image_id]
            
            # Handle case where descriptors might be None (no keypoints detected)
            if image_dsc is None or len(image_dsc) == 0:
                predictions.append([-1])
                continue
            
            matches = self.retrieve(
                query_keypoints=image_kp,
                query_descriptors=image_dsc,
                n=2,  # Use K=10 for competition format
                norm_type=similarity_metric,
                ratio_threshold=ratio_threshold,
                min_matches=min_matches,
            )
            predictions.append(matches)
        
        print(f"Retrieved matches for {len(predictions)} query images")
        
        # Evaluate if ground truth is available
        metrics = None
        if self.query_dataset.has_ground_truth() and len(self.ground_truth) > 0:
            map_at_1 = self.evaluate_map_at_k(predictions, k=2)
            metrics = {"mAP@1": map_at_1}
            print(f"mAP@1: {map_at_1:.4f}")

        predictions_as_single_item_lists = [[[x] for x in sublist] for sublist in predictions]

        # Save results if requested
        if save_results:
            self.save_results(
                predictions_as_single_item_lists,
                local_descriptor_method,
                week_folder,
                metrics
            )
        
        return metrics if metrics else {}
