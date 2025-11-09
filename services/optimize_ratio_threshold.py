"""
Script to find the optimal ratio_threshold that maximizes F1 score (or mAP) for keypoint-based image retrieval.

Run this script from the project root directory:
    python services/optimize_ratio_threshold.py
"""
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
from typing import List, Dict
from dataloader.dataloader import DatasetType, WeekFolder
from services.keypoint_image_retrieval_system import KeyPointImageRetrievalSystem
from descriptors.descriptors import DescriptorMethod
from preprocessing.preprocessors import PreprocessingMethod
from utils.metrics import mapk
import cv2 as cv


def compute_precision_at_k(ground_truth: List[List[int]], predictions: List[List[int]], k: int = 5) -> float:
    """Compute precision@k for image retrieval."""
    precisions = []
    for gt, pred in zip(ground_truth, predictions):
        if len(pred) > k:
            pred = pred[:k]
        if len(gt) == 0:
            precisions.append(0.0)
            continue
        
        # Count how many predicted items are in ground truth
        hits = sum(1 for p in pred if p in gt)
        precision = hits / len(pred) if len(pred) > 0 else 0.0
        precisions.append(precision)
    
    return np.mean(precisions)


def compute_recall_at_k(ground_truth: List[List[int]], predictions: List[List[int]], k: int = 5) -> float:
    """Compute recall@k for image retrieval."""
    recalls = []
    for gt, pred in zip(ground_truth, predictions):
        if len(pred) > k:
            pred = pred[:k]
        if len(gt) == 0:
            recalls.append(0.0)
            continue
        
        # Count how many ground truth items are in predictions
        hits = sum(1 for g in gt if g in pred)
        recall = hits / len(gt) if len(gt) > 0 else 0.0
        recalls.append(recall)
    
    return np.mean(recalls)


def compute_f1_at_k(ground_truth: List[List[int]], predictions: List[List[int]], k: int = 5) -> float:
    """Compute F1@k for image retrieval."""
    precision = compute_precision_at_k(ground_truth, predictions, k)
    recall = compute_recall_at_k(ground_truth, predictions, k)
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def optimize_ratio_threshold(
    ratio_thresholds: List[float] = None,
    index_dataset: DatasetType = DatasetType.BBDD,
    query_dataset: DatasetType = DatasetType.QSD1_W4,
    local_descriptor_method: DescriptorMethod = DescriptorMethod.ORB,
    preprocessing: PreprocessingMethod = PreprocessingMethod.NONE,
    similarity_metric = cv.NORM_HAMMING,
    min_matches: int = 10,
    metric: str = "f1@5",  # Options: "f1@5", "f1@1", "map@5", "map@1"
    k: int = 5,
) -> Dict:
    """
    Find the optimal ratio_threshold that maximizes the specified metric.
    
    Args:
        ratio_thresholds: List of ratio_threshold values to test. If None, uses default range.
        metric: Metric to optimize ("f1@5", "f1@1", "map@5", "map@1")
        k: Value of k for F1@k or mAP@k
    
    Returns:
        Dictionary with optimal ratio_threshold and all results
    """
    if ratio_thresholds is None:
        # Default range: 0.4 to 0.85 in steps of 0.05
        ratio_thresholds = np.arange(0.40, 0.85, 0.05).round(2).tolist()
    
    print("=" * 70)
    print(f"OPTIMIZING RATIO_THRESHOLD FOR {metric.upper()} FOR {local_descriptor_method.name} DESCRIPTOR")
    print("=" * 70)
    print(f"Testing {len(ratio_thresholds)} values: {ratio_thresholds}")
    print()
    
    # Initialize retrieval system and compute descriptors once (expensive operation)
    print("Initializing retrieval system and computing descriptors...")
    retrieval_system = KeyPointImageRetrievalSystem()
    retrieval_system.index_dataset.load_dataset(index_dataset)
    retrieval_system.query_dataset.load_dataset(query_dataset)
    
    if retrieval_system.query_dataset.has_ground_truth():
        retrieval_system.load_ground_truth()
    
    # Compute descriptors once (this is the expensive part)
    retrieval_system.compute_keypoint_descriptors(
        local_descriptor_method,
        preprocessing
    )
    
    ground_truth = retrieval_system.ground_truth
    print("Descriptors computed. Testing different ratio_threshold values...\n")
    
    results = []
    best_score = -1
    best_ratio_threshold = None
    
    for ratio_threshold in ratio_thresholds:
        print(f"Testing ratio_threshold = {ratio_threshold:.2f}...", end=" ")
        
        # Now retrieve with current ratio_threshold (descriptors already computed)
        predictions = []
        for image_id, *_ in retrieval_system.query_dataset.iterate_images():
            image_kp, image_dsc = retrieval_system.query_descriptors[image_id]
            if image_dsc is None or len(image_dsc) == 0:
                predictions.append([-1])
                continue
            matches = retrieval_system.retrieve(
                query_keypoints=image_kp,
                query_descriptors=image_dsc,
                n=10,
                norm_type=similarity_metric,
                ratio_threshold=ratio_threshold,
                min_matches=min_matches,
            )
            predictions.append(matches)
        
        # Compute metrics
        if metric.startswith("f1"):
            score = compute_f1_at_k(ground_truth, predictions, k=k)
            precision = compute_precision_at_k(ground_truth, predictions, k=k)
            recall = compute_recall_at_k(ground_truth, predictions, k=k)
            result = {
                "ratio_threshold": ratio_threshold,
                "f1": score,
                "precision": precision,
                "recall": recall,
                "map@1": mapk(ground_truth, predictions, k=1),
                "map@5": mapk(ground_truth, predictions, k=5),
            }
        elif metric.startswith("map"):
            score = mapk(ground_truth, predictions, k=k)
            precision = compute_precision_at_k(ground_truth, predictions, k=k)
            recall = compute_recall_at_k(ground_truth, predictions, k=k)
            result = {
                "ratio_threshold": ratio_threshold,
                f"map@{k}": score,
                "precision": precision,
                "recall": recall,
                "f1@5": compute_f1_at_k(ground_truth, predictions, k=5),
                "map@1": mapk(ground_truth, predictions, k=1),
                "map@5": mapk(ground_truth, predictions, k=5),
            }
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        results.append(result)
        
        # Track best
        if score > best_score:
            best_score = score
            best_ratio_threshold = ratio_threshold
        
        print(f"{metric.upper()}={score:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")
    
    print()
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"Best ratio_threshold: {best_ratio_threshold:.2f}")
    print(f"Best {metric.upper()}: {best_score:.4f}")
    print()
    
    # Sort results by score
    results_sorted = sorted(results, key=lambda x: x.get("f1", x.get(f"map@{k}", 0)), reverse=True)
    
    print("Top 5 configurations:")
    for i, result in enumerate(results_sorted[:5], 1):
        ratio = result["ratio_threshold"]
        if metric.startswith("f1"):
            score = result["f1"]
        else:
            score = result[f"map@{k}"]
        print(f"  {i}. ratio_threshold={ratio:.2f}: {metric.upper()}={score:.4f}, "
              f"Precision={result['precision']:.4f}, Recall={result['recall']:.4f}")
    
    return {
        "best_ratio_threshold": best_ratio_threshold,
        "best_score": best_score,
        "best_metric": metric,
        "all_results": results,
    }


if __name__ == "__main__":
    # Optimize for F1@5 (recommended for balanced precision/recall)
    results_orb = optimize_ratio_threshold(
        ratio_thresholds=None,  # Will use default range [0.4, 0.85] in steps of 0.05
        index_dataset=DatasetType.BBDD,
        query_dataset=DatasetType.QSD1_W4,
        local_descriptor_method=DescriptorMethod.ORB,
        similarity_metric=cv.NORM_HAMMING,
        min_matches=10,
        metric="f1@5",  # Optimize for F1@5
        k=5,
    )
    
    results_sift = optimize_ratio_threshold(
        ratio_thresholds=None,  # Will use default range [0.4, 0.85] in steps of 0.05
        index_dataset=DatasetType.BBDD,
        query_dataset=DatasetType.QSD1_W4,
        local_descriptor_method=DescriptorMethod.SIFT,
        similarity_metric=cv.NORM_L2,
        min_matches=10,
        metric="f1@5",  # Optimize for F1@5
        k=5,
    )
    
    print("\n" + "=" * 70)
    print("OPTIMIZATION COMPLETE")
    print("=" * 70)
    print(f"Use ratio_threshold={results_orb['best_ratio_threshold']:.2f} in main_w4.py")
    print(f"Use ratio_threshold={results_sift['best_ratio_threshold']:.2f} in main_w4.py")

