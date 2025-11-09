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
from utils.metrics import kp_mapk
import cv2 as cv
from matplotlib import pyplot as plt


def compute_precision_at_k(ground_truth: List[List[int]], predictions: List[List[int]], k: int = 2) -> float:
    """Compute precision@k for image retrieval."""
    precisions = []
    for gt, pred in zip(ground_truth, predictions):
        if len(pred) > k:
            pred = pred[:k]
        # If ground-truth is empty, treat as no relevant items
        if not gt:
            precisions.append(0.0)
            continue

        # Fractional hit: number of matched ids divided by the maximum cardinality
        # between prediction and ground truth. This makes a full match count as
        # 1.0, a single match when one side has two items count as 0.5, etc.
        matched = len(set(pred) & set(gt))
        denom = max(len(pred) if len(pred) > 0 else 1, len(gt))
        precisions.append(matched / denom)

    return np.mean(precisions)


def compute_recall_at_k(ground_truth: List[List[int]], predictions: List[List[int]], k: int = 2) -> float:
    """Compute recall@k for image retrieval."""
    recalls = []
    for gt, pred in zip(ground_truth, predictions):
        if len(pred) > k:
            pred = pred[:k]
        # If ground-truth is empty, treat as no relevant items
        if not gt:
            recalls.append(0.0)
            continue

        matched = len(set(pred) & set(gt))
        denom = max(len(pred) if len(pred) > 0 else 1, len(gt))
        recalls.append(matched / denom)

    return np.mean(recalls)


def compute_f1_at_k(ground_truth: List[List[int]], predictions: List[List[int]], k: int = 2) -> float:
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
    metric: str = "f1@1",  # Options: "f1@5"
    k: int = 2,
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
        # Default range: 0.4 to 0.9 in steps of 0.05
        ratio_thresholds = np.arange(0.40, 0.95, 0.05).round(2).tolist()
    
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
        for image_id, image, *_ in retrieval_system.query_dataset.iterate_images():
            image_kp, image_dsc = retrieval_system.query_descriptors[image_id]
            if image_dsc is None or len(image_dsc) == 0:
                predictions.append([-1])
                continue
            matches = retrieval_system.retrieve(
                query_keypoints=image_kp,
                query_descriptors=image_dsc,
                query_image=image,
                n=2,
                norm_type=similarity_metric,
                ratio_threshold=ratio_threshold,
                min_matches=min_matches,
            )
            predictions.append(matches)
        
        # Compute metrics
        if metric.startswith("f1"):
            # Optimize F1 at provided k (e.g., f1@1 means k=1)
            score = compute_f1_at_k(ground_truth, predictions, k=k)
            precision = compute_precision_at_k(ground_truth, predictions, k=k)
            recall = compute_recall_at_k(ground_truth, predictions, k=k)
            # For keypoint-aware mean AP, use kp_mapk. To preserve the "map@1"
            # semantics used in the rest of the codebase we call kp_mapk with k=2
            # (queries with up to 2 correct painting ids are considered a single
            # example for map@1 evaluation).
            result = {
                "ratio_threshold": ratio_threshold,
                "f1": score,
                "precision": precision,
                "recall": recall,
                "map@1": kp_mapk(ground_truth, predictions, k=2),
                "map@5": kp_mapk(ground_truth, predictions, k=5),
            }
        elif metric.startswith("map"):
            # Use keypoint-aware map when evaluating map metrics as well
            score = kp_mapk(ground_truth, predictions, k=k if k > 1 else 2)
            precision = compute_precision_at_k(ground_truth, predictions, k=k)
            recall = compute_recall_at_k(ground_truth, predictions, k=k)
            result = {
                "ratio_threshold": ratio_threshold,
                f"map@{k}": score,
                "precision": precision,
                "recall": recall,
                "f1@5": compute_f1_at_k(ground_truth, predictions, k=5),
                # report both map@1 and map@5 using keypoint-aware kp_mapk
                "map@1": kp_mapk(ground_truth, predictions, k=2),
                "map@5": kp_mapk(ground_truth, predictions, k=5),
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
        ratio_thresholds=None,  # Will use default range [0.40, 0.85] in steps of 0.05
        index_dataset=DatasetType.BBDD,
        query_dataset=DatasetType.QSD1_W4,
        local_descriptor_method=DescriptorMethod.ORB,
        preprocessing=PreprocessingMethod.GAUSSIAN,
        similarity_metric=cv.NORM_HAMMING,
        min_matches=10,
        metric="f1@1",  # Optimize for F1@1
        k=2,
    )
    
    results_sift = optimize_ratio_threshold(
        ratio_thresholds=None,  # Will use default range [0.4, 0.85] in steps of 0.05
        index_dataset=DatasetType.BBDD,
        query_dataset=DatasetType.QSD1_W4,
        preprocessing=PreprocessingMethod.GAUSSIAN,
        local_descriptor_method=DescriptorMethod.SIFT,
        similarity_metric=cv.NORM_L2,
        min_matches=10,
        metric="f1@1",  # Optimize for F1@1
        k=2,
    )
    
    print("\n" + "=" * 70)
    print("OPTIMIZATION COMPLETE")
    print("=" * 70)
    print(f"Use ratio_threshold={results_orb['best_ratio_threshold']:.2f} in main_w4.py")
    print(f"Use ratio_threshold={results_sift['best_ratio_threshold']:.2f} in main_w4.py")

    # Plot F1 vs ratio_threshold for ORB
    try:
        orb_thresholds = [r['ratio_threshold'] for r in results_orb['all_results']]
        orb_f1 = [r.get('f1', 0.0) for r in results_orb['all_results']]
        plt.figure(figsize=(8, 4))
        plt.plot(orb_thresholds, orb_f1, marker='o')
        plt.title('ORB: F1 vs Lowe ratio threshold')
        plt.xlabel('ratio_threshold')
        plt.ylabel('F1')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    except Exception:
        print('Could not plot ORB results (missing data).')

    # Plot F1 vs ratio_threshold for SIFT
    try:
        sift_thresholds = [r['ratio_threshold'] for r in results_sift['all_results']]
        sift_f1 = [r.get('f1', 0.0) for r in results_sift['all_results']]
        plt.figure(figsize=(8, 4))
        plt.plot(sift_thresholds, sift_f1, marker='o', color='orange')
        plt.title('SIFT: F1 vs Lowe ratio threshold')
        plt.xlabel('ratio_threshold')
        plt.ylabel('F1')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    except Exception:
        print('Could not plot SIFT results (missing data).')

