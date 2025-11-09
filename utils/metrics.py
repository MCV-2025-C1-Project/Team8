"""
From: https://github.com/benhamner/Metrics
"""

import numpy as np
from typing import List, Union, Dict


def apk(
    actual: List[Union[int, str]], predicted: List[Union[int, str]], k: int = 10
) -> float:
    """
    Computes the average precision at k.

    This function computes the average prescision at k between two lists of
    items.
    """
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mapk(
    actual: List[List[Union[int, str]]],
    predicted: List[List[Union[int, str]]],
    k: int = 10,
) -> float:
    """
    Computes the mean average precision at k.

    This function computes the mean average prescision at k between two lists
    of lists of items.
    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


def kp_apk(
    actual: List[Union[int, str]], predicted: List[Union[int, str]], k: int = 10
) -> float:
    """
    Computes the average precision at k.

    This function computes the average prescision at k between two lists of
    items.
    """
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / max(len(actual), len(predicted), 1)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def kp_mapk(
    actual: List[List[Union[int, str]]],
    predicted: List[List[Union[int, str]]],
    k: int = 10,
) -> float:
    """
    Computes the mean average precision at k (keypoint version).

    This function computes the mean average prescision at k between two lists
    of lists of items.
    """
    return np.mean([kp_apk(a, p, k) for a, p in zip(actual, predicted)])

# Binary Mask Evaluation Metrics

def precision_binary_mask(ground_truth: np.ndarray, predicted: np.ndarray) -> float:
    """
    Compute precision for binary mask evaluation.
    
    Precision = True Positives / (True Positives + False Positives)
    """
    # Ensure binary masks
    gt_binary = (ground_truth > 0).astype(np.uint8)
    pred_binary = (predicted > 0).astype(np.uint8)
    
    # True Positives: pixels correctly predicted as foreground
    tp = np.sum((gt_binary == 1) & (pred_binary == 1))
    
    # False Positives: pixels incorrectly predicted as foreground
    fp = np.sum((gt_binary == 0) & (pred_binary == 1))
    
    # Avoid division by zero
    if (tp + fp) == 0:
        return 1.0 if tp == 0 else 0.0
    
    return tp / (tp + fp)


def recall_binary_mask(ground_truth: np.ndarray, predicted: np.ndarray) -> float:
    """
    Compute recall for binary mask evaluation.
    
    Recall = True Positives / (True Positives + False Negatives)
    """
    # Ensure binary masks
    gt_binary = (ground_truth > 0).astype(np.uint8)
    pred_binary = (predicted > 0).astype(np.uint8)
    
    # True Positives: pixels correctly predicted as foreground
    tp = np.sum((gt_binary == 1) & (pred_binary == 1))
    
    # False Negatives: pixels incorrectly predicted as background
    fn = np.sum((gt_binary == 1) & (pred_binary == 0))
    
    # Avoid division by zero
    if (tp + fn) == 0:
        return 1.0 if tp == 0 else 0.0
    
    return tp / (tp + fn)


def f1_binary_mask(ground_truth: np.ndarray, predicted: np.ndarray) -> float:
    """
    Compute F1-score for binary mask evaluation.
    
    F1 = 2 * (Precision * Recall) / (Precision + Recall)
    """
    prec = precision_binary_mask(ground_truth, predicted)
    rec = recall_binary_mask(ground_truth, predicted)
    
    # Avoid division by zero
    if (prec + rec) == 0:
        return 0.0
    
    return 2 * (prec * rec) / (prec + rec)


# Background Removal Evaluation Functions

def evaluate_single_image(ground_truth: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    """
    Evaluate background removal performance for a single image.
    """
    return {
        'precision': precision_binary_mask(ground_truth, predicted),
        'recall': recall_binary_mask(ground_truth, predicted),
        'f1': f1_binary_mask(ground_truth, predicted)
    }


def evaluate_background_removal(
    ground_truth_masks: List[np.ndarray], 
    predicted_masks: List[np.ndarray]
) -> Dict[str, float]:
    """
    Evaluate background removal performance across multiple images.
    """
    if len(ground_truth_masks) != len(predicted_masks):
        raise ValueError("Number of ground truth and predicted masks must match")
    
    precisions = []
    recalls = []
    f1_scores = []
    
    for gt_mask, pred_mask in zip(ground_truth_masks, predicted_masks):
        # Ensure same shape
        if gt_mask.shape != pred_mask.shape:
            raise ValueError(f"Mask shapes don't match: {gt_mask.shape} vs {pred_mask.shape}")
        
        prec = precision_binary_mask(gt_mask, pred_mask)
        rec = recall_binary_mask(gt_mask, pred_mask)
        f1 = f1_binary_mask(gt_mask, pred_mask)
        
        precisions.append(prec)
        recalls.append(rec)
        f1_scores.append(f1)
    
    return {
        'precision': np.mean(precisions),
        'recall': np.mean(recalls),
        'f1': np.mean(f1_scores),
        'precision_std': np.std(precisions),
        'recall_std': np.std(recalls),
        'f1_std': np.std(f1_scores)
    }