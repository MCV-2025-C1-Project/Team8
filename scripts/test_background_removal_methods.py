#!/usr/bin/env python3

"""
Background Removal Testing Script

This script tests both Rectangles and Kmeans background removal methods on a single image
from the QSD2_W2 dataset. It visualizes the results and extracts precision, recall, and F1 metrics.

Usage:
    python scripts/test_single_image.py

The script will test images 00001, 00005, and 00010 by default.
You can modify the test_images list to test different images.
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.background_removers import remove_background_by_rectangles, remove_background_by_kmeans
from utils.metrics import precision_binary_mask, recall_binary_mask, f1_binary_mask

def test_single_image(image_id="00001", dataset_path="data/qsd2_w2"):
    """Test background removal on a single image."""
    
    print("=" * 60)
    print(f"TESTING BACKGROUND REMOVAL ON IMAGE {image_id}")
    print("=" * 60)
    
    # Define paths
    image_path = os.path.join(dataset_path, f"{image_id}.jpg")
    gt_mask_path = os.path.join(dataset_path, f"{image_id}.png")
    
    # Load image and ground truth mask
    img = cv2.imread(image_path)
    gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return
    if gt_mask is None:
        print(f"Error: Could not load ground truth mask from {gt_mask_path}")
        return
    
    print(f"Image shape: {img.shape}")
    print(f"GT mask shape: {gt_mask.shape}")
    print(f"GT mask unique values: {np.unique(gt_mask)}")
    
    # Apply Rectangles background removal
    print("\nApplying Rectangles method...")
    try:
        pred_mask_rect, processed_img_rect = remove_background_by_rectangles(
            img, 
            offset=50, 
            h_delta=20, 
            s_delta=60, 
            v_delta=60, 
            visualise=False
        )
        
        # Calculate metrics for Rectangles
        precision_rect = precision_binary_mask(gt_mask, pred_mask_rect)
        recall_rect = recall_binary_mask(gt_mask, pred_mask_rect)
        f1_rect = f1_binary_mask(gt_mask, pred_mask_rect)
        
        print(f"Rectangles - Precision: {precision_rect:.3f}, Recall: {recall_rect:.3f}, F1: {f1_rect:.3f}")
        
    except Exception as e:
        print(f"Error in Rectangles method: {e}")
        return
    
    # Apply Kmeans background removal
    print("Applying Kmeans method...")
    try:
        pred_mask_kmeans, processed_img_kmeans = remove_background_by_kmeans(
            img, 
            k=5, 
            margin=45
        )
        
        # Calculate metrics for Kmeans
        precision_kmeans = precision_binary_mask(gt_mask, pred_mask_kmeans)
        recall_kmeans = recall_binary_mask(gt_mask, pred_mask_kmeans)
        f1_kmeans = f1_binary_mask(gt_mask, pred_mask_kmeans)
        
        print(f"Kmeans - Precision: {precision_kmeans:.3f}, Recall: {recall_kmeans:.3f}, F1: {f1_kmeans:.3f}")
        
    except Exception as e:
        print(f"Error in Kmeans method: {e}")
        return
    
    # Create difference maps
    def create_difference_map(gt_mask, pred_mask):
        """Create a difference map showing false positives and false negatives."""
        gt_binary = (gt_mask > 0).astype(np.uint8)
        pred_binary = (pred_mask > 0).astype(np.uint8)
        
        diff_map = np.zeros_like(gt_mask, dtype=np.float32)
        
        # False Negatives: GT=1, Pred=0 (should be foreground but predicted as background)
        fn_mask = (gt_binary == 1) & (pred_binary == 0)
        diff_map[fn_mask] = 1.0  # Red in hot colormap
        
        # False Positives: GT=0, Pred=1 (should be background but predicted as foreground)
        fp_mask = (gt_binary == 0) & (pred_binary == 1)
        diff_map[fp_mask] = 0.5  # Yellow in hot colormap
        
        return diff_map
    
    diff_rect = create_difference_map(gt_mask, pred_mask_rect)
    diff_kmeans = create_difference_map(gt_mask, pred_mask_kmeans)
    
    # Create visualization
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f'Background Removal Comparison - Image {image_id}', fontsize=16, fontweight='bold')
    
    # Row 1: Original, GT, Rectangles Mask, Rectangles Processed
    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(gt_mask, cmap='gray')
    axes[0, 1].set_title('Ground Truth Mask', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(pred_mask_rect, cmap='gray')
    axes[0, 2].set_title('Rectangles Predicted Mask', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(cv2.cvtColor(processed_img_rect, cv2.COLOR_BGR2RGB))
    axes[0, 3].set_title('Rectangles Processed Image', fontsize=12, fontweight='bold')
    axes[0, 3].axis('off')
    
    # Row 2: Kmeans Mask, Kmeans Processed, GT vs Rectangles Diff, GT vs Kmeans Diff
    axes[1, 0].imshow(pred_mask_kmeans, cmap='gray')
    axes[1, 0].set_title('Kmeans Predicted Mask', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(cv2.cvtColor(processed_img_kmeans, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title('Kmeans Processed Image', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(diff_rect, cmap='hot')
    axes[1, 2].set_title('GT vs Rectangles Difference\n(Red=FN, Yellow=FP)', fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')
    
    axes[1, 3].imshow(diff_kmeans, cmap='hot')
    axes[1, 3].set_title('GT vs Kmeans Difference\n(Red=FN, Yellow=FP)', fontsize=12, fontweight='bold')
    axes[1, 3].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n{'='*60}")
    print("METRICS SUMMARY")
    print(f"{'='*60}")
    print(f"Rectangles: Precision={precision_rect:.3f}, Recall={recall_rect:.3f}, F1={f1_rect:.3f}")
    print(f"Kmeans:     Precision={precision_kmeans:.3f}, Recall={recall_kmeans:.3f}, F1={f1_kmeans:.3f}")
    print(f"{'='*60}")

if __name__ == "__main__":
    print("Background Removal Testing Script")
    print("=" * 50)
    print("This script tests both Rectangles and Kmeans background removal methods")
    print("on QSD2_W2 images and extracts precision, recall, and F1 metrics.")
    print("=" * 50)
    
    # Test with one image
    test_images = ["00001"]
    
    for i, image_id in enumerate(test_images):
        print(f"\nTesting image {i+1}/{len(test_images)}: {image_id}")
        test_single_image(image_id)
        if i < len(test_images) - 1:  # Don't print separator after last image
            print("\n" + "="*80 + "\n")
    
    print("\n" + "="*50)
    print("ALL TESTS COMPLETED")
    print("="*50)
