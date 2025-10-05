#!/usr/bin/env python3
"""
Main entry point for the Image Retrieval System
"""

from services.image_retrieval_system_week_1 import ImageRetrievalSystem
from utils.descriptors import DescriptorMethod
from utils.measures import SimilarityMeasure

def main():
    print("IMAGE RETRIEVAL SYSTEM - WEEK 1")
    print("=" * 60)
    print("Using QSD1 & BBDD datasets")
    print("Methods: CieLab and HSV Histograms")
    print("=" * 60)

    retrieval_system = ImageRetrievalSystem()

    print("\nRunning evaluations...")
    
    # Method 1: CieLab Histogram  
    lab_results = retrieval_system.run_evaluation(
        DescriptorMethod.LAB,
        SimilarityMeasure.HIST_INTERSECT,
        save_results=True
    )
    
    # Method 2: HSV Histogram
    hsv_results = retrieval_system.run_evaluation(
        DescriptorMethod.HSV,
        SimilarityMeasure.HIST_INTERSECT,
        save_results=True
    )

    print(f"\n{'='*60}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*60}")
    
    print(f"Method 1 (CieLab Histogram):")
    print(f"   mAP@1: {lab_results['mAP@1']:.4f}")
    print(f"   mAP@5: {lab_results['mAP@5']:.4f}")
    
    print(f"\nMethod 2 (HSV Histogram):")
    print(f"   mAP@1: {hsv_results['mAP@1']:.4f}")
    print(f"   mAP@5: {hsv_results['mAP@5']:.4f}")

    # Determine best method
    if hsv_results['mAP@1'] > lab_results['mAP@1']:
        best_method = "HSV"
        best_score = hsv_results['mAP@1']
    else:
        best_method = "CieLab"
        best_score = lab_results['mAP@1']
    
    print(f"\nBest performing method: {best_method} (mAP@1: {best_score:.4f})")
    print(f"\nResults saved to results/week1/QST1/ directory")
    print("Evaluation completed successfully!")


if __name__ == "__main__":
    main()