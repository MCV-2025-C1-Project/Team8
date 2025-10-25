#!/usr/bin/env python3

"""
LBP Descriptor Testing Script

This script tests LBP (Local Binary Pattern) descriptors with and without histogram equalization
preprocessing on the QSD1_W3 dataset. It compares performance metrics and extracts mAP scores.

Usage:
    python scripts/test_lbp.py

The script will test LBP descriptors with different preprocessing methods and extract mAP@1 and mAP@5 metrics.
"""

import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataloader.dataloader import DatasetType, WeekFolder
from descriptors.descriptors import DescriptorMethod
from services.image_retrieval_system import ImageRetrievalSystem
from utils.measures import SimilarityMeasure
from preprocessing.preprocessors import PreprocessingMethod


def test_lbp_descriptors():
    """Test LBP descriptors with different preprocessing methods."""
    
    print("=" * 60)
    print("LBP DESCRIPTOR TESTING ON QSD1_W3")
    print("=" * 60)
    
    # LBP parameters
    radius = 1
    n_neighbors = 8
    lbp_method = 'uniform'
    
    print(f"\nLBP Parameters:")
    print(f"  Radius: {radius}")
    print(f"  Neighbors: {n_neighbors}")
    print(f"  Method: {lbp_method}")
    
    print("\n" + "=" * 60)
    print("METHOD 1: LBP WITHOUT HISTOGRAM EQUALIZATION")
    print("=" * 60)
    
    # Create separate instance for each method to avoid descriptor reuse
    retrieval_system_1 = ImageRetrievalSystem()
    
    # LBP Results without preprocessing
    lbp_no_preprocessing_results = retrieval_system_1.run(
        method=DescriptorMethod.LBP,
        measure=SimilarityMeasure.HIST_INTERSECT,
        index_dataset=DatasetType.BBDD,
        preprocessing=PreprocessingMethod.NONE,
        query_dataset=DatasetType.QSD1_W3,
        week_folder=WeekFolder.WEEK_3,
        save_results=False,
        radius=radius,
        n_neighbors=n_neighbors,
        lbp_method=lbp_method,
    )
    
    print("\n" + "=" * 60)
    print("METHOD 2: LBP WITH HISTOGRAM EQUALIZATION")
    print("=" * 60)
    
    # Create separate instance for each method to avoid descriptor reuse
    retrieval_system_2 = ImageRetrievalSystem()
    
    # LBP Results with histogram equalization
    lbp_hist_eq_results = retrieval_system_2.run(
        method=DescriptorMethod.LBP,
        measure=SimilarityMeasure.HIST_INTERSECT,
        index_dataset=DatasetType.BBDD,
        preprocessing=PreprocessingMethod.HIST_EQ,
        query_dataset=DatasetType.QSD1_W3,
        week_folder=WeekFolder.WEEK_3,
        save_results=False,
        radius=radius,
        n_neighbors=n_neighbors,
        lbp_method=lbp_method,
    )
    
    print("\n" + "=" * 60)
    print("METHOD 3: LBP WITH MEDIAN FILTER + HISTOGRAM EQUALIZATION")
    print("=" * 60)

    
    # Print results summary
    print("\n" + "=" * 60)
    print("LBP TESTING RESULTS SUMMARY (QSD1_W3)")
    print("=" * 60)
    
    print(f"\nMETHOD 1 (LBP + NO PREPROCESSING):")
    print(f"  mAP@1: {lbp_no_preprocessing_results.get('mAP@1', float('nan')):.3f}")
    print(f"  mAP@5: {lbp_no_preprocessing_results.get('mAP@5', float('nan')):.3f}")
    print(f"  Parameters: radius={radius}, n_neighbors={n_neighbors}, method={lbp_method}")
    
    print(f"\nMETHOD 2 (LBP + HISTOGRAM EQUALIZATION):")
    print(f"  mAP@1: {lbp_hist_eq_results.get('mAP@1', float('nan')):.3f}")
    print(f"  mAP@5: {lbp_hist_eq_results.get('mAP@5', float('nan')):.3f}")
    print(f"  Parameters: radius={radius}, n_neighbors={n_neighbors}, method={lbp_method}")
    

if __name__ == "__main__":
    print("LBP Descriptor Testing Script")
    print("=" * 50)
    print("This script tests LBP descriptors with different preprocessing methods")
    print("on the QSD1_W3 dataset and extracts mAP metrics.")
    print("=" * 50)
    
    try:
        results = test_lbp_descriptors()
        print("\nTest completed successfully!")
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()
