#!/usr/bin/env python3
"""
Main entry point for the Image Retrieval System
"""

from dataloader.dataloader import DatasetType
from services.image_retrieval_system_week_1 import ImageRetrievalSystem
from utils.descriptors import DescriptorMethod
from utils.measures import SimilarityMeasure

def main():
    
    print("=" * 50)
    print("IMAGE RETRIEVAL SYSTEM - WEEK 1")
    print("=" * 50)

    retrieval_system = ImageRetrievalSystem()

    print("\n🔍 VALIDATION PHASE (QSD1_W1)")
    print("-" * 30)
    
    # Method 1: CieLab Histogram
    print("\n📊 METHOD 1: CieLab Histogram")
    print("-" * 25)
    lab_validation_results = retrieval_system.run(
        method=DescriptorMethod.LAB,
        measure=SimilarityMeasure.HIST_INTERSECT,
        index_dataset=DatasetType.BBDD,
        query_dataset=DatasetType.QSD1_W1,
        save_results=True
    )
    
    # Method 2: HSV Histogram
    print("\n📊 METHOD 2: HSV Histogram")
    print("-" * 25)
    hsv_validation_results = retrieval_system.run(
        method=DescriptorMethod.HSV,
        measure=SimilarityMeasure.HIST_INTERSECT,
        index_dataset=DatasetType.BBDD,
        query_dataset=DatasetType.QSD1_W1,
        save_results=True
    )

    print("\n📊 VALIDATION RESULTS")
    print("-" * 20)
    print(f"CieLab:  mAP@1={lab_validation_results['mAP@1']:.3f}, mAP@5={lab_validation_results['mAP@5']:.3f}")
    print(f"HSV:     mAP@1={hsv_validation_results['mAP@1']:.3f}, mAP@5={hsv_validation_results['mAP@5']:.3f}")

    # Determine best method
    if hsv_validation_results['mAP@1'] > lab_validation_results['mAP@1']:
        best_method = "HSV"
        best_score = hsv_validation_results['mAP@1']
    else:
        best_method = "CieLab"
        best_score = lab_validation_results['mAP@1']
    
    print(f"🏆 Best: {best_method} (mAP@1: {best_score:.3f})")

    print("\n🎯 TEST PHASE (QST1_W1)")
    print("-" * 25)
    
    # Method 1: CieLab Histogram
    print("\n📊 METHOD 1: CieLab Histogram")
    print("-" * 25)
    retrieval_system.run(
        method=DescriptorMethod.LAB,
        measure=SimilarityMeasure.HIST_INTERSECT,
        index_dataset=DatasetType.BBDD,
        query_dataset=DatasetType.QST1_W1,
        save_results=True
    )
    
    # Method 2: HSV Histogram
    print("\n📊 METHOD 2: HSV Histogram")
    print("-" * 25)
    retrieval_system.run(
        method=DescriptorMethod.HSV,
        measure=SimilarityMeasure.HIST_INTERSECT,
        index_dataset=DatasetType.BBDD,
        query_dataset=DatasetType.QST1_W1,
        save_results=True
    )

    print("\n✅ COMPLETED")
    print(f"📁 Results: results/week1/QSD1_W1/ & results/week1/QST1_W1/")


if __name__ == "__main__":
    main()