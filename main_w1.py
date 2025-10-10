#!/usr/bin/env python3
"""
Main entry point for the Image Retrieval System
"""

from dataloader.dataloader import DatasetType, WeekFolder
from services.image_retrieval_system import ImageRetrievalSystem
from descriptors.descriptors import DescriptorMethod
from utils.measures import SimilarityMeasure
from preprocessing.color_adjustments import PreprocessingMethod

def main():
    
    print("=" * 50)
    print("IMAGE RETRIEVAL SYSTEM - WEEK 1")
    print("=" * 50)

    retrieval_system = ImageRetrievalSystem()

    print("\nVALIDATION PHASE (QSD1_W1)")
    print("-" * 30)

    # LAB 
    print("\nMETHOD 1: CieLab Histogram")
    lab_results = retrieval_system.run(
        method=DescriptorMethod.LAB,
        measure=SimilarityMeasure.HIST_INTERSECT,
        index_dataset=DatasetType.BBDD,
        query_dataset=DatasetType.QSD1_W1,
        week_folder=WeekFolder.WEEK_1,
        save_results=True,
        bins=32,
        preprocessing=PreprocessingMethod.GAMMA,
    )

    # HSV 
    print("\nMETHOD 2: HSV Histogram")
    hsv_results = retrieval_system.run(
        method=DescriptorMethod.HSV,
        measure=SimilarityMeasure.HIST_INTERSECT,
        index_dataset=DatasetType.BBDD,
        query_dataset=DatasetType.QSD1_W1,
        week_folder=WeekFolder.WEEK_1,
        save_results=True,
        bins=32,
        preprocessing=PreprocessingMethod.HIST_EQ,
    )

    print("\nVALIDATION RESULTS")
    print("-" * 20)
    print(f"CieLab:  mAP@1={lab_results.get('mAP@1', float('nan')):.3f}, mAP@5={lab_results.get('mAP@5', float('nan')):.3f}")
    print(f"HSV:     mAP@1={hsv_results.get('mAP@1', float('nan')):.3f}, mAP@5={hsv_results.get('mAP@5', float('nan')):.3f}")

    print("\nTEST PHASE (QST1_W1)")
    print("-" * 25)

    # LAB on QST1_W1 
    print("\nMETHOD 1 TEST: CieLab Histogram")
    retrieval_system.run(
        method=DescriptorMethod.LAB,
        measure=SimilarityMeasure.HIST_INTERSECT,
        index_dataset=DatasetType.BBDD,
        query_dataset=DatasetType.QST1_W1,
        week_folder=WeekFolder.WEEK_1,
        save_results=True,
        bins=32,
        preprocessing=PreprocessingMethod.GAMMA,
    )

    # HSV on QST1_W1 
    print("\nMETHOD 2 TEST: HSV Histogram")
    retrieval_system.run(
        method=DescriptorMethod.HSV,
        measure=SimilarityMeasure.HIST_INTERSECT,
        index_dataset=DatasetType.BBDD,
        query_dataset=DatasetType.QST1_W1,
        week_folder=WeekFolder.WEEK_1,
        save_results=True,
        bins=32,
        preprocessing=PreprocessingMethod.HIST_EQ,
    )

    print("\nCOMPLETED")
    print("Results saved to results/week_1/QSD1_W1/ & results/week_1/QST1_W1/")


if __name__ == "__main__":
    main()