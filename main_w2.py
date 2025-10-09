#!/usr/bin/env python3

from dataloader.dataloader import DatasetType
from services.image_retrieval_system_week_2 import ImageRetrievalSystem
from descriptors.color_histograms import DescriptorMethod
from utils.measures import SimilarityMeasure
from utils.preprocessing import PreprocessingMethod

def main():
    
    print("=" * 50)
    print("🔎 IMAGE RETRIEVAL SYSTEM - WEEK 2")
    print("=" * 50)

    retrieval_system = ImageRetrievalSystem()

    print("\n📊 VALIDATION PHASE (QSD1_W1)")
    print("-" * 30)

    # LAB 
    print("\n✨ METHOD 1: CieLab Histogram")
    lab_results = retrieval_system.run(
        method=DescriptorMethod.LAB,
        ns_blocks=[1, 2, 3],
        measure=SimilarityMeasure.HIST_INTERSECT,
        index_dataset=DatasetType.BBDD,
        query_dataset=DatasetType.QSD1_W1,
        save_results=True,
        bins=32,
        # preprocessing=PreprocessingMethod.GAMMA,
    )
    # HSV 
    print("\n✨ METHOD 2: HSV Histogram")
    hsv_results = retrieval_system.run(
        method=DescriptorMethod.HSV,
        ns_blocks=[1, 2, 3],
        measure=SimilarityMeasure.HIST_INTERSECT,
        index_dataset=DatasetType.BBDD,
        query_dataset=DatasetType.QSD1_W1,
        save_results=True,
        bins=32,
        preprocessing=PreprocessingMethod.HIST_EQ,
    )

    print("\n✅ VALIDATION RESULTS")
    print("-" * 20)
    print(f"CieLab:  mAP@1={lab_results.get('mAP@1', float('nan')):.3f}, mAP@5={lab_results.get('mAP@5', float('nan')):.3f}")
    print(f"HSV:     mAP@1={hsv_results.get('mAP@1', float('nan')):.3f}, mAP@5={hsv_results.get('mAP@5', float('nan')):.3f}")


if __name__ == "__main__":
    main()