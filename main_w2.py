#!/usr/bin/env python3

from dataloader.dataloader import DatasetType, WeekFolder
from services.image_retrieval_system import ImageRetrievalSystem
from descriptors.descriptors import DescriptorMethod
from utils.measures import SimilarityMeasure
from preprocessing.color_adjustments import PreprocessingMethod

def main():
    
    print("=" * 50)
    print("🔎 IMAGE RETRIEVAL SYSTEM - WEEK 2")
    print("=" * 50)

    retrieval_system = ImageRetrievalSystem()

    print("\n📊 VALIDATION PHASE (QSD1_W1)")
    print("-" * 30)

    # HSV 
    print("\n✨ METHOD 1: HSV Histogram")
    hsv_results = retrieval_system.run(
        method=DescriptorMethod.HSV,
        measure=SimilarityMeasure.HIST_INTERSECT,
        index_dataset=DatasetType.BBDD,
        query_dataset=DatasetType.QSD1_W1,
        week_folder=WeekFolder.WEEK_2,
        save_results=True,
        bins=32,
        preprocessing=PreprocessingMethod.HIST_EQ,
    )
    # HSV BLOCK HISTOGRAM
    print("\n✨ METHOD 2: HSV Block Histogram")
    hsv_block_results = retrieval_system.run(
        method=DescriptorMethod.HSV_BLOCKS,
        ns_blocks=[1, 2, 4],  # Better spatial coverage: 1+4+16=21 blocks
        measure=SimilarityMeasure.HIST_INTERSECT,
        index_dataset=DatasetType.BBDD,
        query_dataset=DatasetType.QSD1_W1,
        week_folder=WeekFolder.WEEK_2,
        save_results=True,
        bins=32,
        preprocessing=PreprocessingMethod.HIST_EQ,
    )
    

    print("\n✅ VALIDATION RESULTS")
    print("-" * 20)
    print(f"HSV:         mAP@1={hsv_results.get('mAP@1', float('nan')):.3f}, mAP@5={hsv_results.get('mAP@5', float('nan')):.3f}")
    print(f"HSV_BLOCKS:  mAP@1={hsv_block_results.get('mAP@1', float('nan')):.3f}, mAP@5={hsv_block_results.get('mAP@5', float('nan')):.3f}")


if __name__ == "__main__":
    main()