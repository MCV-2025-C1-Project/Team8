#!/usr/bin/env python3

from dataloader.dataloader import DatasetType, WeekFolder
from services.image_retrieval_system import ImageRetrievalSystem
from services.background_removal_image_retrieval_system import BackgroundRemovalImageRetrievalSystem
from descriptors.descriptors import DescriptorMethod
from utils.measures import SimilarityMeasure
from preprocessing.preprocessors import PreprocessingMethod

def main():
    
    print("=" * 60)
    print("IMAGE RETRIEVAL SYSTEM - WEEK 2")
    print("=" * 60)

    retrieval_system = ImageRetrievalSystem()

    print("\n" + "=" * 60)
    print("VALIDATION PHASE 1: STANDARD IMAGE RETRIEVAL (QSD1_W1)")
    print("=" * 60)

    # HSV 
    print("\nMETHOD 1: HSV Histogram")
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
    print("\nMETHOD 2: HSV Block Histogram")
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
    

    print("\nVALIDATION RESULTS (QSD1_W1)")
    print("-" * 30)
    print(f"HSV:         mAP@1={hsv_results.get('mAP@1', float('nan')):.3f}, mAP@5={hsv_results.get('mAP@5', float('nan')):.3f}")
    print(f"HSV_BLOCKS:  mAP@1={hsv_block_results.get('mAP@1', float('nan')):.3f}, mAP@5={hsv_block_results.get('mAP@5', float('nan')):.3f}")

    print("\n" + "=" * 60)
    print("VALIDATION PHASE 2: BACKGROUND REMOVAL + IMAGE RETRIEVAL (QSD2_W2)")
    print("=" * 60)

    # Initialize combined system
    bg_retrieval_system = BackgroundRemovalImageRetrievalSystem()

    print("\nProcessing QSD2_W2 with background removal...")

    combined_results = bg_retrieval_system.run(
        method=DescriptorMethod.HSV_BLOCKS,
        measure=SimilarityMeasure.HIST_INTERSECT,
        index_dataset=DatasetType.BBDD,
        query_dataset=DatasetType.QSD2_W2,
        week_folder=WeekFolder.WEEK_2,
        save_results=True,
        evaluate_bg_removal=True,
        bins=32,  
        background_remover=PreprocessingMethod.BG_RECTANGLES, 
        preprocessing=PreprocessingMethod.HIST_EQ,  
        offset=50,
        visualise=False
    )

    print("\nVALIDATION RESULTS (QSD2_W2)")
    print("-" * 30)
    
    # Display retrieval results
    if 'mAP@1' in combined_results and 'mAP@5' in combined_results:
        print(f"RETRIEVAL PERFORMANCE:")
        print(f"  mAP@1: {combined_results['mAP@1']:.3f}")
        print(f"  mAP@5: {combined_results['mAP@5']:.3f}")
    
    # Display background removal results
    if 'precision' in combined_results and 'recall' in combined_results and 'f1' in combined_results:
        print(f"\nBACKGROUND REMOVAL QUALITY:")
        print(f"  Precision: {combined_results['precision']:.3f}")
        print(f"  Recall:    {combined_results['recall']:.3f}")
        print(f"  F1-Score:  {combined_results['f1']:.3f}")

    print("\n" + "=" * 60)
    print("TEST PHASE: IMAGE RETRIEVAL ON TEST DATASETS")
    print("=" * 60)

    print("\nTEST 1: QST1_W2 (NO BACKGROUND REMOVAL)")
    print("-" * 40)
    
    qst1_results = retrieval_system.run(
        method=DescriptorMethod.HSV_BLOCKS,
        measure=SimilarityMeasure.HIST_INTERSECT,
        index_dataset=DatasetType.BBDD,
        query_dataset=DatasetType.QST1_W2,
        week_folder=WeekFolder.WEEK_2,
        save_results=True,
        bins=32,
        preprocessing=PreprocessingMethod.HIST_EQ,
        ns_blocks=[1, 2, 4]
    )

    print("\nTEST 2: QST2_W2 (WITH BACKGROUND REMOVAL)")
    print("-" * 40)
    
    qst2_results = bg_retrieval_system.run(
        method=DescriptorMethod.HSV_BLOCKS,
        measure=SimilarityMeasure.HIST_INTERSECT,
        index_dataset=DatasetType.BBDD,
        query_dataset=DatasetType.QST2_W2,
        week_folder=WeekFolder.WEEK_2,
        save_results=True,
        evaluate_bg_removal=False,  # No ground truth for evaluation
        bins=32,
        background_remover=PreprocessingMethod.BG_RECTANGLES,
        preprocessing=PreprocessingMethod.HIST_EQ,
        offset=50,
        visualise=False
    )

    print("\n" + "=" * 50)
    print("ALL PHASES COMPLETED SUCCESSFULLY")
    print("=" * 50)
    print("+ Validation Phase 1: QSD1_W1 (Standard Retrieval)")
    print("+ Validation Phase 2: QSD2_W2 (Background Removal + Retrieval)")
    print("+ Test Phase 1: QST1_W2 (Standard Retrieval)")
    print("+ Test Phase 2: QST2_W2 (Background Removal + Retrieval)")
    print("=" * 50)


if __name__ == "__main__":
    main()