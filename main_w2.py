#!/usr/bin/env python3

from dataloader.dataloader import DatasetType, WeekFolder
from services.image_retrieval_system import ImageRetrievalSystem
from services.background_removal_image_retrieval_system import BackgroundRemovalImageRetrievalSystem
from descriptors.descriptors import DescriptorMethod
from utils.measures import SimilarityMeasure
from preprocessing.preprocessors import PreprocessingMethod

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

    print("\n" + "=" * 50)
    print("🎭 BACKGROUND REMOVAL + IMAGE RETRIEVAL SYSTEM")
    print("=" * 50)

    # Initialize combined system
    bg_retrieval_system = BackgroundRemovalImageRetrievalSystem()

    print("\n🔧 QSD2_W2 BACKGROUND REMOVAL + RETRIEVAL PIPELINE")
    print("-" * 50)

    try:
        combined_results = bg_retrieval_system.run(
            method=DescriptorMethod.HSV,
            measure=SimilarityMeasure.HIST_INTERSECT,
            index_dataset=DatasetType.BBDD,
            query_dataset=DatasetType.QSD2_W2,
            week_folder=WeekFolder.WEEK_2,
            save_results=True,
            evaluate_bg_removal=True,
            bins=32,
            preprocessing=PreprocessingMethod.BG_RECTANGLES,
            offset=50,
            visualise=False
        )

        print("\n📊 COMBINED RESULTS")
        print("-" * 20)
        
        # Display retrieval results
        if 'mAP@1' in combined_results and 'mAP@5' in combined_results:
            print(f"🔍 RETRIEVAL PERFORMANCE:")
            print(f"  mAP@1: {combined_results['mAP@1']:.3f}")
            print(f"  mAP@5: {combined_results['mAP@5']:.3f}")
        
        # Display background removal results
        if 'precision' in combined_results and 'recall' in combined_results and 'f1' in combined_results:
            print(f"\n🎭 BACKGROUND REMOVAL QUALITY:")
            print(f"  Precision: {combined_results['precision']:.3f}")
            print(f"  Recall:    {combined_results['recall']:.3f}")
            print(f"  F1-Score:  {combined_results['f1']:.3f}")

    except Exception as e:
        print(f"❌ Error during combined pipeline: {e}")
        print("Make sure the qsd2_w2 and BBDD datasets are properly loaded.")


if __name__ == "__main__":
    main()