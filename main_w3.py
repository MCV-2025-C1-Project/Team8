from dataloader.dataloader import DatasetType, WeekFolder
from descriptors.descriptors import DescriptorMethod
from services.image_retrieval_system import ImageRetrievalSystem
from services.background_removal_image_retrieval_system import BackgroundRemovalImageRetrievalSystem
from services.noise_filtering_assessment import NoiseFilteringAssessment
from utils.measures import SimilarityMeasure
from preprocessing.preprocessors import PreprocessingMethod


def main():
    
    print("=" * 9)
    print(" WEEK 3")
    print("=" * 9)

    print("\n" + "=" * 60)
    print("TASK 1: NOISE FILTERING (QSD1_W3)")
    print("=" * 60)

    nfa = NoiseFilteringAssessment()

    print(f"\n--- NOISE FILTERING GENERAL ASSESSMENT ---\n")    
    
    nfa.run(
        dataset_type=DatasetType.QSD1_W3,
        preprocessors=[
            PreprocessingMethod.AVERAGE,
            PreprocessingMethod.GAUSSIAN,
            PreprocessingMethod.MEDIAN,
        ]
    )

    print(f"\n--- NOISE FILTERING SINGLE IMAGE ASSESSMENT ---\n")
    # run single image qualitative assessment
    for image_id in [5, 6, 16]: # [5, 6, 16] are the image IDs of the noisy images to be assessed
        nfa.run_single_image(
            dataset_type=DatasetType.QSD1_W3,
            image_id=image_id,
            preprocessor=PreprocessingMethod.MEDIAN,
        )

    print("\n" + "=" * 60)
    print("TASK 2: IMAGE RETRIEVAL WITH TEXTURE DESCRIPTORS IN VALIDATION SET QSD1_W3")
    print("=" * 60 + "\n")

    n_coefficients = 6
    
    print("\nMETHOD 1: DCT with Histogram Equalization (NO NOISE REMOVAL)")
    print("-" * 50)
    # Create separate instance for each method to avoid descriptor reuse
    retrieval_system_1 = ImageRetrievalSystem()
    # DCT Results with HIST_EQ only
    dct_hist_eq_results = retrieval_system_1.run(
        method=DescriptorMethod.DCT,
        measure=SimilarityMeasure.HIST_INTERSECT,
        index_dataset=DatasetType.BBDD,
        preprocessing=PreprocessingMethod.HIST_EQ,
        query_dataset=DatasetType.QSD1_W3,
        week_folder=WeekFolder.WEEK_3,
        save_results=False,
        n_coefficients=n_coefficients,
    )
    
    print("\nMETHOD 2: DCT with Median Filter + Histogram Equalization (WITH NOISE REMOVAL)")
    print("-" * 50)
    # Create separate instance for each method to avoid descriptor reuse
    retrieval_system_2 = ImageRetrievalSystem()
    # DCT Results with MEDIAN_HIST_EQ (noise removal + histogram equalization)
    dct_median_hist_eq_results = retrieval_system_2.run(
        method=DescriptorMethod.DCT,
        measure=SimilarityMeasure.HIST_INTERSECT,
        index_dataset=DatasetType.BBDD,
        preprocessing=PreprocessingMethod.MEDIAN_HIST_EQ,
        query_dataset=DatasetType.QSD1_W3,
        week_folder=WeekFolder.WEEK_3,
        save_results=False,
        n_coefficients=n_coefficients,
    )
    
    print("\nMETHOD 3: HSV_BLOCKS with Median Filter + Histogram Equalization (WITH NOISE REMOVAL)")
    print("-" * 50)
    # Create separate instance for each method to avoid descriptor reuse
    retrieval_system_3 = ImageRetrievalSystem()
    # HSV_BLOCKS Results with MEDIAN_HIST_EQ (noise removal + histogram equalization)
    hsv_median_hist_eq_results = retrieval_system_3.run(
        method=DescriptorMethod.HSV_BLOCKS,
        ns_blocks=[4, 6],
        measure=SimilarityMeasure.HIST_INTERSECT,
        index_dataset=DatasetType.BBDD,
        preprocessing=PreprocessingMethod.MEDIAN_HIST_EQ,
        query_dataset=DatasetType.QSD1_W3,
        week_folder=WeekFolder.WEEK_3,
        save_results=False,
        bins=32,
    )
    
    print("\nTASK 2: VALIDATION RESULTS (QSD1_W3)")
    print("-" * 30)
    print(f"METHOD 1 (DCT + HIST_EQ):     mAP@1={dct_hist_eq_results.get('mAP@1', float('nan')):.3f}, mAP@5={dct_hist_eq_results.get('mAP@5', float('nan')):.3f}, n_coefficients={n_coefficients}")
    print(f"METHOD 2 (DCT + MEDIAN + HIST): mAP@1={dct_median_hist_eq_results.get('mAP@1', float('nan')):.3f}, mAP@5={dct_median_hist_eq_results.get('mAP@5', float('nan')):.3f}, n_coefficients={n_coefficients}")
    print(f"METHOD 3 (HSV_BLOCKS + MEDIAN + HIST): mAP@1={hsv_median_hist_eq_results.get('mAP@1', float('nan')):.3f}, mAP@5={hsv_median_hist_eq_results.get('mAP@5', float('nan')):.3f}, ns_blocks=[4,6]")
    
    print("\n" + "=" * 60)
    print("TASK 3: MULTIPLE IMAGE RETRIEVAL WITH BACKGROUND REMOVAL IN VALIDATION SET QSD2_W3")
    print("=" * 60)

    bg_retrieval_system = BackgroundRemovalImageRetrievalSystem()

    print("\nMETHOD 1: HSV_BLOCKS with K-Means Background Removal")
    k_means_results = bg_retrieval_system.run(
        method=DescriptorMethod.HSV_BLOCKS,
        ns_blocks=[4, 6],

        measure=SimilarityMeasure.HIST_INTERSECT,
        index_dataset=DatasetType.BBDD,
        query_dataset=DatasetType.QSD2_W3,
        week_folder=WeekFolder.WEEK_3,
        save_results=False,
        preprocessing=PreprocessingMethod.HIST_EQ,

        background_remover=PreprocessingMethod.BG_KMEANS,
        subdivide=True,
    )

    print("\nTASK 3: VALIDATION RESULTS (QSD2_W3)")
    print("-" * 30)
    if 'mAP@1' in k_means_results and 'mAP@5' in k_means_results:
        print(f"K-Means RETRIEVAL PERFORMANCE:")
        print(f"  mAP@1: {k_means_results['mAP@1']:.3f}")
        print(f"  mAP@5: {k_means_results['mAP@5']:.3f}")
    if 'precision' in k_means_results and 'recall' in k_means_results and 'f1' in k_means_results:
        print(f"\nK-Means BACKGROUND REMOVAL QUALITY:")
        print(f"  Precision: {k_means_results['precision']:.3f}")
        print(f"  Recall:    {k_means_results['recall']:.3f}")
        print(f"  F1-Score:  {k_means_results['f1']:.3f}")

    print("\n" + "=" * 60)
    print("TEST PHASE: IMAGE RETRIEVAL ON TEST DATASETS")
    print("=" * 60)

    print("\nTEST 2: QST2_W2 (WITH BACKGROUND REMOVAL)")
    print("-" * 40)

    k_means_results = bg_retrieval_system.run(
        method=DescriptorMethod.HSV_BLOCKS,
        ns_blocks=[4, 6],

        measure=SimilarityMeasure.HIST_INTERSECT,
        index_dataset=DatasetType.BBDD,
        query_dataset=DatasetType.QST2_W3,
        week_folder=WeekFolder.WEEK_3,
        save_results=True,
        preprocessing=PreprocessingMethod.HIST_EQ,
        evaluate_bg_removal=False,  # No ground truth for evaluation

        background_remover=PreprocessingMethod.BG_KMEANS,
        subdivide=True,
        # visualise=True
    )


if __name__ == "__main__":
    main()