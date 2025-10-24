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
    print("TASK 2: IMAGE RETRIEVAL WITH TEXTURE DESCRIPTORS (QSD1_W3)")
    print("=" * 60 + "\n")

    retrieval_system = ImageRetrievalSystem()
    n_coefficients = 6
    
    # DCT Results
    dct_results = retrieval_system.run(
        method=DescriptorMethod.DCT,
        measure=SimilarityMeasure.HIST_INTERSECT,
        index_dataset=DatasetType.BBDD,
        preprocessing=PreprocessingMethod.HIST_EQ,
        query_dataset=DatasetType.QSD1_W3,
        week_folder=WeekFolder.WEEK_3,
        save_results=True,
        n_coefficients=n_coefficients,
    )
    
    print("\nVALIDATION RESULTS (QSD1_W3)")
    print("-" * 30)
    print(f"DCT:         mAP@1={dct_results.get('mAP@1', float('nan')):.3f}, mAP@5={dct_results.get('mAP@5', float('nan')):.3f}, n_coefficients={n_coefficients}")

    print("\n" + "=" * 60)
    print("TASK 3: MULTIPLE IMAGE RETRIEVAL WITH BACKGROUND REMOVAL (QSD2_W3)")
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
        save_results=True,
        preprocessing=PreprocessingMethod.HIST_EQ,

        background_remover=PreprocessingMethod.BG_KMEANS,
        subdivide=True,
    )

    print("\nVALIDATION RESULTS (QSD2_W3)")
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