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




if __name__ == "__main__":
    main()