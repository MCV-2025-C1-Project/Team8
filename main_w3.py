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
    nfa.run(
        dataset_type=DatasetType.QSD1_W3,
        preprocessors=[
            PreprocessingMethod.GAUSSIAN,
            PreprocessingMethod.MEDIAN,
        ]
    )

    print("\n" + "=" * 60)
    print("TASK 2: IMAGE RETRIEVAL WITH TEXTURE DESCRIPTORS (QSD1_W3)")
    print("=" * 60)

    retrieval_system = ImageRetrievalSystem()
    n_coefficients = 6
    dct_results = retrieval_system.run(
        method=DescriptorMethod.DCT,
        measure=SimilarityMeasure.HIST_INTERSECT,
        index_dataset=DatasetType.BBDD,
        query_dataset=DatasetType.QSD1_W3,
        week_folder=WeekFolder.WEEK_3,
        save_results=True,
        n_coefficients=n_coefficients,
    )
    print("\nVALIDATION RESULTS (QSD1_W3)")
    print("-" * 30)
    print(f"DCT:         mAP@1={dct_results.get('mAP@1', float('nan')):.3f}, mAP@5={dct_results.get('mAP@5', float('nan')):.3f}, n_coefficients={n_coefficients}")

    print("\n" + "=" * 60)
    print("TASK 3: MULTIPLE IMAGE RETRIEVAL WITH BACKGROUND REMOVAL (QSD1_W3)")
    print("=" * 60)

    bg_retrieval_system = BackgroundRemovalImageRetrievalSystem()




if __name__ == "__main__":
    main()