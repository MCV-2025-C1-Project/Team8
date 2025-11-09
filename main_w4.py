from dataloader.dataloader import DatasetType, WeekFolder
from services.keypoint_image_retrieval_system import KeyPointImageRetrievalSystem
from descriptors.descriptors import DescriptorMethod
from preprocessing.preprocessors import PreprocessingMethod
import cv2 as cv



def main():
        
    print("=" * 9)
    print(" WEEK 4")
    print("=" * 9)

    print("\n" + "=" * 60)
    print("TASK 3: KEYPOINT DESCRIPTOR IMAGE RETRIEVAL SYSTEM IN VALIDATION SET QSD1_W4")
    print("=" * 60)
    
    print("\nMETHOD 1: QSD1_W4 (ORB DESCRIPTOR)")
    print("-" * 40)

    retrieval_system = KeyPointImageRetrievalSystem()
    retrieval_system.run(
        local_descriptor_method=DescriptorMethod.ORB,
        index_dataset=DatasetType.BBDD,
        query_dataset=DatasetType.QSD1_W4,
        week_folder=WeekFolder.WEEK_4,
        save_results=True,
        preprocessing=PreprocessingMethod.MEDIAN,
        similarity_metric=cv.NORM_HAMMING, 
        ratio_threshold=0.65, 
    )
    
    print("\nMETHOD 2: QSD1_W4 (SIFT DESCRIPTOR)")
    print("-" * 40)

    retrieval_system.run(
        local_descriptor_method=DescriptorMethod.SIFT,
        index_dataset=DatasetType.BBDD,
        query_dataset=DatasetType.QSD1_W4,
        week_folder=WeekFolder.WEEK_4,
        save_results=True,
        preprocessing=PreprocessingMethod.GAUSSIAN,
        similarity_metric=cv.NORM_L2,
        ratio_threshold=0.45,
    )


    print("\n" + "=" * 60)
    print("TASK 4: KEYPOINT DESCRIPTOR IMAGE RETRIEVAL SYSTEM IN TEST SET QST1_W4")
    print("=" * 60)
    
    print("\n BEST PERFORMING METHOD IN VALIDATION SET: SIFT DESCRIPTOR")
    print("-" * 40)

    retrieval_system.run(
        local_descriptor_method=DescriptorMethod.SIFT,
        index_dataset=DatasetType.BBDD,
        query_dataset=DatasetType.QST1_W4,
        week_folder=WeekFolder.WEEK_4,
        save_results=True,
        preprocessing=PreprocessingMethod.GAUSSIAN,
        similarity_metric=cv.NORM_L2,
        ratio_threshold=0.45,
    )

if __name__ == "__main__":
    main()