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
    print("TASK 3: KEYPOINT DESCRIPTOR IMAGE RETRIEVAL SYSTEM")
    print("=" * 60)

    retrieval_system = KeyPointImageRetrievalSystem()
    retrieval_system.run(
        local_descriptor_method=DescriptorMethod.ORB,
        index_dataset=DatasetType.BBDD,
        query_dataset=DatasetType.QSD1_W4,
        week_folder=WeekFolder.WEEK_4,
        save_results=True,
        preprocessing=PreprocessingMethod.GAMMA,
        gamma=0.9,  # or try 1.2 for brighter
        similarity_metric=cv.NORM_HAMMING,  # Options: cv.NORM_HAMMING, cv.NORM_HAMMING2, cv.NORM_L1, cv.NORM_L2
        ratio_threshold=0.60, 
    )



if __name__ == "__main__":
    main()