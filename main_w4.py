from dataloader.dataloader import DatasetType, WeekFolder
from services.keypoint_image_retrieval_system import KeyPointImageRetrievalSystem
from keypoint_descriptors.keypoint_descriptors import KeypointDescriptorMethod



def main():
        
    print("=" * 9)
    print(" WEEK 4")
    print("=" * 9)

    print("\n" + "=" * 60)
    print("TASK 3: KEYPOINT DESCRIPTOR IMAGE RETRIEVAL SYSTEM")
    print("=" * 60)

    retrieval_system = KeyPointImageRetrievalSystem()
    retrieval_system.run(
        local_descriptor_method=KeypointDescriptorMethod.ORB,
        index_dataset=DatasetType.BBDD,
        query_dataset=DatasetType.QSD1_W4,
        week_folder=WeekFolder.WEEK_4,
    )



if __name__ == "__main__":
    main()