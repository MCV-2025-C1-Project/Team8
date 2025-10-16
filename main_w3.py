from services.image_retrieval_system import ImageRetrievalSystem
from services.background_removal_image_retrieval_system import BackgroundRemovalImageRetrievalSystem
from services.noise_filtering_assessment import NoiseFilteringAssessment


def main():
    
    print("=" * 9)
    print(" WEEK 3")
    print("=" * 9)

    print("\n" + "=" * 60)
    print("TASK 1: NOISE FILTERING (QSD1_W3)")
    print("=" * 60)

    nfa = NoiseFilteringAssessment()

    print("\n" + "=" * 60)
    print("TASK 2: IMAGE RETRIEVAL WITH TEXTURE DESCRIPTORS (QSD1_W3)")
    print("=" * 60)

    retrieval_system = ImageRetrievalSystem()

    print("\n" + "=" * 60)
    print("TASK 3: MULTIPLE IMAGE RETRIEVAL WITH BACKGROUND REMOVAL (QSD1_W3)")
    print("=" * 60)

    bg_retrieval_system = BackgroundRemovalImageRetrievalSystem()




if __name__ == "__main__":
    main()