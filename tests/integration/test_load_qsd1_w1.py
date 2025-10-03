import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from dataloader.dataloader import DataLoader, DatasetType


def test_load_qsd1_w1():
    """Integration test for loading qsd1_w1 dataset."""
    print("=" * 60)
    print("INTEGRATION TEST: Loading qsd1_w1 Dataset")
    print("=" * 60)

    try:
        # Initialize dataloader
        dl = DataLoader()
        print(f"✓ DataLoader initialized")
        print(f"  Root path: {dl.root_path}")
        print(f"  Data path: {dl.data_path}")

        # Load qsd1_w1 dataset
        print(f"\n📁 Loading qsd1_w1 dataset...")
        dl.load_dataset(DatasetType.QSD1_W1)

        # Get dataset info
        info = dl.get_dataset_info()
        print(f"✓ qsd1_w1 dataset loaded successfully!")
        print(f"  Dataset type: {info['dataset_type']}")
        print(f"  Number of images: {info['num_images']}")
        if info["image_ids"]:
            print(
                f"  Image IDs range: {min(info['image_ids'])} - {max(info['image_ids'])}"
            )
        else:
            print(f"  No images loaded!")

        # Test iteration and show first 3 images
        print(f"\n🖼️  Testing image iteration (first 3 images):")
        count = 0
        for image_id, image, info_text, relationship in dl.iterate_images():
            print(f"  [{count+1}] ID: {image_id}")
            print(f"      Shape: {image.shape}")
            print(f"      Info: {info_text}")
            print(f"      Relationship: {relationship}")
            count += 1
            if count >= 3:
                break

        # Test get_image_by_id
        if info["image_ids"]:
            test_id = info["image_ids"][0]
            image_data = dl.get_image_by_id(test_id)
            print(f"\n🔍 Testing get_image_by_id({test_id}):")
            print(f"  ✓ Retrieved image with shape: {image_data['image'].shape}")

        print(f"\n✅ qsd1_w1 integration test PASSED!")
        return True

    except Exception as e:
        print(f"\n❌ qsd1_w1 integration test FAILED!")
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_load_qsd1_w1()
