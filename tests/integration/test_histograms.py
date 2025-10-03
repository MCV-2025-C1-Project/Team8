import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from dataloader.dataloader import DataLoader, DatasetType
from utils.descriptors import histogram_1_channel, histogram_3_channels


def test_histograms():
    """Integration test for histogram descriptors using DataLoader."""
    print("=" * 60)
    print("INTEGRATION TEST: Histogram Descriptors")
    print("=" * 60)

    try:
        # Initialize dataloader and load BBDD dataset
        dl = DataLoader()
        print("✓ DataLoader initialized")

        print("\n📁 Loading BBDD dataset...")
        dl.load_dataset(DatasetType.BBDD)

        # Get dataset info
        info = dl.get_dataset_info()
        print(f"✓ BBDD dataset loaded: {info['num_images']} images")

        # Get the first image for testing
        if not info["image_ids"]:
            print("❌ No images found in dataset!")
            return False

        test_id = info["image_ids"][0]
        image_data = dl.get_image_by_id(test_id)

        if image_data is None:
            print(f"❌ Could not retrieve image with ID {test_id}")
            return False

        img = image_data["image"]
        print(f"✓ Retrieved test image (ID: {test_id})")
        print(f"  Shape: {img.shape}")
        print(f"  Info: {image_data['info']}")

        # Ensure image is uint8
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)

        # Convert to grayscale for 1-channel histogram
        img_gray = np.array(Image.fromarray(img).convert("L"), dtype=np.uint8)

        # Compute histograms
        print("\n📊 Computing histograms...")
        hist_gray = histogram_1_channel(img_gray, bins=256)
        hist_rgb = histogram_3_channels(img, bins=256)

        print(f"✓ Gray histogram computed: shape {hist_gray.shape}")
        print(f"✓ RGB histogram computed: shape {hist_rgb.shape}")

        # Verify histogram properties
        assert hist_gray.shape == (
            256,
        ), f"Expected gray histogram shape (256,), got {hist_gray.shape}"
        assert hist_rgb.shape == (
            768,
        ), f"Expected RGB histogram shape (768,), got {hist_rgb.shape}"
        assert (
            np.sum(hist_gray) == img_gray.size
        ), "Gray histogram sum should equal number of pixels"
        assert (
            np.sum(hist_rgb) == img.size
        ), "RGB histogram sum should equal total number of pixel values"

        print("✓ Histogram validation passed")

        # Plot histograms
        print("\n📈 Plotting histograms...")
        fig, axs = plt.subplots(1, 4, figsize=(20, 4))

        axs[0].bar(range(256), hist_gray, color="gray")
        axs[0].set_title("Gray Histogram")
        axs[0].set_xlabel("Pixel intensity")
        axs[0].set_ylabel("Frequency")

        axs[1].bar(range(256), hist_rgb[0:256], color="red")
        axs[1].set_title("Red Histogram")
        axs[1].set_xlabel("Pixel intensity")
        axs[1].set_ylabel("Frequency")

        axs[2].bar(range(256), hist_rgb[256:512], color="green")
        axs[2].set_title("Green Histogram")
        axs[2].set_xlabel("Pixel intensity")
        axs[2].set_ylabel("Frequency")

        axs[3].bar(range(256), hist_rgb[512:768], color="blue")
        axs[3].set_title("Blue Histogram")
        axs[3].set_xlabel("Pixel intensity")
        axs[3].set_ylabel("Frequency")

        plt.tight_layout()
        plt.show()

        print("\n✅ Histogram integration test PASSED!")
        return True

    except Exception as e:
        print(f"\n❌ Histogram integration test FAILED!")
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_histograms()
