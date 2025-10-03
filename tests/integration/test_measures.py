import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
from PIL import Image
from dataloader.dataloader import DataLoader, DatasetType
from utils.descriptors import histogram_1_channel, histogram_3_channels
from utils.measures import (
    euclidean_dist,
    l1_dist,
    x2_dist,
    hist_intersect,
    hellinger_kernel,
)


def test_measures_integration() -> bool:
    """Integration test for distance measures using real image data."""
    print("=" * 60)
    print("INTEGRATION TEST: Distance Measures")
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

        # Get two different images for comparison
        if len(info["image_ids"]) < 2:
            print("❌ Need at least 2 images for comparison!")
            return False

        img1_id = info["image_ids"][0]
        img2_id = info["image_ids"][1]

        img1_data = dl.get_image_by_id(img1_id)
        img2_data = dl.get_image_by_id(img2_id)

        if img1_data is None or img2_data is None:
            print("❌ Could not retrieve test images")
            return False

        img1 = img1_data["image"]
        img2 = img2_data["image"]

        print(f"✓ Retrieved test images:")
        print(f"  Image 1 (ID: {img1_id}): {img1.shape} - {img1_data['info']}")
        print(f"  Image 2 (ID: {img2_id}): {img2.shape} - {img2_data['info']}")

        # Ensure images are uint8
        if img1.dtype != np.uint8:
            img1 = img1.astype(np.uint8)
        if img2.dtype != np.uint8:
            img2 = img2.astype(np.uint8)

        # Convert to grayscale for histogram comparison
        img1_gray = np.array(Image.fromarray(img1).convert("L"), dtype=np.uint8)
        img2_gray = np.array(Image.fromarray(img2).convert("L"), dtype=np.uint8)

        # Compute histograms
        print("\n📊 Computing histograms...")
        h1 = histogram_1_channel(img1_gray, bins=256).astype(np.float32)
        h2 = histogram_1_channel(img2_gray, bins=256).astype(np.float32)

        # Normalize histograms for proper distance calculations
        h1 /= np.sum(h1)
        h2 /= np.sum(h2)

        print(f"✓ Histograms computed and normalized")

        # Test all distance measures
        print("\n📏 Computing distance measures...")

        euclidean = euclidean_dist(h1, h2)
        l1 = l1_dist(h1, h2)
        chi_squared = x2_dist(h1, h2)
        intersection = hist_intersect(h1, h2)
        hellinger = hellinger_kernel(h1, h2)

        print(f"✓ Euclidean distance: {euclidean:.6f}")
        print(f"✓ L1 distance: {l1:.6f}")
        print(f"✓ Chi-squared distance: {chi_squared:.6f}")
        print(f"✓ Histogram intersection (similarity): {intersection:.6f}")
        print(f"✓ Hellinger kernel (similarity): {hellinger:.6f}")

        # Validate results
        print("\n🔍 Validating results...")

        # All distances should be non-negative
        assert (
            euclidean >= 0
        ), f"Euclidean distance should be non-negative, got {euclidean}"
        assert l1 >= 0, f"L1 distance should be non-negative, got {l1}"
        assert (
            chi_squared >= 0
        ), f"Chi-squared distance should be non-negative, got {chi_squared}"

        # Similarities should be in valid ranges
        assert (
            0 <= intersection <= 1
        ), f"Histogram intersection should be in [0,1], got {intersection}"
        assert (
            0 <= hellinger <= 1
        ), f"Hellinger kernel should be in [0,1], got {hellinger}"

        # L1 distance should be <= 2 for normalized histograms
        assert (
            l1 <= 2
        ), f"L1 distance for normalized histograms should be <= 2, got {l1}"

        print("✓ All validation checks passed")

        # Test with identical histograms (should give specific results)
        print("\n🔄 Testing with identical histograms...")
        euclidean_same = euclidean_dist(h1, h1)
        l1_same = l1_dist(h1, h1)
        chi_squared_same = x2_dist(h1, h1)
        intersection_same = hist_intersect(h1, h1)
        hellinger_same = hellinger_kernel(h1, h1)

        # For identical histograms
        assert (
            abs(euclidean_same) < 1e-10
        ), f"Euclidean distance for identical histograms should be ~0, got {euclidean_same}"
        assert (
            abs(l1_same) < 1e-10
        ), f"L1 distance for identical histograms should be ~0, got {l1_same}"
        assert (
            abs(chi_squared_same) < 1e-10
        ), f"Chi-squared distance for identical histograms should be ~0, got {chi_squared_same}"
        assert (
            abs(intersection_same - 1.0) < 1e-6
        ), f"Histogram intersection for identical histograms should be ~1, got {intersection_same}"
        assert (
            abs(hellinger_same - 1.0) < 1e-6
        ), f"Hellinger kernel for identical histograms should be ~1, got {hellinger_same}"

        print("✓ Identical histogram tests passed")

        print("\n✅ Distance measures integration test PASSED!")
        return True

    except Exception as e:
        print(f"\n❌ Distance measures integration test FAILED!")
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_measures_integration()
