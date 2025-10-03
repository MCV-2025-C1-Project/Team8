import numpy as np


def histogram_1_channel(img, bins=256):
    """
    Args:
        img (np.ndarray): Image with shape (height, width), dtype=uint8, range=[0,255].
        bins (int): Number of bins of the returned histogram
    Returns:
        (np.ndarray): 1D vector of img's histogram.
    """
    assert img.dtype == np.uint8, "Image must be of dtype uint8"
    assert img.ndim == 2, "Image must be (H, W)"
    pixel_intensities = img.flatten()
    return np.bincount(pixel_intensities, minlength=bins)


def histogram_3_channels(img, bins=256):
    """
    Args:
        img (np.ndarray): Image with shape (height, width, 3), dtype=uint8, range=[0,255].
        bins (int): Number of bins of the returned histogram
    Returns:
        (np.ndarray): 1D vector with 3 concatenated histograms, 1 per each channel of img.
    """
    assert img.dtype == np.uint8, "Image must be of dtype uint8"
    assert img.ndim == 3 and img.shape[2] == 3, "Image must be (H, W, 3)"
    a = histogram_1_channel(img[:,:,0], bins=bins)
    b = histogram_1_channel(img[:,:,1], bins=bins)
    c = histogram_1_channel(img[:,:,2], bins=bins)
    return np.concatenate([a, b, c], axis=0)

def test_histograms():
    import os
    from PIL import Image
    import matplotlib.pyplot as plt

    current_file = os.path.abspath(__file__)         # .../utils/descriptors.py
    root_path = os.path.dirname(os.path.dirname(current_file))  # puja dos nivells

    img_path = os.path.join(root_path, "data", "BBDD", "bbdd_00000.jpg")

    img = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
    img_gray = np.array(Image.open(img_path).convert("L"), dtype=np.uint8)

    hist_gray = histogram_1_channel(img_gray, bins=256)
    hist_rgb = histogram_3_channels(img, bins=256)

    # ---------- PLOT ----------
    fig, axs = plt.subplots(1, 4, figsize=(20, 4))

    axs[0].bar(range(256), hist_gray, color='gray')
    axs[0].set_title("Gray Histogram")
    axs[0].set_xlabel("Pixel intensity")
    axs[0].set_ylabel("Frequency")

    axs[1].bar(range(256), hist_rgb[0:256], color='red')
    axs[1].set_title("Red Histogram")
    axs[1].set_xlabel("Pixel intensity")
    axs[1].set_ylabel("Frequency")

    axs[2].bar(range(256), hist_rgb[256:512], color='green')
    axs[2].set_title("Green Histogram")
    axs[2].set_xlabel("Pixel intensity")
    axs[2].set_ylabel("Frequency")

    axs[3].bar(range(256), hist_rgb[512:768], color='blue')
    axs[3].set_title("Blue Histogram")
    axs[3].set_xlabel("Pixel intensity")
    axs[3].set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_histograms()