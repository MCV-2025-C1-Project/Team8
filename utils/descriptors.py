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
    return np.concat([a, b, c], axis=0)
