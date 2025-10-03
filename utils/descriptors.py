import numpy as np
from typing import Tuple
from PIL import Image


def histogram_1_channel(img: np.ndarray, bins: int = 256) -> np.ndarray:
    """
    Args:
        img (np.ndarray): Image with shape (height, width) or (height, width, 3), any dtype.
        bins (int): Number of bins of the returned histogram
    Returns:
        (np.ndarray): Normalized 1D vector of img's histogram.
    """
    # Convert to uint8 if needed
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)

    # Convert to grayscale if RGB
    if img.ndim == 3:
        img = np.array(Image.fromarray(img).convert("L"), dtype=np.uint8)
    elif img.ndim != 2:
        raise ValueError("Image must be 2D (grayscale) or 3D (RGB)")

    pixel_intensities = img.flatten()
    histogram = np.bincount(pixel_intensities, minlength=bins).astype(np.float32)
    return histogram / np.sum(histogram)


def histogram_3_channels(img: np.ndarray, bins: int = 256) -> np.ndarray:
    """
    Args:
        img (np.ndarray): Image with shape (height, width) or (height, width, 3), any dtype.
        bins (int): Number of bins of the returned histogram
    Returns:
        (np.ndarray): Normalized 1D vector with 3 concatenated histograms, 1 per each channel of img.
    """
    # Convert to uint8 if needed
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)

    # Convert grayscale to RGB if needed
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=2)
    elif img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("Image must be 2D (grayscale) or 3D (RGB)")

    # Compute histogram for each channel separately (without normalization)
    a = np.bincount(img[:, :, 0].flatten(), minlength=bins).astype(np.float32)
    b = np.bincount(img[:, :, 1].flatten(), minlength=bins).astype(np.float32)
    c = np.bincount(img[:, :, 2].flatten(), minlength=bins).astype(np.float32)

    # Concatenate and normalize the combined histogram
    histogram = np.concatenate([a, b, c], axis=0)
    return histogram / np.sum(histogram)
