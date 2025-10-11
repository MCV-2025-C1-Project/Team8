import numpy as np
from PIL import Image


def three_d_histogram(img: np.ndarray, bins: int = 256) -> np.ndarray:
    """
    Args:
        img (np.ndarray): Image with shape (height, width) or (height, width, 3), any dtype.
        bins (int): Number of bins of the returned histogram
    Returns:
        (np.ndarray): Normalized 3D vector of img's histogram.
    """
    # Convert to uint8 if needed
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)

    # Split channels
    a, b, c = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    a = a.flatten()
    b = b.flatten()
    c = c.flatten()

    # Compute 3D histogram
    histogram, _ = np.histogramdd(
        sample=np.stack((a, b, c), axis=1),
        bins=bins,
        range=[[0, 256], [0, 256], [0, 256]]
    )

    # Normalise
    histogram = histogram.astype(np.float32)
    histogram /= histogram.sum()

    return histogram