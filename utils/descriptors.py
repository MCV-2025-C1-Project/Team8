from enum import Enum
import numpy as np
import cv2
from typing import Tuple
from PIL import Image
from utils.preprocessing import PreprocessingMethod


def histogram_grayscale(img: np.ndarray, bins: int = 256) -> np.ndarray:
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

    # Fixed-length histogram using explicit bin edges over [0, 256)
    pixel_intensities = img.flatten()
    histogram, _ = np.histogram(pixel_intensities, bins=bins, range=(0, 256))
    histogram = histogram.astype(np.float32)
    return histogram / np.sum(histogram)


def histogram_rgb(img: np.ndarray, bins: int = 256) -> np.ndarray:
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

    # Compute histogram for each channel separately with fixed bins
    a, _ = np.histogram(img[:, :, 0].flatten(), bins=bins, range=(0, 256))
    b, _ = np.histogram(img[:, :, 1].flatten(), bins=bins, range=(0, 256))
    c, _ = np.histogram(img[:, :, 2].flatten(), bins=bins, range=(0, 256))
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    c = c.astype(np.float32)

    # Concatenate and normalize the combined histogram
    histogram = np.concatenate([a, b, c], axis=0)
    return histogram / np.sum(histogram)


def histogram_lab(img: np.ndarray, bins: int = 256) -> np.ndarray:
    """
    Args:
        img (np.ndarray): Image with shape (height, width) or (height, width, 3), any dtype.
        bins (int): Number of bins of the returned histogram
    Returns:
        (np.ndarray): Normalized 1D vector with 3 concatenated L*a*b* histograms.
    """
    # Convert to uint8 if needed
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)

    # Convert grayscale to RGB if needed
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=2)
    elif img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("Image must be 2D (grayscale) or 3D (RGB)")

    # Convert RGB to L*a*b*
    lab_img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    
    # Compute histogram for each L*a*b* channel separately with fixed bins
    l_hist, _ = np.histogram(lab_img[:, :, 0].flatten(), bins=bins, range=(0, 256))
    a_hist, _ = np.histogram(lab_img[:, :, 1].flatten(), bins=bins, range=(0, 256))
    b_hist, _ = np.histogram(lab_img[:, :, 2].flatten(), bins=bins, range=(0, 256))
    l_hist = l_hist.astype(np.float32)
    a_hist = a_hist.astype(np.float32)
    b_hist = b_hist.astype(np.float32)

    # Concatenate and normalize the combined histogram
    histogram = np.concatenate([l_hist, a_hist, b_hist], axis=0)
    return histogram / np.sum(histogram)


def histogram_hsv(img: np.ndarray, bins: int = 256) -> np.ndarray:
    """
    Args:
        img (np.ndarray): Image with shape (height, width) or (height, width, 3), any dtype.
        bins (int): Number of bins of the returned histogram
    Returns:
        (np.ndarray): Normalized 1D vector with 3 concatenated HSV histograms.
    """
    # Convert to uint8 if needed
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)

    # Convert grayscale to RGB if needed
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=2)
    elif img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("Image must be 2D (grayscale) or 3D (RGB)")

    # Convert RGB to HSV
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    
    # Compute histogram for each HSV channel separately with fixed bins
    # OpenCV HSV ranges: H in [0,180], S and V in [0,255]
    a, _ = np.histogram(hsv_img[:, :, 0].flatten(), bins=bins, range=(0, 180))
    b, _ = np.histogram(hsv_img[:, :, 1].flatten(), bins=bins, range=(0, 256))
    c, _ = np.histogram(hsv_img[:, :, 2].flatten(), bins=bins, range=(0, 256))
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    c = c.astype(np.float32)

    # Concatenate and normalize the combined histogram
    histogram = np.concatenate([a, b, c], axis=0)
    return histogram / np.sum(histogram)



class DescriptorMethod(Enum):
    GRAYSCALE = "grayscale"
    RGB = "rgb"
    LAB = "lab"
    HSV = "hsv"
    
    def compute(
        self, 
        img: np.ndarray, 
        bins: int = 256, 
        preprocessing: PreprocessingMethod = PreprocessingMethod.NONE, 
        **preprocessing_kwargs
    ) -> np.ndarray:
        
        """Compute the descriptor for the given image with optional preprocessing."""
        # Apply preprocessing
        if preprocessing != PreprocessingMethod.NONE:
            img = preprocessing.apply(img, **preprocessing_kwargs)
        
        if self == DescriptorMethod.GRAYSCALE:
            return histogram_grayscale(img, bins)
        elif self == DescriptorMethod.RGB:
            return histogram_rgb(img, bins)
        elif self == DescriptorMethod.LAB:
            return histogram_lab(img, bins)
        elif self == DescriptorMethod.HSV:
            return histogram_hsv(img, bins)
        else:
            raise ValueError(f"Unknown descriptor method: {self}")
    
    @property
    def name(self) -> str:
        return self.value.upper()