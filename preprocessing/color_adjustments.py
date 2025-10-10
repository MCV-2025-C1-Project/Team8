import numpy as np
import cv2
from typing import Tuple

def apply_histogram_equalization(img: np.ndarray) -> np.ndarray:
    """Apply histogram equalization to improve contrast."""
    if img.ndim == 3:
        # For color images, apply equalization to each channel
        equalized = np.zeros_like(img)
        for i in range(3):
            equalized[:, :, i] = cv2.equalizeHist(img[:, :, i])
        return equalized
    else:
        # For grayscale images
        return cv2.equalizeHist(img)

def apply_gaussian_blur(img: np.ndarray, kernel_size: Tuple[int, int] = (3, 3), sigma: float = 0) -> np.ndarray:
    """Apply Gaussian blur to reduce noise."""
    return cv2.GaussianBlur(img, kernel_size, sigma)


def apply_median_filter(img: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Apply median filter to reduce noise while preserving edges."""
    return cv2.medianBlur(img, kernel_size)


def apply_gamma_correction(img: np.ndarray, gamma: float = 0.8) -> np.ndarray:
    """Apply gamma correction to adjust brightness and contrast."""
    # Build lookup table
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
    
    # Apply gamma correction
    return cv2.LUT(img, table)