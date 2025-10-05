import numpy as np
import cv2
from typing import Tuple
from enum import Enum

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


class PreprocessingMethod(Enum):
    """Simple preprocessing methods for image retrieval."""
    
    NONE = "none"
    GAMMA = "gamma"
    GAUSSIAN = "gaussian"
    MEDIAN = "median"
    
    def apply(self, img: np.ndarray, **kwargs) -> np.ndarray:
        """Apply the preprocessing method to the image."""
        if self == PreprocessingMethod.NONE:
            return img
        elif self == PreprocessingMethod.GAMMA:
            gamma = kwargs.get('gamma', 0.8)
            return apply_gamma_correction(img, gamma)
        elif self == PreprocessingMethod.GAUSSIAN:
            kernel_size = kwargs.get('kernel_size', (3, 3))
            sigma = kwargs.get('sigma', 0)
            return apply_gaussian_blur(img, kernel_size, sigma)
        elif self == PreprocessingMethod.MEDIAN:
            kernel_size = kwargs.get('kernel_size', 3)
            return apply_median_filter(img, kernel_size)
        else:
            raise ValueError(f"Unknown preprocessing method: {self}")
    
    @property
    def name(self) -> str:
        """Get the display name of the preprocessing method."""
        return self.value.upper()
