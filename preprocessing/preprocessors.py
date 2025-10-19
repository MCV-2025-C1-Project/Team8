import numpy as np
import cv2
from typing import Tuple, Callable
from enum import Enum
from preprocessing.color_adjustments import (
    apply_histogram_equalization,
    apply_average_filter,
    apply_gaussian_blur, 
    apply_median_filter, 
    apply_gamma_correction
)
from preprocessing.background_removers import (
    remove_background_by_kmeans, 
    remove_background_by_rectangles,
    BackgroundRemovalMethod
)


class PreprocessingMethod(Enum):
    """Comprehensive preprocessing methods for image retrieval."""
    
    # No preprocessing
    NONE = "none"
    
    # Color adjustments
    GAMMA = "gamma"
    HIST_EQ = "hist_eq"
    AVERAGE = "average"
    GAUSSIAN = "gaussian"
    MEDIAN = "median"
    
    # Background removal methods
    BG_KMEANS = "bg_kmeans"
    BG_RECTANGLES = "bg_rectangles"
    
    def apply(self, img: np.ndarray, **kwargs) -> np.ndarray:
        """Apply the preprocessing method to the image."""
        if self == PreprocessingMethod.NONE:
            return img
        elif self == PreprocessingMethod.HIST_EQ:
            return apply_histogram_equalization(img)
        elif self == PreprocessingMethod.GAMMA:
            gamma = kwargs.get('gamma', 0.8)
            return apply_gamma_correction(img, gamma)
        elif self == PreprocessingMethod.AVERAGE:
            kernel_size = kwargs.get('kernel_size', (3, 3))
            return apply_average_filter(img, kernel_size)
        elif self == PreprocessingMethod.GAUSSIAN:
            kernel_size = kwargs.get('kernel_size', (3, 3))
            sigma = kwargs.get('sigma', 0)
            return apply_gaussian_blur(img, kernel_size, sigma)
        elif self == PreprocessingMethod.MEDIAN:
            kernel_size = kwargs.get('kernel_size', 3)
            return apply_median_filter(img, kernel_size)
        elif self == PreprocessingMethod.BG_KMEANS:
            k = kwargs.get('k', 5)
            margin = kwargs.get('margin', 45)
            return remove_background_by_kmeans(img, k, margin)[1]  # Return only processed image
        elif self == PreprocessingMethod.BG_RECTANGLES:
            offset = kwargs.get('offset', 50)
            h_delta = kwargs.get('h_delta', 20)
            s_delta = kwargs.get('s_delta', 60)
            v_delta = kwargs.get('v_delta', 60)
            visualise = kwargs.get('visualise', False)
            return remove_background_by_rectangles(img, offset, h_delta, s_delta, v_delta, visualise)[1]  # Return only processed image
        else:
            raise ValueError(f"Unknown preprocessing method: {self}")
    
    @property
    def name(self) -> str:
        """Get the display name of the preprocessing method."""
        return self.value.upper()
    
    @property
    def is_background_removal(self) -> bool:
        """Check if this is a background removal method."""
        return self.value.startswith("bg_")
    
    @property
    def is_color_adjustment(self) -> bool:
        """Check if this is a color adjustment method."""
        return self.value in ["gamma", "hist_eq", "gaussian", "median"]
    
    @property
    def is_no_preprocessing(self) -> bool:
        """Check if this is no preprocessing."""
        return self.value == "none"
    
    def get_background_removal_method(self) -> BackgroundRemovalMethod:
        """Get the corresponding BackgroundRemovalMethod enum."""
        if self == PreprocessingMethod.BG_KMEANS:
            return BackgroundRemovalMethod.KMEANS
        elif self == PreprocessingMethod.BG_RECTANGLES:
            return BackgroundRemovalMethod.RECTANGLES
        else:
            raise ValueError(f"Not a background removal method: {self}")
