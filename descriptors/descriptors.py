from enum import Enum
from typing import List
from preprocessing.color_adjustments import PreprocessingMethod
from descriptors.color_histograms import histogram_grayscale, histogram_rgb, histogram_lab, histogram_hsv
from descriptors.spatial_histograms import block_histogram, spatial_pyramid_histogram



class DescriptorMethod(Enum):
    # Color histograms
    GRAYSCALE = "grayscale"
    RGB = "rgb"
    LAB = "lab"
    HSV = "hsv"
    
    # Block histograms
    GRAYSCALE_BLOCKS = "grayscale_blocks"
    RGB_BLOCKS = "rgb_blocks"
    LAB_BLOCKS = "lab_blocks"
    HSV_BLOCKS = "hsv_blocks"
    
    # Spatial pyramid histograms
    GRAYSCALE_PYRAMID = "grayscale_pyramid"
    RGB_PYRAMID = "rgb_pyramid"
    LAB_PYRAMID = "lab_pyramid"
    HSV_PYRAMID = "hsv_pyramid"
    
    def _get_base_color_function(self):
        """Get the base color histogram function for spatial methods."""        
        if "grayscale" in self.value:
            return histogram_grayscale
        elif "rgb" in self.value:
            return histogram_rgb
        elif "lab" in self.value:
            return histogram_lab
        elif "hsv" in self.value:
            return histogram_hsv
        else:
            raise ValueError(f"Unknown color method in: {self.value}")
    
    def compute(
        self, 
        img, 
        bins: int = 256, 
        preprocessing: PreprocessingMethod = PreprocessingMethod.NONE,
        # Spatial parameters
        ns_blocks: List[int] = None,
        max_level: int = 2,
        **preprocessing_kwargs
    ):
        """Compute the descriptor for the given image with optional preprocessing."""
        
        if preprocessing != PreprocessingMethod.NONE:
            img = preprocessing.apply(img, **preprocessing_kwargs)
        
        # Simple color histograms - delegate to color_histograms module
        if self.is_color_only:
            # Map to color function
            color_function_map = {
                DescriptorMethod.GRAYSCALE: histogram_grayscale,
                DescriptorMethod.RGB: histogram_rgb,
                DescriptorMethod.LAB: histogram_lab,
                DescriptorMethod.HSV: histogram_hsv,
            }
            
            color_function = color_function_map[self]
            return color_function(img, bins)
        
        elif self.is_block_histogram:
            if ns_blocks is None:
                ns_blocks = [1, 2, 3]  # Default block configuration
            
            base_function = self._get_base_color_function()
            return block_histogram(img, base_function, ns_blocks, bins, preprocessing, **preprocessing_kwargs)
        
        elif self.is_pyramid_histogram:
            # Get base color function
            base_function = self._get_base_color_function()
            return spatial_pyramid_histogram(img, base_function, max_level, bins, preprocessing, **preprocessing_kwargs)
        
        else:
            raise ValueError(f"Unknown descriptor method: {self}")
    
    @property
    def name(self) -> str:
        return self.value.upper()
    
    @property
    def is_spatial(self) -> bool:
        return self.is_block_histogram or self.is_pyramid_histogram
    
    @property
    def is_color_only(self) -> bool:
        return not self.is_spatial
    
    @property
    def is_block_histogram(self) -> bool:
        return self.value.endswith("_blocks")
    
    @property
    def is_pyramid_histogram(self) -> bool:
        return self.value.endswith("_pyramid")
    
    @property
    def base_color_method(self) -> str:
        if "grayscale" in self.value:
            return "grayscale"
        elif "rgb" in self.value:
            return "rgb"
        elif "lab" in self.value:
            return "lab"
        elif "hsv" in self.value:
            return "hsv"
        else:
            return "unknown"
