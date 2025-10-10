import numpy as np
from typing import Callable, List
from preprocessing.color_adjustments import PreprocessingMethod
from utils.spatial import split_image_blocks

def block_histogram(
    img: np.ndarray,
    color_function: Callable[[np.ndarray, int], np.ndarray],
    ns_blocks: List[int],
    bins: int = 256,
    preprocessing: PreprocessingMethod = PreprocessingMethod.NONE,
    **preprocessing_kwargs
) -> np.ndarray:
    combined_histogram = []

    for n_blocks in ns_blocks:
        # Split image into n x n blocks
        blocks = split_image_blocks(img, n_blocks)

        for block in blocks:
            # Compute histogram for each block and concatenate them
            histogram = color_function(block, bins)
            combined_histogram = np.concatenate((combined_histogram, histogram))

    return combined_histogram / np.sum(combined_histogram)  # Normalize histogram


def spatial_pyramid_histogram(
    img: np.ndarray,
    color_function: Callable[[np.ndarray, int], np.ndarray],
    max_level: int = 2,
    bins: int = 256,
    preprocessing: PreprocessingMethod = PreprocessingMethod.NONE,
    **preprocessing_kwargs
) -> np.ndarray:

    pyramid_histograms = []

    for level in range(max_level + 1):
        # Calculate number of blocks for this level: 2^level x 2^level
        n_blocks = 2 ** level
        
        if level == 0:
            # Level 0: Global histogram (entire image)
            histogram = color_function(img, bins)
            pyramid_histograms.append(histogram)
        else:
            # Level > 0: Split into blocks and compute histogram for each
            blocks = split_image_blocks(img, n_blocks)
            
            for block in blocks:
                histogram = color_function(block, bins)
                pyramid_histograms.append(histogram)
    
    # Concatenate all histograms from all levels
    combined_histogram = np.concatenate(pyramid_histograms)
    
    return combined_histogram / np.sum(combined_histogram)