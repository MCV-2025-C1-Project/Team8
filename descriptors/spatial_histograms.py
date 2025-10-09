import numpy as np
from descriptors.color_histograms import DescriptorMethod
from preprocessing.color_adjustments import PreprocessingMethod
from utils.spatial import split_image_blocks

def block_histogram(
    img: np.ndarray,
    method: DescriptorMethod,
    ns_blocks: list,
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
            histogram = method.compute(block, bins=bins, preprocessing=preprocessing, **preprocessing_kwargs)
            combined_histogram = np.concatenate((combined_histogram, histogram))

    return combined_histogram / np.sum(combined_histogram)  # Normalize histogram
