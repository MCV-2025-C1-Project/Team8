import numpy as np
from PIL import Image
from utils.dct import DCT_2D, DCT_conv, zigzag_scan
from utils.patches import cut_image_into_8x8_patches

dct_2D_patches = DCT_2D(bsize=8)

def dct_descriptor(img: np.ndarray, n_coefficients: int) -> np.ndarray:
    assert 0 < n_coefficients <= 64, "n_coefficients must be in the range [1, 64]"
    
    # Resize to default 640x640
    img = Image.fromarray(img).convert("L")
    img = img.resize((640, 640), Image.BILINEAR)
    img = np.array(img)

    img_patches = cut_image_into_8x8_patches(img)
    
    descriptor = np.zeros(shape=len(img_patches)*n_coefficients, dtype=np.int32)
    for i, img_patch in enumerate(img_patches):
        dct_coefficients = DCT_conv(img_patch, dct_2D_patches)
        # Flatten the 8x8 DCT coefficients to a 64-length vector with zigzag order
        start = i * n_coefficients
        end = start + n_coefficients
        descriptor[start:end] = zigzag_scan(dct_coefficients)[:n_coefficients]
    
    return np.array(descriptor)


def lbp_descriptor(img: np.ndarray, radius: int = 1, n_neighbors: int = 8, lbp_method: str = 'uniform') -> np.ndarray:
    """
    Local Binary Pattern (LBP) texture descriptor - optimized version.
    
    Args:
        img: Input image as numpy array
        radius: Radius of the circle around each pixel (default: 1)
        n_neighbors: Number of neighbors to sample around each pixel (default: 8)
        lbp_method: LBP method - 'uniform' or 'nri_uniform' (default: 'uniform')
    
    Returns:
        LBP histogram as numpy array
    """
    # Convert to grayscale if needed and resize to default 640x640
    if len(img.shape) == 3:
        img = Image.fromarray(img).convert("L")
    else:
        img = Image.fromarray(img)
    
    img = img.resize((640, 640), Image.BILINEAR)
    img = np.array(img, dtype=np.float32)
    
    # Get image dimensions
    height, width = img.shape
    
    # Pre-compute sampling coordinates for efficiency
    angles = np.linspace(0, 2 * np.pi, n_neighbors, endpoint=False)
    x_offsets = np.round(radius * np.cos(angles)).astype(int)
    y_offsets = np.round(radius * np.sin(angles)).astype(int)
    
    # Initialize LBP image
    lbp_image = np.zeros((height, width), dtype=np.uint8)
    
    # LBP computation with vectorized operations
    for y in range(radius, height - radius):
        for x in range(radius, width - radius):
            center = img[y, x]
            
            # Get all neighbors at once using vectorized indexing
            neighbor_x = x + x_offsets
            neighbor_y = y + y_offsets
            
            # Handle boundary conditions efficiently
            valid_mask = (neighbor_x >= 0) & (neighbor_x < width) & (neighbor_y >= 0) & (neighbor_y < height)
            neighbor_values = np.where(valid_mask, 
                                     img[neighbor_y, neighbor_x], 
                                     center)
            
            # Vectorized comparison and bit conversion
            binary_pattern = (neighbor_values >= center).astype(int)
            
            # Convert to decimal using bit shifting (much faster than string conversion)
            lbp_value = 0
            for i, bit in enumerate(binary_pattern):
                lbp_value += bit * (2 ** (n_neighbors - 1 - i))
            
            # Apply uniform pattern reduction if specified
            if lbp_method == 'uniform':
                lbp_value = _uniform_pattern_fast(lbp_value, n_neighbors)
            
            lbp_image[y, x] = lbp_value
    
    # Create histogram
    if lbp_method == 'uniform':
        # For uniform patterns, we have n_neighbors + 2 bins (0, 1, ..., n_neighbors, non-uniform)
        n_bins = n_neighbors + 2
    else:
        # For non-uniform patterns, we have 2^n_neighbors bins
        n_bins = 2 ** n_neighbors
    
    histogram, _ = np.histogram(lbp_image.flatten(), bins=n_bins, range=(0, n_bins))
    
    # Normalize histogram
    histogram = histogram.astype(np.float32)
    histogram = histogram / (np.sum(histogram) + 1e-8)  # Add small epsilon to avoid division by zero
    
    return histogram


def _uniform_pattern_fast(lbp_value: int, n_neighbors: int) -> int:
    """
    Fast uniform pattern conversion using bit operations.
    A uniform pattern is one that has at most 2 transitions between 0 and 1.
    """
    # Count transitions using bit operations (much faster than string operations)
    transitions = 0
    for i in range(n_neighbors):
        current_bit = (lbp_value >> (n_neighbors - 1 - i)) & 1
        next_bit = (lbp_value >> (n_neighbors - 1 - ((i + 1) % n_neighbors))) & 1
        if current_bit != next_bit:
            transitions += 1
    
    # If more than 2 transitions, it's non-uniform
    if transitions > 2:
        return n_neighbors + 1  # Non-uniform pattern bin
    else:
        return lbp_value
