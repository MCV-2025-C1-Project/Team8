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


def lbp_descriptor(img: np.ndarray, radius: int = 1, n_neighbors: int = 8, method: str = 'uniform') -> np.ndarray:
    """
    Local Binary Pattern (LBP) texture descriptor.
    
    Args:
        img: Input image as numpy array
        radius: Radius of the circle around each pixel (default: 1)
        n_neighbors: Number of neighbors to sample around each pixel (default: 8)
        method: LBP method - 'uniform' or 'nri_uniform' (default: 'uniform')
    
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
    
    # Initialize LBP image
    lbp_image = np.zeros((height, width), dtype=np.uint8)
    
    # Calculate LBP for each pixel
    for y in range(radius, height - radius):
        for x in range(radius, width - radius):
            center = img[y, x]
            binary_string = ""
            
            # Sample points around the center pixel
            for i in range(n_neighbors):
                angle = 2 * np.pi * i / n_neighbors
                x_offset = int(round(radius * np.cos(angle)))
                y_offset = int(round(radius * np.sin(angle)))
                
                neighbor_x = x + x_offset
                neighbor_y = y + y_offset
                
                # Handle boundary conditions
                if 0 <= neighbor_x < width and 0 <= neighbor_y < height:
                    neighbor_value = img[neighbor_y, neighbor_x]
                else:
                    neighbor_value = center  # Use center value for out-of-bounds
                
                # Create binary string
                binary_string += "1" if neighbor_value >= center else "0"
            
            # Convert binary string to decimal
            lbp_value = int(binary_string, 2)
            
            # Apply uniform pattern reduction if specified
            if method == 'uniform':
                lbp_value = _uniform_pattern(lbp_value, n_neighbors)
            
            lbp_image[y, x] = lbp_value
    
    # Create histogram
    if method == 'uniform':
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


def _uniform_pattern(lbp_value: int, n_neighbors: int) -> int:
    """
    Convert LBP value to uniform pattern.
    A uniform pattern is one that has at most 2 transitions between 0 and 1.
    """
    # Convert to binary string with leading zeros
    binary_str = format(lbp_value, f'0{n_neighbors}b')
    
    # Count transitions
    transitions = 0
    for i in range(n_neighbors):
        if binary_str[i] != binary_str[(i + 1) % n_neighbors]:
            transitions += 1
    
    # If more than 2 transitions, it's non-uniform
    if transitions > 2:
        return n_neighbors + 1  # Non-uniform pattern bin
    else:
        return lbp_value