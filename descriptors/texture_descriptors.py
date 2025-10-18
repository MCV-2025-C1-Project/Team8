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

