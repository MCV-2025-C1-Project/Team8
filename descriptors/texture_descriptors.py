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


def lbp_descriptor(img: np.ndarray,
                   radius: int = 1,
                   n_neighbors: int = 8,
                   lbp_method: str = 'uniform') -> np.ndarray:
    """
    Local Binary Pattern (LBP) texture descriptor - fast & simple.
    - Vectorized (no Python loops over pixels).
    - Optional 'nri_uniform' mapping.
    """

    # ---- to grayscale, resize like your original ----
    if img.ndim == 3:
        img = Image.fromarray(img).convert("L")
    else:
        img = Image.fromarray(img)
    img = img.resize((640, 640), Image.BILINEAR)
    img = np.asarray(img, dtype=np.float32)

    H, W = img.shape

    # ---- precompute integer offsets on circle (same as your rounding) ----
    angles = np.linspace(0, 2*np.pi, n_neighbors, endpoint=False)
    dx = np.rint(radius * np.cos(angles)).astype(int)
    dy = np.rint(radius * np.sin(angles)).astype(int)

    # ---- pad once; slice for each neighbor (fast) ----
    pad = int(radius) + 1  # safe margin for shifts
    padded = np.pad(img, pad, mode='edge')

    neighbors = []
    for k in range(n_neighbors):
        y0 = pad + dy[k]
        x0 = pad + dx[k]
        neigh = padded[y0:y0+H, x0:x0+W]
        neighbors.append(neigh)
    neigh_stack = np.stack(neighbors, axis=0)               # (P, H, W)

    # ---- threshold vs center, pack bits to integer code image ----
    center = img[None, :, :]                                # (1, H, W)
    bits = (neigh_stack >= center).astype(np.uint32)        # (P, H, W)
    weights = (1 << np.arange(n_neighbors, dtype=np.uint32))[:, None, None]
    code_img = (bits * weights).sum(axis=0)                 # (H, W), uint32

    # ---- histogram mapping ----
    if lbp_method == 'uniform':  # RIU2: map uniform to popcount (0..P), non-uniform -> P+1
        n_bins = n_neighbors + 2

        lut_size = 1 << n_neighbors
        codes = np.arange(lut_size, dtype=np.uint32)

        # bit matrix: columns are bit positions (LSB-first)
        bm = ((codes[:, None] >> np.arange(n_neighbors, dtype=np.uint32)) & 1).astype(np.uint8)
        ones = bm.sum(axis=1)
        # circular transitions
        trans = (bm[:, :-1] != bm[:, 1:]).sum(axis=1) + (bm[:, -1] != bm[:, 0])

        lut = np.where(trans <= 2, ones, n_neighbors + 1).astype(np.uint16)
        mapped = lut[code_img]
        hist, _ = np.histogram(mapped, bins=n_bins, range=(0, n_bins))

    elif lbp_method == 'nri_uniform':
        # Non-rotation-invariant uniform: unique id per uniform code, last bin for non-uniform
        lut_size = 1 << n_neighbors
        codes = np.arange(lut_size, dtype=np.uint32)
        bm = ((codes[:, None] >> np.arange(n_neighbors, dtype=np.uint32)) & 1).astype(np.uint8)
        trans = (bm[:, :-1] != bm[:, 1:]).sum(axis=1) + (bm[:, -1] != bm[:, 0])

        uniform_codes = np.where(trans <= 2)[0]
        U = len(uniform_codes)
        lut = np.full(lut_size, U, dtype=np.uint16)
        lut[uniform_codes] = np.arange(U, dtype=np.uint16)

        mapped = lut[code_img]
        n_bins = U + 1
        hist, _ = np.histogram(mapped, bins=n_bins, range=(0, n_bins))

    else:
        # raw codes (2^P bins)
        n_bins = 1 << n_neighbors
        hist, _ = np.histogram(code_img, bins=n_bins, range=(0, n_bins))

    # ---- normalize ----
    hist = hist.astype(np.float32)
    hist /= (hist.sum() + 1e-8)
    return hist
