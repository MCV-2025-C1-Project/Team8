import numpy as np


def get_ck(k: int, N: int) -> float:
    if k == 0:
        return np.sqrt(1/N)
    else:
        return np.sqrt(2/N)

def get_basis_functions(bsize: int) -> np.ndarray:
    """
    Input: number of basis functions
    Output: 'bsize' 1D basis functions
    """
    N = bsize
    basis_functions = np.zeros(shape=(bsize, bsize))
    for k in range(bsize):
        for n in range(N):
            ck = get_ck(k, N)
            basis_functions[k,n] = ck * np.cos( (np.pi * k * ((2*n) + 1)) / (2*N) )
    return basis_functions


def DCT_2D(bsize):
    """
    Input: the size of the 2D image patch
    Output: dct2, 64 2D DCT basis functions, each basis function with a dimesion of 8x8.
    """
    basis_functions = get_basis_functions(bsize=bsize)
    # Outer product of basis functions
    image_patches = np.zeros(shape=(bsize, bsize, bsize, bsize))
    # Shape explanation:
    #    first two dimensions (bsize x bsize) are the 64 (8 x 8) image patches
    #    last two dimensions (bsize x bsize) are the actual 64 (8 x 8) values within each image patch
    for u in range(bsize):
        for v in range(bsize):
            outer_product = np.outer(basis_functions[u], basis_functions[v])
            image_patches[u,v] = outer_product
    return image_patches


def zigzag_scan(coefficients):
    rows, cols = coefficients.shape
    result = []
    for s in range(rows + cols - 1):
        if s % 2 == 0:
            for i in range(s + 1):
                j = s - i
                if i < rows and j < cols:
                    result.append(coefficients[i][j])
        else:
            for i in range(s + 1):
                j = s - i
                if j < rows and i < cols:
                    result.append(coefficients[j][i])
    return result

def DCT_conv_slow(image_patch, dct_2D_patches):
    # First, scale to range [-128, 128]
    image_patch -= 128
    # To get each coefficient, we multiply the image patch element-wise to its
    # corresponding DCT patch. From the resulting matrix, we sum all of its elements
    # and we get the coefficient at position (i,j).
    coefficients = np.zeros(shape=(dct_2D_patches.shape[0], dct_2D_patches.shape[1]))
    for i in range(dct_2D_patches.shape[0]):
        for j in range(dct_2D_patches.shape[1]):
            coefficients[i,j] = sum(sum(np.multiply(image_patch, dct_2D_patches[i,j])))
    # Round the coefficient to the nearest integer
    return coefficients.astype(int)

def DCT_conv(image_patch, dct_2D_patches):
    # Scale to range [-128, 128]
    image_patch = image_patch - 128

    # Compute elementwise multiplication between image_patch and all DCT patches
    # dct_2D_patches has shape (M, N, h, w)
    # image_patch has shape (h, w)
    # We expand image_patch to broadcast across all patches
    products = dct_2D_patches * image_patch[None, None, :, :]

    # Sum over the spatial dimensions (h, w)
    coefficients = products.sum(axis=(-1, -2))

    # Round to nearest integer and convert to int
    return np.rint(coefficients).astype(int)