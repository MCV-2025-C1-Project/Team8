import numpy as np


def cut_image_into_8x8_patches(img: np.ndarray):
    '''
    # Cuts image into pieces
    # Input: a black and white frame
    # Output the input, cut into 8x8 pixel patches organized in a 'pile'
    # of patches. Estimated dimensions: many x 8 x 8
    '''
    if img.ndim == 3:
        image_patches_a = cut_image_into_8x8_patches(img[:, :, 0])
        image_patches_b = cut_image_into_8x8_patches(img[:, :, 1])
        image_patches_c = cut_image_into_8x8_patches(img[:, :, 2])
        return np.concatenate((image_patches_a, image_patches_b, image_patches_c), axis=0)
    elif img.ndim == 2:
        width, height = img.shape
        assert width % 8 == 0 and height % 8 == 0
        n_patches = int(img.shape[0]/8) * int(img.shape[1]/8)
        image_patches = np.zeros(shape=(n_patches, 8, 8))
        counter = 0
        for i in range(0, width, 8):
            for j in range(0, height, 8):
                image_patches[counter] = img[i:i+8, j:j+8]
                counter += 1
        return image_patches
