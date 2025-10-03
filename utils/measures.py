import numpy as np


def euclidean_dist(v1, v2):
    return np.sqrt(np.sum((v2-v1)**2))

def l1_dist(v1, v2):
    return np.sum(np.abs(v2-v1))

def x2_dist(v1, v2, eps=1e-10):
    return 0.5 * np.sum(((v1 - v2) ** 2) / (v1 + v2 + eps))

def hist_intersect(v1, v2):
    return np.sum(np.minimum(v1,v2))

def hellinger_kernel(v1, v2):
    return np.sum(np.sqrt(v1*v2))

if __name__ == "__main__":
    import os
    from PIL import Image
    from descriptors import histogram_1_channel, histogram_3_channels

    current_file = os.path.abspath(__file__)
    root_path = os.path.dirname(os.path.dirname(current_file))

    img1_path = os.path.join(root_path, "data", "BBDD", "bbdd_00000.jpg")
    img2_path = os.path.join(root_path, "data", "BBDD", "bbdd_00001.jpg")

    img1 = np.array(Image.open(img1_path).convert("L"), dtype=np.uint8)
    img2 = np.array(Image.open(img2_path).convert("L"), dtype=np.uint8)

    h1 = histogram_1_channel(img1, bins=256).astype(np.float32)
    h2 = histogram_1_channel(img2, bins=256).astype(np.float32)

    h1 /= np.sum(h1)
    h2 /= np.sum(h2)

    print("Euclidean distance:", euclidean_dist(h1, h2))
    print("L1 distance:", l1_dist(h1, h2))
    print("Chi-squared distance:", x2_dist(h1, h2))
    print("Histogram intersection (similarity):", hist_intersect(h1, h2))
    print("Hellinger kernel (similarity):", hellinger_kernel(h1, h2))
