import cv2
import numpy as np
import matplotlib.pyplot as plt

def remove_background_by_kmeans(img: np.ndarray, k: int = 5, margin: int = 45):

    h, w = img.shape[:2]
    img_blurred = cv2.GaussianBlur(img, (7, 7), 0)
    img_lab = cv2.cvtColor(img_blurred, cv2.COLOR_BGR2Lab)

    # --- K-MEANS ---
    pixels = img_lab.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, _ = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    labels = labels.flatten().reshape((h, w))
    labels_copy = labels.copy()

    # --- FLOOD FILL FROM BORDERS ---
    border_mask = np.zeros((h, w), dtype=bool)
    border_mask[:margin, :] = True       # top
    border_mask[-margin:, :] = True      # bottom
    border_mask[:, :margin] = True       # left
    border_mask[:, -margin:] = True      # right

    for x in range(w):
        for y in list(range(0, margin)) + list(range(h-margin, h)):
            if labels_copy[y, x] != -1:
                cv2.floodFill(labels_copy, None, (x, y), -1)

    for y in range(h):
        for x in list(range(0, margin)) + list(range(w-margin, w)):
            if labels_copy[y, x] != -1:
                cv2.floodFill(labels_copy, None, (x, y), -1)

    mask = np.where(labels_copy >= 0, 1, 0)

    # --- REMOVE SMALL BLACK HOLES ---
    kernel = np.ones((10, 10), np.uint8)
    mask_dilated = cv2.dilate(mask.astype(np.uint8), kernel)
    mask_eroded = cv2.erode(mask_dilated, kernel)
    mask = mask_eroded

    # --- FILL HOLES ---
    mask[mask == 0] = 2
    cv2.floodFill(mask, None, (0, 0), 0)
    mask[mask == 2] = 1

    mask[mask == 1] = 2
    cv2.floodFill(mask, None, (w//2, h//2), 1)
    mask[mask == 2] = 0

    # --- PLOT ---
    plt.figure(figsize=(24,6))

    original_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.subplot(1,4,1)
    plt.imshow(original_rgb)
    plt.title("Original Image")
    plt.axis("off")

    # Add margin lines
    plt.plot([0, w], [margin, margin], 'r--')  # top line
    plt.plot([0, w], [h-margin, h-margin], 'r--')  # bottom line
    plt.plot([margin, margin], [0, h], 'r--')  # left line
    plt.plot([w-margin, w-margin], [0, h], 'r--')  # right line

    palette = np.array([
        [255, 0, 0],      # red
        [0, 255, 0],      # green
        [0, 0, 255],      # blue
        [255, 255, 0],    # yellow
        [255, 0, 255],    # magenta
        [0, 255, 255],    # cyan
        [255, 128, 0],    # orange
        [128, 0, 255],    # purple
        [128, 128, 128],  # gray
        [0, 128, 128],    # teal
    ], dtype=np.uint8)
    colored_clusters = palette[labels]
    plt.subplot(1,4,2)
    plt.imshow(colored_clusters)
    plt.title(f"Clusters K-Means (k={k})")
    plt.axis("off")

    plt.subplot(1,4,3)
    plt.imshow(mask, cmap='gray')
    plt.title("Binary Mask: 1 = object, 0 = background")
    plt.axis("off")

    # Apply mask to original image
    masked_img = original_rgb.copy()
    masked_img[mask == 0] = 0

    plt.subplot(1,4,4)
    plt.imshow(masked_img)
    plt.title("Image with Mask Applied")
    plt.axis("off")

    plt.show()

if __name__ == "__main__":
    import glob
    path = "data/qsd2_w2/"
    import time
    init_time = time.time()
    for image_path in glob.glob(f"{path}*.jpg"):
        img = cv2.imread(image_path)
        remove_background_by_kmeans(img)
    print("Temps total:", time.time() - init_time)
