import cv2
import numpy as np
np.set_printoptions(precision=2)
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from enum import Enum
from typing import Optional


class BackgroundRemovalMethod(Enum):
    """Enum for background removal methods."""
    KMEANS = "kmeans"
    RECTANGLES = "rectangles"


def get_background_removal_function(method: BackgroundRemovalMethod):

    if method == BackgroundRemovalMethod.KMEANS:
        return remove_background_by_kmeans
    elif method == BackgroundRemovalMethod.RECTANGLES:
        return remove_background_by_rectangles
    else:
        raise ValueError(f"Unknown background removal method: {method}")

def remove_black_margins(mask_255: np.ndarray, processed_img: np.ndarray):
    # Iteratively strip fully-black rows/columns at the image edges (connected to margins)
    # mask_255: 0 for background, 255 for object
    while True:
        h, w = mask_255.shape[:2]
        if h == 0 or w == 0:
            break
        removed = False

        # top row
        if np.all(mask_255[0, :] == 0):
            mask_255 = mask_255[1:, :]
            processed_img = processed_img[1:, :, :]
            removed = True
            continue

        # bottom row
        if np.all(mask_255[-1, :] == 0):
            mask_255 = mask_255[:-1, :]
            processed_img = processed_img[:-1, :, :]
            removed = True
            continue

        # left column
        if np.all(mask_255[:, 0] == 0):
            mask_255 = mask_255[:, 1:]
            processed_img = processed_img[:, 1:, :]
            removed = True
            continue

        # right column
        if np.all(mask_255[:, -1] == 0):
            mask_255 = mask_255[:, :-1]
            processed_img = processed_img[:, :-1, :]
            removed = True
            continue

        if not removed:
            break
    return processed_img

def detect_split_column(kmeans_labels: np.ndarray, k: int = 100, max_changes: int = 3, visualise: bool = False) -> Optional[int]:
    h, w = kmeans_labels.shape
    mid = w // 2
    x_start = max(0, mid - k)
    x_end = min(w, mid + k)

    candidate_columns = []

    kernel = np.ones((10, 10), np.uint8)
    kmeans_labels = cv2.dilate(kmeans_labels.astype(np.uint8), kernel)

    consecutive = 0
    for x in range(x_start, x_end):
        col = kmeans_labels[:, x]
        changes = np.diff(col) != 0
        n_changes = np.sum(changes)

        if n_changes <= max_changes:
            consecutive += 1
            if consecutive == 10:
                candidate_columns.append(x-5)
        else:
            consecutive = 0

    if visualise:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 6))
        plt.imshow(kmeans_labels, cmap='tab20')
        plt.title("K-Means Labels with No Split Column Detected")
        plt.axis('off')

        if candidate_columns:
            split_x = candidate_columns[len(candidate_columns) // 2]
            plt.axvline(x=split_x, color='red', linestyle='--')

        plt.show()

    if candidate_columns:
        split_x = candidate_columns[len(candidate_columns) // 2]
        return split_x

    return None


def remove_background_by_kmeans(img: np.ndarray, k: int = 5, margin: int = 10, subdivide: bool = False, visualise: bool = True):
    h, w = img.shape[:2]
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)

    # --- K-MEANS ---
    pixels = img_lab.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, _ = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    labels = labels.flatten().reshape((h, w))
    labels_copy = labels.copy()

    if subdivide:
        split_x = detect_split_column(labels_copy, visualise=visualise)
        if split_x is not None:
            [left_mask], [left_processed] = remove_background_by_kmeans(
                img[:, :split_x, :], k, margin, subdivide=False, visualise=visualise)
            [right_mask], [right_processed] = remove_background_by_kmeans(
                img[:, split_x:, :], k, margin, subdivide=False, visualise=visualise)
            return [left_mask, right_mask], [left_processed, right_processed]

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

    if visualise:
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
    
    # Convert mask to 0-255 range for consistency
    mask_255 = (mask * 255).astype(np.uint8)
    
    # Apply mask to original image to create processed image
    processed_img = cv2.bitwise_and(img, img, mask=mask_255)

    # Remove black margins from processed image
    processed_img = remove_black_margins(mask_255, processed_img)
        
    return [mask_255], [processed_img]

def remove_background_by_rectangles(
        img: np.ndarray,
        offset: int = 40,
        h_delta: int = 20,
        s_delta: int = 60,
        v_delta:int = 60,
        visualise: bool = False,
    ):
    # convert to hsv
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    h, w = hsv.shape[:2]
    top = hsv[:offset, :, :]
    bottom = hsv[h-offset:, :, :]
    right = hsv[:, w-offset:, :]
    left = hsv[:, :offset, :]

    mean_top = np.mean(top.reshape(-1, 3), axis=0)
    mean_bottom = np.mean(bottom.reshape(-1, 3), axis=0)
    mean_left = np.mean(left.reshape(-1, 3), axis=0)
    mean_right = np.mean(right.reshape(-1, 3), axis=0)

    mean_background = np.mean([mean_top, mean_bottom, mean_left, mean_right], axis=0)

    # define thresholds
    lower_bound = np.array([
        max(mean_background[0] - h_delta, 0),
        max(mean_background[1] - s_delta, 0),
        max(mean_background[2] - v_delta, 0)
    ], dtype=np.uint8)

    upper_bound = np.array([
        min(mean_background[0] + h_delta, 179),
        min(mean_background[1] + s_delta, 255),
        min(mean_background[2] + v_delta, 255)
    ], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # background suppression stronger near edges, weaker near center
    y, x = np.indices((h, w))
    dist_edge = np.minimum.reduce([x, y, w - 1 - x, h - 1 - y]).astype(np.float32)
    # decay rate controls how fast background suppression drops toward center
    decay = 0.005
    weight = np.exp(-decay * dist_edge)
    weighted_mask = (mask.astype(np.float32) * weight).astype(np.uint8)
    # binarise weighted mask
    _, weighted_mask_bin = cv2.threshold(weighted_mask, 128, 255, cv2.THRESH_BINARY)

    # invert and apply mask
    mask_inv = cv2.bitwise_not(weighted_mask_bin)
    result_hsv = cv2.bitwise_and(hsv, hsv, mask=mask_inv)
    result_bgr = cv2.cvtColor(result_hsv, cv2.COLOR_HSV2BGR)

    result_bgr = cv2.cvtColor(result_hsv, cv2.COLOR_HSV2BGR)
    if visualise:
        plt.figure(figsize=(20, 6))
        plt.subplot(1, 5, 1)

        # original image
        plt.title("Original")
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        ax = plt.gca()
        # top
        ax.add_patch(Rectangle((0, 0), w, offset, linewidth=2, edgecolor='red', facecolor='red', alpha=0.3, label='Top'))
        # bottom
        ax.add_patch(Rectangle((0, h-offset), w, offset, linewidth=2, edgecolor='blue', facecolor='blue', alpha=0.3, label='Bottom'))
        # left
        ax.add_patch(Rectangle((0, 0), offset, h, linewidth=2, edgecolor='green', facecolor='green', alpha=0.3, label='Left'))
        # right
        ax.add_patch(Rectangle((w-offset, 0), offset, h, linewidth=2, edgecolor='yellow', facecolor='yellow', alpha=0.3, label='Right'))
        ax.legend(loc='lower right')

        # mean colours
        plt.subplot(1, 5, 2)
        plt.title("Mean colours")
        mean_colors_hsv = np.array([mean_top, mean_bottom, mean_left, mean_right, mean_background], dtype=np.uint8).reshape(1, 5, 3)
        mean_colors_bgr = cv2.cvtColor(mean_colors_hsv, cv2.COLOR_HSV2BGR)
        mean_colors_rgb = cv2.cvtColor(mean_colors_bgr, cv2.COLOR_BGR2RGB)
        plt.imshow(mean_colors_rgb)
        plt.xticks(range(5), ['Top', 'Bottom', 'Left', 'Right', 'Average'], rotation=45)
        plt.yticks([])
        plt.axis('on')

        # weighted distance
        plt.subplot(1, 5, 3)
        plt.title("Weighted distance to edge")
        plt.imshow(weight, cmap='viridis')
        plt.colorbar(label='Wegithed distance to edge')
        plt.axis('off')

        # final mask
        plt.subplot(1, 5, 4)
        plt.title("Mask")
        plt.imshow(mask_inv, cmap='gray')
        plt.axis('off')

        # result
        plt.subplot(1, 5, 5)
        plt.title("Result")
        plt.imshow(cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        plt.tight_layout()
        plt.show()
    
    # Remove black margins from result image
    result_bgr = remove_black_margins(mask_inv, result_bgr)

    return [mask_inv], [result_bgr]

if __name__ == "__main__":
    remover = "kmeans"  # "kmeans" or "rectangles"

    import glob
    path = "data/qsd2_w3/"
    import time
    init_time = time.time()
    for image_path in glob.glob(f"{path}*.jpg"):
        # image_path = path + "00018.jpg"
        img = cv2.imread(image_path)
        img = cv2.medianBlur(img, 3)
        if remover == "kmeans":
            remove_background_by_kmeans(img, subdivide=True, visualise=True)
        elif remover == "rectangles":
            remove_background_by_rectangles(img, offset=40, h_delta=20, s_delta=60, v_delta=60, visualise=True)
    print("Temps total:", time.time() - init_time)
