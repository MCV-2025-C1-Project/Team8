from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np


# Helper function: crop a centered region
def crop_center(img, zoom_factor):
    """Crop a centered square region to zoom into details."""
    h, w = img.shape[:2]
    crop_h, crop_w = h // zoom_factor, w // zoom_factor
    start_y = h // 2 - crop_h // 2
    start_x = w // 2 - crop_w // 2
    return img[start_y:start_y + crop_h, start_x:start_x + crop_w]

def plot_psnr(
        original_img,
        noisy_img,
        filtered_img,
        kernel_sizes,
        psnr_values,
        title="Average filter PSNR assessment wrt. kernel size",
        zoom_factor=4,
    ):
    """Plot PSNR against kernel sizes for Average filter with zoomed-in image regions."""

    zoom_original = crop_center(original_img, zoom_factor)
    zoom_noisy = crop_center(noisy_img, zoom_factor)
    zoom_filtered = crop_center(filtered_img, zoom_factor)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    axes[0, 0].imshow(zoom_original, cmap='gray')
    axes[0, 0].set_title(f"Original (Zoom x{zoom_factor})")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(zoom_noisy, cmap='gray')
    axes[0, 1].set_title(f"Noisy (Zoom x{zoom_factor})")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(zoom_filtered, cmap='gray')
    axes[1, 0].set_title(f"Filtered (Kernel Size={kernel_sizes[-1]}, Zoom x{zoom_factor})")
    axes[1, 0].axis("off")

    axes[1, 1].plot(kernel_sizes[:len(psnr_values)], psnr_values, marker='o', linestyle='-', color='b')
    axes[1, 1].set_title("PSNR vs. Kernel Size")
    axes[1, 1].set_xlabel("Kernel Size")
    axes[1, 1].set_ylabel("PSNR (dB)")
    axes[1, 1].grid(True)

    # Highlight best PSNR
    max_psnr = max(psnr_values)
    best_kernel_size = kernel_sizes[psnr_values.index(max_psnr)]
    axes[1, 1].scatter(best_kernel_size, max_psnr, color='red',
                       label=f'Best kernel_size=({best_kernel_size},{best_kernel_size})\nPSNR={max_psnr:.2f} dB')
    axes[1, 1].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_gaussian_psnr(
        original_img,
        noisy_img,
        filtered_img,
        sigma_values,
        psnr_values,
        title="Gaussian filter PSNR assessment wrt. sigma",
        zoom_factor=4,
    ):
    """Plot PSNR against sigma values for Gaussian filter with zoomed-in image regions."""

    zoom_original = crop_center(original_img, zoom_factor)
    zoom_noisy = crop_center(noisy_img, zoom_factor)
    zoom_filtered = crop_center(filtered_img, zoom_factor)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    axes[0, 0].imshow(zoom_original, cmap='gray')
    axes[0, 0].set_title(f"Original (Zoom x{zoom_factor})")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(zoom_noisy, cmap='gray')
    axes[0, 1].set_title(f"Noisy (Zoom x{zoom_factor})")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(zoom_filtered, cmap='gray')
    axes[1, 0].set_title(f"Filtered (σ={sigma_values[-1]:.2f}, Zoom x{zoom_factor})")
    axes[1, 0].axis("off")

    axes[1, 1].plot(sigma_values[:len(psnr_values)], psnr_values, marker='o', linestyle='-', color='b')
    axes[1, 1].set_title("PSNR vs. σ")
    axes[1, 1].set_xlabel("Sigma (σ)")
    axes[1, 1].set_ylabel("PSNR (dB)")
    axes[1, 1].grid(True)

    # Highlight best PSNR
    max_psnr = max(psnr_values)
    best_sigma = sigma_values[psnr_values.index(max_psnr)]
    axes[1, 1].scatter(best_sigma, max_psnr, color='red',
                       label=f'Best σ={best_sigma:.2f}\nPSNR={max_psnr:.2f} dB')
    axes[1, 1].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_kp_results(query_image, query_kp, index_image, index_kp, good_matches, homography_matches, H, scale=0.5):
    """
    Visualize keypoint matches and draw the outline of the index image
    projected onto the query image using homography.
    """
    query_image = cv.cvtColor(query_image, cv.COLOR_RGB2BGR)
    index_image = cv.cvtColor(index_image, cv.COLOR_RGB2BGR)

    matched_vis = cv.drawMatches(
        query_image, query_kp,
        index_image, index_kp,
        good_matches, None,
        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    
    for m in good_matches:
        pt1 = tuple(np.round(query_kp[m.queryIdx].pt).astype(int))
        pt2 = tuple(np.round(index_kp[m.trainIdx].pt).astype(int))
        pt2 = (pt2[0] + query_image.shape[1], pt2[1])
        cv.line(matched_vis, pt1, pt2, (0, 0, 255), 2, cv.LINE_AA)

    for m in homography_matches:
        pt1 = tuple(np.round(query_kp[m.queryIdx].pt).astype(int))
        pt2 = tuple(np.round(index_kp[m.trainIdx].pt).astype(int))
        pt2 = (pt2[0] + query_image.shape[1], pt2[1])
        cv.line(matched_vis, pt1, pt2, (0, 255, 0), 2, cv.LINE_AA)

    if H is not None:
        # Invert homography to map index to query
        H_inv = np.linalg.inv(H)

        h, w = index_image.shape[:2]
        scale = 500/w
        index_corners = np.float32([
            [0, 0], [0, h - 1],
            [w - 1, h - 1], [w - 1, 0]
        ]).reshape(-1, 1, 2)


        dst_corners = cv.perspectiveTransform(index_corners, H_inv)
        query_with_poly = cv.polylines(
            query_image.copy(),
            [np.int32(dst_corners)],
            True,
            (0, 255, 0), 3, cv.LINE_AA
        )

        def resize(img, scale):
            return cv.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))

        matched_vis_small = resize(matched_vis, scale)
        query_with_poly_small = resize(query_with_poly, scale)

        # Show the results
        cv.imshow("Query with Index Outline", query_with_poly_small)
        cv.imshow("Matches (Query vs. Index)", matched_vis_small)
        cv.waitKey(0)
        cv.destroyAllWindows()