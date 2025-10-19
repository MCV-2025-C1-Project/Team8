from matplotlib import pyplot as plt


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