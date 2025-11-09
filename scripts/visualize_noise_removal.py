import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import matplotlib.pyplot as plt
import cv2
import numpy as np
from dataloader.dataloader import DataLoader, DatasetType
from preprocessing.noise_detector import is_noisy

# Parameters
IMAGE_ID = 25  # Change this to visualize other images
NOISE_THRESHOLD = 13.0

# Load dataset and image
loader = DataLoader()
loader.load_dataset(DatasetType.QSD1_W4)

# Find image with IMAGE_ID
image = None
for img_id, img, *_ in loader.iterate_images():
    if img_id == IMAGE_ID:
        image = img
        break
if image is None:
    raise ValueError(f"Image ID {IMAGE_ID} not found in QSD1_W4 dataset.")

# Convert to grayscale
if image.ndim == 3:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
else:
    gray = image

# Apply Gaussian blur
blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
noise_map = cv2.absdiff(gray, blurred)
noise_level = np.std(noise_map)


# Plot only grayscale, blurred, and noise map
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].imshow(gray, cmap='gray')
axes[0].set_title('Grayscale')
axes[0].axis('off')

axes[1].imshow(blurred, cmap='gray')
axes[1].set_title('Blurred (Gaussian)')
axes[1].axis('off')

im = axes[2].imshow(noise_map, cmap='hot')
axes[2].set_title(f'Noise Map\nNoise Level: {noise_level:.2f}')
axes[2].axis('off')
fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

plt.suptitle(f'Noise Detection Example (Image ID {IMAGE_ID})', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

print(f"Noise level for image {IMAGE_ID}: {noise_level:.2f} (Threshold: {NOISE_THRESHOLD})")
print(f"Is image noisy? {'Yes' if noise_level > NOISE_THRESHOLD else 'No'}")
