import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cv2
import numpy as np
from dataloader.dataloader import DatasetType, DataLoader

def is_noisy(img: np.ndarray, noise_threshold: float = 13.0) -> bool:
    """
    Detect noise using Gaussian blur difference.
    Works well for sensor noise (most common in photos).
    """
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # Use Gaussian blur for detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
    noise = cv2.absdiff(gray, blurred)
    noise_level = np.std(noise)
    
    return noise_level > noise_threshold


if __name__ == "__main__":
    dataset = DataLoader()
    dataset.load_dataset(DatasetType.QSD1_W4)

    noisy_count = 0
    total_count = 0

    for image_id, image, *_ in dataset.iterate_images():
        total_count += 1
        if is_noisy(image):
            noisy_count += 1
            print(f"Image ID {image_id} is detected as noisy.")
        else:
            print(f"Image ID {image_id} is NOT noisy.")

    print(f"Total images: {total_count}, Noisy images: {noisy_count}")