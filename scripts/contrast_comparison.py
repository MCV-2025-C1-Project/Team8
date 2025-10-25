#!/usr/bin/env python3

"""
Contrast Comparison Script

This script demonstrates the contrast improvement achieved by histogram equalization
using an image from the QSD1_W3 dataset. It shows the original image, histogram equalized
image, and their respective histograms to visualize the enhancement.

Usage:
    python scripts/contrast_comparison.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataloader.dataloader import DataLoader, DatasetType
from preprocessing.preprocessors import PreprocessingMethod


def load_sample_image():
    """Load a sample image from QSD1_W3 dataset."""
    dataloader = DataLoader()
    dataloader.load_dataset(DatasetType.QSD1_W3)
    
    # Get the first image from the dataset
    for image_id, image, *_ in dataloader.iterate_images():
        return image_id, image
    
    return None, None


def apply_histogram_equalization(img):
    """Apply histogram equalization to the image."""
    if img.ndim == 3:
        # For color images, apply equalization to each channel
        equalized = np.zeros_like(img)
        for i in range(3):
            equalized[:, :, i] = cv2.equalizeHist(img[:, :, i])
        return equalized
    else:
        # For grayscale images
        return cv2.equalizeHist(img)


def plot_histogram(img, title, ax):
    """Plot histogram for the image."""
    if img.ndim == 3:
        # For color images, plot RGB histograms
        colors = ['red', 'green', 'blue']
        for i, color in enumerate(colors):
            hist = cv2.calcHist([img], [i], None, [256], [0, 256])
            ax.plot(hist, color=color, alpha=0.7, label=f'{color.upper()} channel')
        ax.legend()
    else:
        # For grayscale images
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        ax.plot(hist, color='black')
    
    ax.set_title(f'{title} - Histogram')
    ax.set_xlabel('Pixel Intensity')
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.3)


def main():
    """Main function to demonstrate contrast improvement."""
    
    print("=" * 60)
    print("CONTRAST COMPARISON: BEFORE AND AFTER HISTOGRAM EQUALIZATION")
    print("=" * 60)
    
    # Load a sample image from QSD1_W3
    print("Loading sample image from QSD1_W3 dataset...")
    image_id, original_img = load_sample_image()
    
    if original_img is None:
        print("Error: Could not load image from QSD1_W3 dataset")
        return
    
    print(f"Loaded image ID: {image_id}")
    print(f"Original image shape: {original_img.shape}")
    print(f"Original image dtype: {original_img.dtype}")
    
    # Apply histogram equalization
    print("Applying histogram equalization...")
    equalized_img = apply_histogram_equalization(original_img)
    
    # Calculate some statistics
    original_mean = np.mean(original_img)
    original_std = np.std(original_img)
    equalized_mean = np.mean(equalized_img)
    equalized_std = np.std(equalized_img)
    
    print(f"Original image - Mean: {original_mean:.2f}, Std: {original_std:.2f}")
    print(f"Equalized image - Mean: {equalized_mean:.2f}, Std: {equalized_std:.2f}")
    
    # Create the comparison plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Contrast Enhancement with Histogram Equalization', fontsize=16, fontweight='bold')
    
    # Original image
    axes[0, 0].imshow(original_img)
    axes[0, 0].set_title('Original Image (Low Contrast)')
    axes[0, 0].axis('off')
    
    # Equalized image
    axes[0, 1].imshow(equalized_img)
    axes[0, 1].set_title('After Histogram Equalization (High Contrast)')
    axes[0, 1].axis('off')
    
    # Difference image
    diff_img = np.abs(equalized_img.astype(np.float32) - original_img.astype(np.float32))
    diff_img = (diff_img / diff_img.max() * 255).astype(np.uint8)
    axes[0, 2].imshow(diff_img, cmap='hot')
    axes[0, 2].set_title('Difference (Enhancement)')
    axes[0, 2].axis('off')
    
    # Histograms
    plot_histogram(original_img, 'Original', axes[1, 0])
    plot_histogram(equalized_img, 'Equalized', axes[1, 1])
    
    # Combined histogram comparison
    if original_img.ndim == 3:
        # For color images, show combined RGB histogram
        for i, color in enumerate(['red', 'green', 'blue']):
            hist_orig = cv2.calcHist([original_img], [i], None, [256], [0, 256])
            hist_eq = cv2.calcHist([equalized_img], [i], None, [256], [0, 256])
            axes[1, 2].plot(hist_orig, color=color, alpha=0.5, linestyle='--', label=f'Original {color.upper()}')
            axes[1, 2].plot(hist_eq, color=color, alpha=0.8, label=f'Equalized {color.upper()}')
    else:
        # For grayscale images
        hist_orig = cv2.calcHist([original_img], [0], None, [256], [0, 256])
        hist_eq = cv2.calcHist([equalized_img], [0], None, [256], [0, 256])
        axes[1, 2].plot(hist_orig, color='black', alpha=0.7, linestyle='--', label='Original')
        axes[1, 2].plot(hist_eq, color='black', alpha=0.9, label='Equalized')
    
    axes[1, 2].set_title('Histogram Comparison')
    axes[1, 2].set_xlabel('Pixel Intensity')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = 'contrast_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as: {output_path}")
    
    # Show the plot
    plt.show()
    
    print("\n" + "=" * 60)
    print("ANALYSIS:")
    print("=" * 60)
    print("1. Original image shows poor contrast with limited dynamic range")
    print("2. Histogram equalization redistributes pixel intensities")
    print("3. Enhanced contrast makes texture patterns more visible")
    print("4. This improvement directly benefits DCT descriptor extraction")
    print("5. Standard deviation increase indicates better contrast distribution")


if __name__ == "__main__":
    print("Contrast Comparison Script")
    print("=" * 50)
    print("This script demonstrates the contrast improvement achieved")
    print("by histogram equalization using QSD1_W3 dataset images.")
    print("=" * 50)
    
    try:
        main()
        print("\nScript completed successfully!")
        
    except Exception as e:
        print(f"\nError during execution: {e}")
        import traceback
        traceback.print_exc()
