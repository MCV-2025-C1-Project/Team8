import numpy as np
import cv2
from typing import Tuple
from PIL import Image


def histogram_grayscale(img: np.ndarray, bins: int = 256) -> np.ndarray:
    """
    Args:
        img (np.ndarray): Image with shape (height, width) or (height, width, 3), any dtype.
        bins (int): Number of bins of the returned histogram
    Returns:
        (np.ndarray): Normalized 1D vector of img's histogram.
    """
    # Convert to uint8 if needed
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)

    # Convert to grayscale if RGB
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    elif img.ndim != 2:
        raise ValueError("Image must be 2D (grayscale) or 3D (RGB)")

    # Ignore fully black pixels
    mask = (img != 0).astype(np.uint8)

    # Compute histogram using OpenCV
    hist = cv2.calcHist([img], [0], mask, [bins], [0, 256])

    # Normalize
    histogram = hist.flatten().astype(np.float32)
    histogram /= np.sum(histogram)
    return histogram


def histogram_rgb(img: np.ndarray, bins: int = 256) -> np.ndarray:
    """
    Args:
        img (np.ndarray): Image with shape (height, width) or (height, width, 3), any dtype.
        bins (int): Number of bins of the returned histogram.
    Returns:
        (np.ndarray): Normalized 1D vector with 3 concatenated histograms, one per channel (R, G, B).
    """
    # Convert to uint8 if needed
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)

    # Convert grayscale to RGB if needed
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=2)
    elif img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("Image must be 2D (grayscale) or 3D (RGB)")

    # Ignore fully black pixels (0,0,0)
    mask = np.any(img != [0, 0, 0], axis=2).astype(np.uint8)

    # Compute per-channel histograms using OpenCV
    r_hist = cv2.calcHist([img], [0], mask, [bins], [0, 256])
    g_hist = cv2.calcHist([img], [1], mask, [bins], [0, 256])
    b_hist = cv2.calcHist([img], [2], mask, [bins], [0, 256])

    # Concatenate and normalize
    histogram = np.concatenate([r_hist.flatten(), g_hist.flatten(), b_hist.flatten()]).astype(np.float32)
    histogram /= np.sum(histogram)
    return histogram


def histogram_lab(img: np.ndarray, bins: int = 256) -> np.ndarray:
    """
    Args:
        img (np.ndarray): Image with shape (height, width) or (height, width, 3), any dtype.
        bins (int): Number of bins of the returned histogram.
    Returns:
        (np.ndarray): Normalized 1D vector with 3 concatenated L*a*b* histograms.
    """
    # Convert to uint8 if needed
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)

    # Convert grayscale to RGB if needed
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=2)
    elif img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("Image must be 2D (grayscale) or 3D (RGB)")

    # Convert RGB to L*a*b*
    lab_img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

    # Ignore fully black pixels
    mask = np.any(img != [0, 0, 0], axis=2).astype(np.uint8)

    # Compute per-channel histograms using OpenCV with mask
    l_hist = cv2.calcHist([lab_img], [0], mask, [bins], [0, 256])
    a_hist = cv2.calcHist([lab_img], [1], mask, [bins], [0, 256])
    b_hist = cv2.calcHist([lab_img], [2], mask, [bins], [0, 256])

    # Concatenate and normalize
    histogram = np.concatenate([l_hist.flatten(), a_hist.flatten(), b_hist.flatten()]).astype(np.float32)
    histogram /= np.sum(histogram)
    return histogram


def histogram_hsv(img: np.ndarray, bins: int = 256) -> np.ndarray:
    """
    Args:
        img (np.ndarray): Image with shape (height, width) or (height, width, 3), any dtype.
        bins (int): Number of bins of the returned histogram
    Returns:
        (np.ndarray): Normalized 1D vector with 3 concatenated HSV histograms.
    """
    # Convert to uint8 if needed
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)

    if img.ndim == 2:
        img = np.stack([img, img, img], axis=2)

    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # Ignore fully black pixels
    mask = np.any(img != [0, 0, 0], axis=2).astype(np.uint8)

    # Compute histogram for each HSV channel using OpenCV with mask
    a = cv2.calcHist([hsv_img], [0], mask, [bins], [0, 180])
    b = cv2.calcHist([hsv_img], [1], mask, [bins], [0, 256])
    c = cv2.calcHist([hsv_img], [2], mask, [bins], [0, 256])

    histogram = np.concatenate([a.flatten(), b.flatten(), c.flatten()]).astype(np.float32)
    histogram /= np.sum(histogram)
    return histogram
