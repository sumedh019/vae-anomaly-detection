"""
image_processing.py — Helper functions for image loading, diffing, and heatmaps.

These are pure utility functions with no Streamlit dependency so they
can be unit-tested independently.
"""

import cv2
import numpy as np
from PIL import Image

from src.config import IMAGE_SIZE


def load_and_resize(uploaded_file) -> np.ndarray:
    """Read an uploaded file into a NumPy RGB array resized to IMAGE_SIZE."""
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize(IMAGE_SIZE, Image.LANCZOS)
    return np.array(image)


def generate_placeholder_reconstruction(image: np.ndarray) -> np.ndarray:
    """
    Create a *fake* reconstructed image by applying a Gaussian blur.

    This is a **placeholder only** — in production this would be the
    output of the trained VAE decoder.
    """
    blurred = cv2.GaussianBlur(image, (21, 21), sigmaX=5)
    return blurred


def compute_difference_map(original: np.ndarray, reconstructed: np.ndarray) -> np.ndarray:
    """
    Compute the absolute per-pixel difference between two images.

    Returns a single-channel (grayscale) difference image.
    """
    # Convert to float to avoid uint8 underflow
    diff = np.abs(original.astype(np.float32) - reconstructed.astype(np.float32))
    # Average across colour channels → grayscale
    diff_gray = np.mean(diff, axis=2).astype(np.uint8)
    return diff_gray


def compute_heatmap(difference_map: np.ndarray) -> np.ndarray:
    """
    Apply a colour-map to the grayscale difference map to produce an
    anomaly heatmap (RGB).
    """
    # Normalise to 0-255 for the colour-map
    norm = cv2.normalize(difference_map, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = cv2.applyColorMap(norm.astype(np.uint8), cv2.COLORMAP_JET)
    # OpenCV uses BGR; convert to RGB for display
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return heatmap_rgb


def compute_anomaly_score(difference_map: np.ndarray) -> float:
    """
    Derive a scalar anomaly score in [0, 1] from the difference map.

    Currently uses a simple mean-intensity heuristic.  A trained model
    would replace this with a proper latent-space distance.
    """
    score = float(np.mean(difference_map) / 255.0)
    return round(min(score, 1.0), 4)

