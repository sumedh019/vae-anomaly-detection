"""
anomaly_detector.py — High-level interface for anomaly detection.

This module wraps the image-processing helpers behind a single
``detect()`` call that the Streamlit pages use.  When a real VAE
model is ready, only this file needs to change.
"""

from dataclasses import dataclass

import numpy as np

from src.config import DEFAULT_THRESHOLD
from src.image_processing import (
    compute_anomaly_score,
    compute_difference_map,
    compute_heatmap,
    generate_placeholder_reconstruction,
)


@dataclass
class DetectionResult:
    """Container for everything produced by a single inspection."""
    original: np.ndarray
    reconstructed: np.ndarray
    difference_map: np.ndarray
    heatmap: np.ndarray
    anomaly_score: float
    threshold: float
    is_anomaly: bool
    is_placeholder: bool = True        # Always True until a real model is loaded


def detect(image: np.ndarray, threshold: float = DEFAULT_THRESHOLD) -> DetectionResult:
    """
    Run the full detection pipeline on a single image.

    Parameters
    ----------
    image : np.ndarray
        RGB image array (already resized).
    threshold : float
        Decision boundary — scores above this are flagged as anomalies.

    Returns
    -------
    DetectionResult
        All outputs needed by the dashboard.
    """
    reconstructed = generate_placeholder_reconstruction(image)
    diff_map = compute_difference_map(image, reconstructed)
    heatmap = compute_heatmap(diff_map)
    score = compute_anomaly_score(diff_map)

    return DetectionResult(
        original=image,
        reconstructed=reconstructed,
        difference_map=diff_map,
        heatmap=heatmap,
        anomaly_score=score,
        threshold=threshold,
        is_anomaly=(score >= threshold),
    )

