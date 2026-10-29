"""
anomaly_detector.py — High-level interface for anomaly detection.

Uses the trained convolutional VAE to reconstruct an input image, then
computes the anomaly score from the mean absolute reconstruction error.

The ``detect()`` function is the single entry point consumed by the
Streamlit dashboard pages.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.config import (
    DEFAULT_THRESHOLD,
    ENCODER_CHANNELS,
    LATENT_DIM,
    MODEL_IMAGE_SIZE,
)
from src.image_processing import compute_difference_map, compute_heatmap
from src.model import VAE

# ──────────────────────────────────────────────
# Checkpoint path (project-relative)
# ──────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = _PROJECT_ROOT / "models" / "vae_bottle.pth"

# ──────────────────────────────────────────────
# Module-level model cache
# ──────────────────────────────────────────────
_cached_model: Optional[VAE] = None
_cached_key = None


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
    is_placeholder: bool = True        # False when genuine VAE inference succeeded


# ──────────────────────────────────────────────
# Model loading (cached)
# ──────────────────────────────────────────────
def _load_model(checkpoint_path: Optional[Path] = None) -> VAE:
    """Load the trained VAE from *checkpoint_path* and cache it.

    Raises
    ------
    FileNotFoundError
        If the checkpoint file does not exist.
    RuntimeError
        If the checkpoint is corrupt or incompatible with the model.
    """
    global _cached_model, _cached_key
    checkpoint_path = Path(checkpoint_path or CHECKPOINT_PATH).resolve()
    key = (str(checkpoint_path), checkpoint_path.stat().st_mtime_ns) if checkpoint_path.is_file() else None
    if _cached_model is not None and key is not None and key == _cached_key:
        return _cached_model

    # ── Validate path ─────────────────────────
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"VAE checkpoint not found at: {checkpoint_path}\n"
            "Train the model first with:  python -m src.train"
        )

    # ── Load checkpoint ───────────────────────
    try:
        # Try weights_only=True first (safer, PyTorch >= 2.0)
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True,
        )
    except Exception as exc:
        raise RuntimeError("Checkpoint could not be loaded safely. Upload a PyTorch state dictionary or a checkpoint produced by src.train.") from exc

    # ── Extract state dict ────────────────────
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            # The dict itself might be a raw state_dict
            state_dict = checkpoint
    else:
        raise RuntimeError(
            f"Unexpected checkpoint format (type={type(checkpoint).__name__}). "
            "Expected a dict with 'model_state_dict' or 'state_dict' key, "
            "or a raw state dictionary."
        )

    # ── Build and load model ──────────────────
    # Use architecture config from checkpoint if available, else defaults
    latent_dim = (
        checkpoint.get("latent_dim", LATENT_DIM)
        if isinstance(checkpoint, dict) else LATENT_DIM
    )
    encoder_channels = (
        checkpoint.get("encoder_channels", ENCODER_CHANNELS)
        if isinstance(checkpoint, dict) else ENCODER_CHANNELS
    )

    model = VAE(
        in_channels=3,
        latent_dim=latent_dim,
        channel_list=encoder_channels,
    )

    try:
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Checkpoint is incompatible with the VAE architecture.\n"
            f"  Checkpoint : {checkpoint_path}\n"
            f"  latent_dim : {latent_dim}\n"
            f"  channels   : {encoder_channels}\n"
            f"  Error      : {exc}"
        ) from exc

    model.eval()
    _cached_model = model
    _cached_key = key
    return model


def _clear_model_cache() -> None:
    """Clear the cached model (useful for testing)."""
    global _cached_model
    _cached_model = None


# ──────────────────────────────────────────────
# Preprocessing
# ──────────────────────────────────────────────
_to_model_tensor = transforms.Compose([
    transforms.Resize(MODEL_IMAGE_SIZE),
    transforms.ToTensor(),  # HWC uint8 -> CHW float [0, 1]
])


def _preprocess(image: np.ndarray) -> torch.Tensor:
    """Convert an RGB NumPy array to a (1, 3, 128, 128) tensor in [0, 1]."""
    pil_image = Image.fromarray(image.astype(np.uint8)).convert("RGB")
    tensor = _to_model_tensor(pil_image)          # (3, 128, 128)
    return tensor.unsqueeze(0)                     # (1, 3, 128, 128)


# ──────────────────────────────────────────────
# Postprocessing
# ──────────────────────────────────────────────
def _tensor_to_numpy(tensor: torch.Tensor, target_size: tuple) -> np.ndarray:
    """Convert a (1, 3, H, W) float tensor to an RGB uint8 NumPy array.

    The result is resized to *target_size* ``(width, height)`` for
    dashboard display.
    """
    # (1, 3, H, W) -> (3, H, W) -> (H, W, 3)
    img_np = tensor.squeeze(0).permute(1, 2, 0).numpy()
    img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
    # Resize to original dashboard dimensions
    pil_img = Image.fromarray(img_np)
    pil_img = pil_img.resize(target_size, Image.LANCZOS)
    return np.array(pil_img)


# ──────────────────────────────────────────────
# Anomaly score
# ──────────────────────────────────────────────
def _compute_anomaly_score(
    original: np.ndarray, reconstructed: np.ndarray,
) -> float:
    """Mean absolute reconstruction error on normalised [0, 1] values.

    Both inputs are uint8 RGB arrays of the same shape.
    """
    diff = np.abs(
        original.astype(np.float32) / 255.0
        - reconstructed.astype(np.float32) / 255.0
    )
    score = float(np.mean(diff))
    return round(min(max(score, 0.0), 1.0), 4)


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────
def detect(
    image: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
    checkpoint_path: Optional[Path] = None,
) -> DetectionResult:
    """
    Run the full detection pipeline on a single image.

    Parameters
    ----------
    image : np.ndarray
        RGB image array (typically 256 x 256 from the dashboard).
    threshold : float
        Decision boundary — scores at or above this are flagged as anomalies.

    Returns
    -------
    DetectionResult
        All outputs needed by the dashboard, with ``is_placeholder=False``.

    Raises
    ------
    FileNotFoundError
        If the model checkpoint does not exist.
    RuntimeError
        If the checkpoint is corrupt or incompatible.
    """
    # ── Load model (cached) ───────────────────
    model = _load_model(checkpoint_path)

    # ── Preprocess ────────────────────────────
    input_tensor = _preprocess(image)               # (1, 3, 128, 128)

    # ── Inference ─────────────────────────────
    with torch.inference_mode():
        # Use the latent mean: inspection must not sample a different z on each rerun.
        mu, _logvar = model.encode(input_tensor)
        recon_tensor = model.decode(mu)

    # ── Postprocess ───────────────────────────
    # target_size = (width, height) matching the original input
    h, w = image.shape[:2]
    target_size = (w, h)
    reconstructed = _tensor_to_numpy(recon_tensor, target_size)

    # ── Difference map & heatmap ──────────────
    diff_map = compute_difference_map(image, reconstructed)
    heatmap = compute_heatmap(diff_map)

    # ── Anomaly score ─────────────────────────
    score = _compute_anomaly_score(image, reconstructed)

    return DetectionResult(
        original=image,
        reconstructed=reconstructed,
        difference_map=diff_map,
        heatmap=heatmap,
        anomaly_score=score,
        threshold=threshold,
        is_anomaly=(score >= threshold),
        is_placeholder=False,
    )
