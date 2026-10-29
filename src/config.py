"""
config.py — Central configuration for the project.

All constants, thresholds, and styling values live here so they are
easy to find and change in one place.
"""

# ──────────────────────────────────────────────
# Project metadata
# ──────────────────────────────────────────────
PROJECT_TITLE = "VAE-Based Industrial Anomaly Detection"
PROJECT_SUBTITLE = "Final-Year B.Tech AIML Project"
VERSION = "0.1.0"

# ──────────────────────────────────────────────
# Anomaly detection defaults
# ──────────────────────────────────────────────
DEFAULT_THRESHOLD = 0.5          # Default anomaly threshold (0-1)
IMAGE_SIZE = (256, 256)          # Width × Height for the Streamlit dashboard

# ──────────────────────────────────────────────
# Dataset / model training
# ──────────────────────────────────────────────
DATASET_ROOT = r"C:\Users\HP\Downloads\mvtec_anomaly_detection\bottle"
MODEL_IMAGE_SIZE = (128, 128)    # Width × Height for VAE training & inference
BATCH_SIZE = 32
NUM_WORKERS = 0                  # 0 = main process (safe default on Windows)

# ──────────────────────────────────────────────
# VAE model architecture
# ──────────────────────────────────────────────
LATENT_DIM = 128                 # Dimensionality of the latent space z
ENCODER_CHANNELS = [32, 64, 128, 256]   # Conv channels per encoder stage
KL_WEIGHT = 1.0                  # β weight for KL divergence (β-VAE)
LEARNING_RATE = 1e-3             # Adam learning rate
NUM_EPOCHS = 50                  # Training epochs (unused until training step)

# ──────────────────────────────────────────────
# Light-theme colour palette (used in charts & UI)
# ──────────────────────────────────────────────
COLORS = {
    "primary":    "#4F8BF9",     # Blue
    "success":    "#2ECC71",     # Green
    "danger":     "#E74C3C",     # Red
    "warning":    "#F39C12",     # Orange
    "background": "#F8F9FA",     # Light grey
    "text":       "#2C3E50",     # Dark slate
    "muted":      "#95A5A6",     # Grey
}

# ──────────────────────────────────────────────
# Supported upload formats
# ──────────────────────────────────────────────
ALLOWED_EXTENSIONS = ["png", "jpg", "jpeg", "bmp"]

