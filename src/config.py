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
IMAGE_SIZE = (256, 256)          # Width × Height that images are resized to

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

