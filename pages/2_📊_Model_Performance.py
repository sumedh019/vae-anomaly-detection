"""
2_📊_Model_Performance.py — Placeholder evaluation charts.

All charts use synthetic data and are clearly marked as placeholders.
They will be replaced once the VAE model is trained.
"""

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from src.config import COLORS, PROJECT_TITLE
from src.session_state import init_state
from src.theme import LIGHT_THEME_CSS

# ──────────────────────────────────────────────
st.set_page_config(page_title=f"Model Performance · {PROJECT_TITLE}", page_icon="📊", layout="wide")
init_state()

st.markdown(LIGHT_THEME_CSS, unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────
st.title("📊 Model Performance")
st.markdown("Evaluation metrics and training curves for the VAE model.")

st.warning(
    "⚠️ **The VAE model has not been trained yet.** "
    "All charts below contain **randomly generated placeholder data** "
    "to demonstrate the dashboard layout. They do **not** represent "
    "real model performance.",
    icon="⚠️",
)

st.divider()

# ──────────────────────────────────────────────
# Metrics table
# ──────────────────────────────────────────────
st.subheader("📋 Classification Metrics")

metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
metrics_col1.metric("Accuracy",  "N/A", help="Model not trained")
metrics_col2.metric("Precision", "N/A", help="Model not trained")
metrics_col3.metric("Recall",    "N/A", help="Model not trained")
metrics_col4.metric("F1 Score",  "N/A", help="Model not trained")

st.divider()

# ──────────────────────────────────────────────
# Placeholder charts
# ──────────────────────────────────────────────
chart_col1, chart_col2 = st.columns(2)

# --- ROC Curve (synthetic) ---
with chart_col1:
    st.subheader("ROC Curve *(placeholder)*")

    np.random.seed(42)
    fpr = np.sort(np.concatenate([[0], np.random.uniform(0, 1, 50), [1]]))
    tpr = np.sort(np.concatenate([[0], np.random.uniform(0, 1, 50), [1]]))
    # Make it look like a reasonable ROC curve
    tpr = np.clip(tpr + 0.15, 0, 1)
    tpr = np.sort(tpr)

    fig_roc, ax_roc = plt.subplots(figsize=(5, 4))
    ax_roc.plot(fpr, tpr, color=COLORS["primary"], linewidth=2, label="VAE (placeholder)")
    ax_roc.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.4, label="Random")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curve — PLACEHOLDER DATA", fontsize=11)
    ax_roc.legend(loc="lower right")
    ax_roc.set_facecolor("#FAFAFA")
    fig_roc.patch.set_facecolor(COLORS["background"])
    fig_roc.tight_layout()
    st.pyplot(fig_roc)

# --- Training Loss Curve (synthetic) ---
with chart_col2:
    st.subheader("Training Loss *(placeholder)*")

    epochs = np.arange(1, 51)
    train_loss = 0.8 * np.exp(-0.06 * epochs) + np.random.normal(0, 0.015, len(epochs))
    val_loss = 0.85 * np.exp(-0.055 * epochs) + np.random.normal(0, 0.02, len(epochs))

    fig_loss, ax_loss = plt.subplots(figsize=(5, 4))
    ax_loss.plot(epochs, train_loss, color=COLORS["primary"], linewidth=2, label="Train Loss")
    ax_loss.plot(epochs, val_loss, color=COLORS["danger"], linewidth=2, label="Val Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Training Loss — PLACEHOLDER DATA", fontsize=11)
    ax_loss.legend()
    ax_loss.set_facecolor("#FAFAFA")
    fig_loss.patch.set_facecolor(COLORS["background"])
    fig_loss.tight_layout()
    st.pyplot(fig_loss)

st.divider()

# ──────────────────────────────────────────────
# Reconstruction samples placeholder
# ──────────────────────────────────────────────
st.subheader("🖼️ Reconstruction Samples *(placeholder)*")

st.info(
    "Once the model is trained, this section will display side-by-side "
    "comparisons of original images and their VAE reconstructions from "
    "the test set.",
    icon="📸",
)

# Generate tiny random images to show the layout
sample_cols = st.columns(4)
for i, col in enumerate(sample_cols):
    with col:
        dummy = np.random.randint(100, 200, (64, 64, 3), dtype=np.uint8)
        st.image(dummy, caption=f"Sample {i+1} (random noise)", use_container_width=True)

st.divider()
st.caption("All data on this page is synthetic. Train the VAE to see real metrics.")

