"""
1_🔍_Image_Inspection.py — Upload an image and view detection results.

All outputs are clearly labelled as placeholder results until a real
VAE model is integrated.
"""

import streamlit as st

from src.anomaly_detector import detect
from src.config import ALLOWED_EXTENSIONS, COLORS, DEFAULT_THRESHOLD, PROJECT_TITLE
from src.image_processing import load_and_resize
from src.session_state import init_state, record_inspection
from src.theme import LIGHT_THEME_CSS

# ──────────────────────────────────────────────
# Page config & state
# ──────────────────────────────────────────────
st.set_page_config(page_title=f"Image Inspection · {PROJECT_TITLE}", page_icon="🔍", layout="wide")
init_state()

# ──────────────────────────────────────────────
# Light-theme CSS (shared)
# ──────────────────────────────────────────────
st.markdown(LIGHT_THEME_CSS, unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────
st.title("🔍 Image Inspection")
st.markdown("Upload an industrial product image to run anomaly detection.")

st.divider()

# ──────────────────────────────────────────────
# Upload & threshold controls
# ──────────────────────────────────────────────
col_upload, col_settings = st.columns([3, 1])

with col_upload:
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=ALLOWED_EXTENSIONS,
        help="Supported formats: PNG, JPG, JPEG, BMP",
    )

with col_settings:
    threshold = st.slider(
        "Detection Threshold",
        min_value=0.0,
        max_value=1.0,
        value=DEFAULT_THRESHOLD,
        step=0.01,
        help="Images with an anomaly score ≥ this value are flagged.",
    )

# ──────────────────────────────────────────────
# Detection pipeline
# ──────────────────────────────────────────────
if uploaded_file is not None:
    # Load image
    image = load_and_resize(uploaded_file)

    # Run detection (placeholder)
    result = detect(image, threshold=threshold)

    # Record to history
    record_inspection(
        filename=uploaded_file.name,
        score=result.anomaly_score,
        is_anomaly=result.is_anomaly,
    )

    st.divider()

    # ── Verdict banner ──────────────────────
    if result.is_anomaly:
        st.error(
            f"🚨 **ANOMALY DETECTED** — Score: **{result.anomaly_score:.4f}** "
            f"(threshold: {result.threshold:.2f})"
        )
    else:
        st.success(
            f"✅ **Normal** — Score: **{result.anomaly_score:.4f}** "
            f"(threshold: {result.threshold:.2f})"
        )

    st.warning(
        "⚠️ **Placeholder result** — The VAE model has not been trained. "
        "The reconstructed image is a Gaussian blur and the score is a "
        "heuristic based on pixel differences.",
        icon="⚠️",
    )

    # ── 2 × 2 image panel ──────────────────
    st.subheader("Inspection Panels")

    row1_col1, row1_col2 = st.columns(2)
    row2_col1, row2_col2 = st.columns(2)

    with row1_col1:
        st.markdown("##### 🖼️ Original Image")
        st.image(result.original, use_container_width=True)

    with row1_col2:
        st.markdown("##### 🔄 Reconstructed Image *(placeholder)*")
        st.image(result.reconstructed, use_container_width=True)

    with row2_col1:
        st.markdown("##### ➖ Difference Map")
        st.image(result.difference_map, use_container_width=True, clamp=True)

    with row2_col2:
        st.markdown("##### 🌡️ Anomaly Heatmap")
        st.image(result.heatmap, use_container_width=True)

    # ── Score details ───────────────────────
    st.divider()
    st.subheader("Score Details")

    m1, m2, m3 = st.columns(3)
    m1.metric("Anomaly Score", f"{result.anomaly_score:.4f}")
    m2.metric("Threshold", f"{result.threshold:.2f}")
    m3.metric("Verdict", "🚨 Anomaly" if result.is_anomaly else "✅ Normal")

else:
    st.info("👆 Upload an image above to begin inspection.", icon="📤")

