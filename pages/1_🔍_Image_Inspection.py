"""
1_🔍_Image_Inspection.py — Upload an image and view detection results.

Runs the VAE anomaly detection pipeline on uploaded images and visualizes
the reconstruction, difference map, and anomaly heatmap.
"""

import streamlit as st

from src.anomaly_detector import detect, CHECKPOINT_PATH
from src.checkpoint_upload import save_uploaded_checkpoint
from src.calibration import load_calibration
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

checkpoint_path = CHECKPOINT_PATH
uploaded_model = None
with st.expander("Model checkpoint", expanded=not CHECKPOINT_PATH.is_file()):
    st.caption("Upload your trained vae_bottle.pth if it is missing from the deployment. Uploads are temporary for this browser session; upload again after a restart.")
    uploaded_model = st.file_uploader("Trained VAE checkpoint", type=["pth", "pt", "ckpt"], key="model_upload")
    if uploaded_model is not None:
        try:
            checkpoint_path = save_uploaded_checkpoint(uploaded_model.getvalue(), st.session_state)
            st.success("Uploaded checkpoint loaded successfully.")
        except Exception as exc:
            st.error(f"Cannot use checkpoint: {exc}")
            st.stop()
    elif not checkpoint_path.is_file():
        st.warning("Upload the trained checkpoint above to enable inspection. The image alone cannot supply model weights.")

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
    try:
        calibration = load_calibration() if uploaded_model is None else None
    except (OSError, ValueError) as exc:
        calibration = None
        st.warning(f"Calibration unavailable: {exc}")
    manual = st.checkbox("Use experimental manual threshold", value=False)
    threshold = st.slider(
        "Detection Threshold",
        min_value=0.0,
        max_value=1.0,
        value=calibration["threshold"] if calibration else DEFAULT_THRESHOLD,
        disabled=not manual,
        step=0.0001,
        format="%.4f",
        help="Images with an anomaly score ≥ this value are flagged.",
    )

    can_classify = calibration is not None or manual
    if not can_classify:
        st.warning("No calibrated threshold. Scores are exploratory; no normal/defect verdict will be issued.")
    elif manual:
        st.warning("Manual threshold: experimental verdict, not a validated defect decision.")
    else:
        st.caption("Threshold calibrated on known-good images. Validate defect recall separately.")

# ──────────────────────────────────────────────
# Detection pipeline
# ──────────────────────────────────────────────
if uploaded_file is not None:
    # Load image
    image = load_and_resize(uploaded_file)

    # Run detection
    try:
        result = detect(image, threshold=threshold, checkpoint_path=checkpoint_path)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        st.error(f"Inspection unavailable: {exc}")
        st.stop()

    verdict = (
        ("🚨 Anomaly" if result.is_anomaly else "✅ Normal")
        if can_classify else "Uncalibrated — review required"
    )
    if can_classify:
        # Do not add duplicate records when unrelated widgets rerun the page.
        import hashlib
        inspection_key = (str(checkpoint_path), hashlib.sha256(uploaded_file.getvalue()).hexdigest(), threshold)
        if st.session_state.get("last_inspection_key") != inspection_key:
            record_inspection(
                filename=uploaded_file.name,
                score=result.anomaly_score,
                is_anomaly=result.is_anomaly,
            )
            st.session_state.last_inspection_key = inspection_key

    st.divider()
    message = (
        f"**{verdict}** — Score: **{result.anomaly_score:.4f}** "
        f"(threshold: {result.threshold:.4f})"
    )
    if not can_classify:
        st.warning(message)
    elif result.is_anomaly:
        st.error(message)
    else:
        st.info(message + ". This does not certify that the product is defect-free.")
    st.caption(
        "VAE reconstruction error is not a defect probability. Heatmap colours show "
        "relative reconstruction differences and do not independently establish a defect. "
        "Use the same product category, framing and lighting as the training data."
    )

    # ── 2 × 2 image panel ──────────────────
    st.subheader("Inspection Panels")

    row1_col1, row1_col2 = st.columns(2)
    row2_col1, row2_col2 = st.columns(2)

    with row1_col1:
        st.markdown("##### 🖼️ Original Image")
        st.image(result.original, use_container_width=True)

    with row1_col2:
        recon_title = (
            "##### 🔄 Reconstructed Image (placeholder)"
            if result.is_placeholder
            else "##### 🔄 Reconstructed Image (VAE)"
        )
        st.markdown(recon_title)
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
    m2.metric("Threshold", f"{result.threshold:.4f}")
    m3.metric("Verdict", verdict)

else:
    st.info("👆 Upload an image above to begin inspection.", icon="📤")

