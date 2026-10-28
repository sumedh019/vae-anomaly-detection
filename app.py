"""
app.py — Streamlit entry point (Overview / Home page).

Run with:  streamlit run app.py
"""

import streamlit as st

from src.config import COLORS, PROJECT_SUBTITLE, PROJECT_TITLE, VERSION
from src.session_state import init_state
from src.theme import LIGHT_THEME_CSS

# ──────────────────────────────────────────────
# Page configuration (must be the first Streamlit call)
# ──────────────────────────────────────────────
st.set_page_config(
    page_title=PROJECT_TITLE,
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Light-theme CSS (shared across all pages)
# ──────────────────────────────────────────────
st.markdown(LIGHT_THEME_CSS, unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Initialise session state
# ──────────────────────────────────────────────
init_state()

# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://img.icons8.com/fluency/96/factory.png",
        width=64,
    )
    st.markdown(f"### {PROJECT_TITLE}")
    st.caption(f"{PROJECT_SUBTITLE}  ·  v{VERSION}")
    st.divider()
    st.markdown("#### 📑 Navigation")
    st.markdown(
        "Use the pages in the sidebar to navigate between\n"
        "**Image Inspection**, **Model Performance**, and\n"
        "**Inspection History**."
    )

# ──────────────────────────────────────────────
# Main content — Overview page
# ──────────────────────────────────────────────
st.title("🏭 " + PROJECT_TITLE)
st.markdown(
    f"*{PROJECT_SUBTITLE}* — A Streamlit dashboard for inspecting "
    "industrial product images using a Variational Autoencoder (VAE)."
)

st.divider()

# KPI cards
st.subheader("📈 Inspection Statistics")

col1, col2, col3, col4 = st.columns(4)

total = st.session_state.total_inspections
anomalies = st.session_state.total_anomalies
normal = total - anomalies
rate = f"{(anomalies / total * 100):.1f}%" if total > 0 else "—"

col1.metric("Total Inspections", total)
col2.metric("Anomalies Detected", anomalies)
col3.metric("Normal", normal)
col4.metric("Anomaly Rate", rate)

st.divider()

# Quick-start guide
st.subheader("🚀 Quick Start")
st.markdown(
    """
1. **Navigate** to the **🔍 Image Inspection** page using the sidebar.
2. **Upload** an industrial product image (PNG, JPG, or BMP).
3. View the **Original**, **Reconstructed**, **Difference Map**, and
   **Anomaly Heatmap** panels.
4. Check the **anomaly score** against the configurable threshold.
5. Review past inspections on the **📋 Inspection History** page.
"""
)

st.info(
    "⚠️ **Note:** The VAE model has **not been trained yet**. "
    "All detection results shown are **placeholder outputs** generated "
    "from simple image-processing heuristics. They will be replaced "
    "once a real model is integrated.",
    icon="ℹ️",
)

# Footer
st.divider()
st.caption(f"v{VERSION}  ·  Built with Streamlit  ·  © 2026")

