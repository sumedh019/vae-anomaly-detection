"""
3_📋_Inspection_History.py — Browse, export, and clear past inspections.

Inspection records are stored in st.session_state and persist for the
duration of the Streamlit server session.
"""

import io

import pandas as pd
import streamlit as st

from src.config import COLORS, PROJECT_TITLE
from src.session_state import clear_history, init_state
from src.theme import LIGHT_THEME_CSS

# ──────────────────────────────────────────────
st.set_page_config(page_title=f"Inspection History · {PROJECT_TITLE}", page_icon="📋", layout="wide")
init_state()

st.markdown(LIGHT_THEME_CSS, unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────
st.title("📋 Inspection History")
st.markdown("Review all inspections performed during this session.")

st.divider()

# ──────────────────────────────────────────────
# History table
# ──────────────────────────────────────────────
history = st.session_state.inspection_history

if history:
    df = pd.DataFrame(history)
    # Reorder columns nicely
    df = df[["timestamp", "filename", "anomaly_score", "verdict"]]
    df.columns = ["Timestamp", "Filename", "Anomaly Score", "Verdict"]

    st.dataframe(df, use_container_width=True, hide_index=True)

    # ── Summary metrics ─────────────────────
    st.divider()
    st.subheader("📈 Session Summary")

    s1, s2, s3, s4 = st.columns(4)
    total = len(df)
    n_anomaly = int((df["Verdict"].str.contains("Anomaly")).sum())
    n_normal = total - n_anomaly
    avg_score = df["Anomaly Score"].mean()

    s1.metric("Total Inspections", total)
    s2.metric("Anomalies", n_anomaly)
    s3.metric("Normal", n_normal)
    s4.metric("Avg Score", f"{avg_score:.4f}")

    # ── Actions ─────────────────────────────
    st.divider()
    action_col1, action_col2, _ = st.columns([1, 1, 3])

    with action_col1:
        # Download CSV
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv_buffer.getvalue(),
            file_name="inspection_history.csv",
            mime="text/csv",
        )

    with action_col2:
        if st.button("🗑️ Clear History", type="secondary"):
            clear_history()
            st.rerun()

else:
    st.info(
        "No inspections yet. Go to the **🔍 Image Inspection** page "
        "to upload an image and run detection.",
        icon="📭",
    )

