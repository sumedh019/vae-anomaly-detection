"""
session_state.py — Helpers for managing Streamlit session state.

Centralises initialisation and access so that every page does not
have to repeat boilerplate.
"""

from datetime import datetime

import streamlit as st


def init_state():
    """Initialise session-state keys if they do not already exist."""
    defaults = {
        "inspection_history": [],   # list of dicts
        "total_inspections": 0,
        "total_anomalies": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def record_inspection(filename: str, score: float, is_anomaly: bool):
    """Append an inspection record and update running counters."""
    st.session_state.inspection_history.append(
        {
            "filename": filename,
            "anomaly_score": round(score, 4),
            "verdict": "🚨 Anomaly" if is_anomaly else "✅ Normal",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    st.session_state.total_inspections += 1
    if is_anomaly:
        st.session_state.total_anomalies += 1


def clear_history():
    """Reset all inspection data."""
    st.session_state.inspection_history = []
    st.session_state.total_inspections = 0
    st.session_state.total_anomalies = 0
    st.session_state.pop("last_inspection_key", None)

