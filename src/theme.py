"""
theme.py — Shared light-theme CSS injected on every page.

Centralises all styling so contrast fixes only need to happen once.
Streamlit's built-in theme (config.toml) handles primary colours;
this CSS handles the remaining elements that the theme system does
not fully control.
"""

from src.config import COLORS

LIGHT_THEME_CSS = f"""
<style>
    /* ─── Global ─────────────────────────────────────────── */
    .stApp {{
        background-color: {COLORS['background']};
        color: {COLORS['text']};
    }}

    /* ─── All text elements: force dark text ─────────────── */
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
    .stApp p, .stApp span, .stApp li, .stApp label, .stApp div {{
        color: {COLORS['text']} !important;
    }}

    /* ─── Markdown text ──────────────────────────────────── */
    .stMarkdown, .stMarkdown p, .stMarkdown span,
    .stMarkdown li, .stMarkdown a {{
        color: {COLORS['text']} !important;
    }}

    /* ─── Captions & small text ──────────────────────────── */
    .stCaption, [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] p,
    [data-testid="stCaptionContainer"] span,
    small {{
        color: {COLORS['muted']} !important;
    }}

    /* ─── Metric cards ───────────────────────────────────── */
    div[data-testid="stMetric"] {{
        background-color: #FFFFFF;
        border: 1px solid #E0E0E0;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
    }}
    div[data-testid="stMetric"] label {{
        color: {COLORS['text']} !important;
    }}
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {{
        color: {COLORS['text']} !important;
    }}
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] {{
        color: {COLORS['text']} !important;
    }}

    /* ─── Sidebar ────────────────────────────────────────── */
    section[data-testid="stSidebar"] {{
        background-color: #FFFFFF;
        border-right: 1px solid #E0E0E0;
    }}
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] li {{
        color: {COLORS['text']} !important;
    }}

    /* Sidebar navigation links */
    section[data-testid="stSidebar"] a,
    section[data-testid="stSidebar"] a span,
    section[data-testid="stSidebar"] [data-testid="stSidebarNavLink"],
    section[data-testid="stSidebar"] [data-testid="stSidebarNavLink"] span {{
        color: {COLORS['text']} !important;
    }}
    section[data-testid="stSidebar"] [data-testid="stSidebarNavLink"]:hover {{
        background-color: #F0F2F6 !important;
    }}
    section[data-testid="stSidebar"] [data-testid="stSidebarNavLink"][aria-selected="true"],
    section[data-testid="stSidebar"] [data-testid="stSidebarNavLink"][aria-selected="true"] span {{
        color: {COLORS['primary']} !important;
        font-weight: 600;
        background-color: #EBF1FF !important;
    }}

    /* ─── Widgets: slider, file uploader, selectbox ──────── */
    .stSlider label, .stSlider p, .stSlider span,
    .stFileUploader label, .stFileUploader p, .stFileUploader span,
    .stSelectbox label, .stSelectbox p,
    .stMultiSelect label, .stMultiSelect p,
    .stTextInput label, .stTextInput p,
    .stNumberInput label, .stNumberInput p {{
        color: {COLORS['text']} !important;
    }}

    /* Slider value text */
    [data-testid="stThumbValue"],
    [data-testid="stTickBarMin"],
    [data-testid="stTickBarMax"] {{
        color: {COLORS['text']} !important;
    }}

    /* ─── File uploader area ─────────────────────────────── */
    [data-testid="stFileUploader"] section {{
        background-color: #FFFFFF !important;
        border: 2px dashed #D0D0D0 !important;
        border-radius: 8px;
    }}
    [data-testid="stFileUploader"] section span,
    [data-testid="stFileUploader"] section small,
    [data-testid="stFileUploader"] section p {{
        color: {COLORS['text']} !important;
    }}

    /* ─── Alerts: info, success, warning, error ──────────── */
    div[data-testid="stAlert"] p,
    div[data-testid="stAlert"] span,
    .stAlert p, .stAlert span,
    div[role="alert"] p, div[role="alert"] span {{
        color: {COLORS['text']} !important;
    }}

    /* ─── Buttons ────────────────────────────────────────── */
    .stButton button {{
        color: {COLORS['text']} !important;
        border: 1px solid #D0D0D0;
        background-color: #FFFFFF;
    }}
    .stButton button:hover {{
        border-color: {COLORS['primary']};
        color: {COLORS['primary']} !important;
        background-color: #F0F4FF;
    }}
    .stButton button[kind="primary"],
    .stButton button[data-testid="stBaseButton-primary"] {{
        color: #FFFFFF !important;
        background-color: {COLORS['primary']};
        border: none;
    }}

    /* Download button */
    .stDownloadButton button {{
        color: {COLORS['text']} !important;
        border: 1px solid #D0D0D0;
        background-color: #FFFFFF;
    }}
    .stDownloadButton button:hover {{
        border-color: {COLORS['primary']};
        color: {COLORS['primary']} !important;
    }}

    /* ─── Data table / dataframe ─────────────────────────── */
    [data-testid="stDataFrame"],
    .stDataFrame {{
        color: {COLORS['text']} !important;
    }}

    /* ─── Dividers ───────────────────────────────────────── */
    hr {{
        border-color: #E0E0E0 !important;
    }}

    /* ─── Tooltip / help icon ────────────────────────────── */
    [data-testid="stTooltipIcon"] {{
        color: {COLORS['muted']} !important;
    }}

    /* ─── Tab labels ─────────────────────────────────────── */
    .stTabs [data-baseweb="tab"] {{
        color: {COLORS['text']} !important;
    }}
    .stTabs [aria-selected="true"] {{
        color: {COLORS['primary']} !important;
    }}

    /* ─── Expander ───────────────────────────────────────── */
    [data-testid="stExpander"] summary span,
    [data-testid="stExpander"] p {{
        color: {COLORS['text']} !important;
    }}
</style>
"""

