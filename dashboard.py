import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

# ============================================================
# 1) FILE PATHS
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

RAW_PATH = str(DATA_DIR / "NOAA Earthquaqe since 1600.csv")
CLEAN_PATH = str(DATA_DIR / "earthquakes_clean.csv")

st.set_page_config(
    page_title="MCI Simulation Dashboard",
    page_icon="🌏",
    layout="wide"
)


# ============================================================
# 2) REFINED UI STYLES (IMPROVED UI/UX — SAME COLOR PALETTE)
# ============================================================
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

      html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
      }

      /* Keep palette */
      .stApp { background: #f8fafc; color: #1e293b; }

      /* Tighter, dashboard-like */
      .block-container {
        padding-top: 1.0rem;
        padding-bottom: 1.25rem;
        max-width: 1320px;
      }

      section[data-testid="stSidebar"] { display: none; }

      /* Modern Card Design */
      .card {
        background: #ffffff;
        border: 1px solid #f1f5f9;
        border-radius: 16px;
        padding: 18px;                 /* slightly tighter */
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        margin-bottom: 14px;
        transition: transform 0.18s ease, box-shadow 0.18s ease;
      }
      .card:hover {
        transform: translateY(-1px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08);
      }

      /* Accent lines (same palette) */
      .accent-red { border-top: 4px solid #ef4444; }
      .accent-blue { border-top: 4px solid #3b82f6; }
      .accent-dark { border-top: 4px solid #1e293b; }

      /* Header Styling */
      .title-container {
        display: flex;
        align-items: center;
        margin-bottom: 0.9rem;
        background: white;
        padding: 0.95rem 1.2rem;
        border-radius: 16px;
        border: 1px solid #e2e8f0;
      }

      .pulse {
        width: 14px; height: 14px; border-radius: 50%;
        background: #ef4444;
        margin-right: 14px;
        animation: shadow-pulse 2s infinite;
      }
      @keyframes shadow-pulse {
        0% { box-shadow: 0 0 0 0px rgba(239, 68, 68, 0.35); }
        70% { box-shadow: 0 0 0 10px rgba(239, 68, 68, 0); }
        100% { box-shadow: 0 0 0 0px rgba(239, 68, 68, 0); }
      }

      /* KPI Styling */
      .kpi-title { 
        color:#64748b; font-size:0.82rem; font-weight:700; 
        text-transform: uppercase; letter-spacing: 0.04em; 
      }
      .kpi-value { font-size:1.65rem; font-weight:800; color: #0f172a; margin: 4px 0; }
      .kpi-sub { color:#94a3b8; font-size:0.85rem; }

      /* Alerts (FIXED border-color colon bug) */
      .alert {
        border-radius: 12px; padding: 16px;
        display:flex; gap:14px; align-items:center;
        margin-bottom: 14px; border: 1px solid;
      }
      .alert-severe { background: #fef2f2; border-color: #fee2e2; color: #991b1b; }
      .alert-moderate { background: #fffbeb; border-color: #fef3c7; color: #92400e; }
      .alert-low { background: #f0fdf4; border-color: #dcfce7; color: #166534; }

      /* Make control widgets feel grouped & tighter */
      div[data-testid="stSlider"] { margin-bottom: 0.55rem; }
      div[data-testid="stSelectbox"] { margin-bottom: 0.55rem; }
      div[data-testid="stCheckbox"] { margin-bottom: 0.25rem; }
      div[data-testid="stToggle"] { margin-top: 0.25rem; margin-bottom: 0.6rem; }

      /* Plotly (rounded + consistent) */
      .stPlotlyChart > div {
        border-radius: 14px !important;
        overflow: hidden !important;
        border: 1px solid #f1f5f9;
      }

      /* Dataframe inside card - prevent "floating" look */
      div[data-testid="stDataFrame"] {
        border-radius: 12px !important;
        overflow: hidden !important;
        border: 1px solid #e2e8f0 !important;
      }

      /* Download button should visually belong to card */
      div[data-testid="stDownloadButton"] { margin-top: 10px !important; }
      div[data-testid="stDownloadButton"] button {
        width: 100%;
        background-color: #ffffff !important;
        color: #1e293b !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        padding: 0.65rem 0.9rem !important;
      }
      div[data-testid="stDownloadButton"] button:hover {
        border-color: #3b82f6 !important;
        color: #3b82f6 !important;
      }

      /* Reduce random top whitespace above Streamlit elements */
      .stMarkdown { margin-bottom: 0.25rem; }
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# 3) DATA PROCESSING
# ============================================================
def auto_clean_raw_to_csv(raw_path: str, clean_path: str) -> None:
    """
    Clean NOAA raw CSV and create a cleaned CSV used by the app.
    - Detects tsunami column automatically and keeps it in output.
    - Standardizes column names and types.
    """
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"NOAA CSV not found: {raw_path}")

    df = pd.read_csv(raw_path)
    df.columns = [c.strip() for c in df.columns]

    # Detect ANY tsunami column name (e.g., Tsunami, Flag Tsunami, etc.)
    tsunami_candidates = [c for c in df.columns if "tsunami" in c.lower()]
    if tsunami_candidates:
        df = df.rename(columns={tsunami_candidates[0]: "tsunami_flag"})

    rename_map = {
        "Id": "event_id",
        "Year": "year",
        "Month": "month",
        "Date": "day",
        "Name": "location_name",
        "Latitude": "latitude",
        "Longitude": "longitude",
        "Focal Depth (km)": "depth_km",
        "Magnitude": "magnitude",
        "Deaths": "deaths",
        "Injuries": "injuries",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    required = ["year", "month", "day", "latitude", "longitude", "depth_km", "magnitude", "deaths", "injuries"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}\nAvailable: {list(df.columns)}")

    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["month"] = df["month"].fillna(1).astype(int)
    df["day"] = df["day"].fillna(1).astype(int)
    df["date"] = pd.to_datetime(dict(year=df["year"], month=df["month"], day=df["day"]), errors="coerce")

    df["location_name"] = df.get("location_name", "Unknown").astype(str).str.strip()

    df["deaths"] = df["deaths"].fillna(0).astype(int)
    df["injuries"] = df["injuries"].fillna(0).astype(int)
    df["total_casualties"] = df["deaths"] + df["injuries"]

    if "tsunami_flag" in df.columns:
        df["tsunami_flag"] = (
            df["tsunami_flag"]
            .astype(str)
            .str.strip()
            .str.lower()
            .replace({
                "y": 1, "yes": 1, "true": 1, "1": 1, "t": 1,
                "n": 0, "no": 0, "false": 0, "0": 0, "f": 0,
                "nan": 0, "none": 0, "": 0
            })
        )
        df["tsunami_flag"] = pd.to_numeric(df["tsunami_flag"], errors="coerce").fillna(0).astype(int)

    df = df.dropna(subset=["date", "year", "latitude", "longitude", "magnitude", "depth_km"])

    final_cols = [
        "event_id", "date", "year", "location_name",
        "latitude", "longitude", "magnitude", "depth_km",
        "deaths", "injuries", "total_casualties", "tsunami_flag"
    ]
    df = df[[c for c in final_cols if c in df.columns]].sort_values("date")

    os.makedirs(os.path.dirname(clean_path), exist_ok=True)
    df.to_csv(clean_path, index=False)

@st.cache_data
def load_data(clean_path: str) -> pd.DataFrame:
    df = pd.read_csv(clean_path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    numeric_cols = ["year", "latitude", "longitude", "magnitude", "depth_km", "deaths", "injuries", "total_casualties"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["date", "year", "latitude", "longitude", "magnitude", "depth_km"])
    df["year"] = df["year"].astype(int)

    if "tsunami_flag" in df.columns:
        df["tsunami_flag"] = pd.to_numeric(df["tsunami_flag"], errors="coerce").fillna(0).astype(int)

    return df

if not os.path.exists(CLEAN_PATH):
    auto_clean_raw_to_csv(RAW_PATH, CLEAN_PATH)
    st.cache_data.clear()

df = load_data(CLEAN_PATH)

# ============================================================
# 4) HEADER
# ============================================================
header_col, = st.columns([1])

with header_col:
    st.markdown(
        """
        <div class="title-container">
          <div class="pulse"></div>
          <div>
            <div style="font-size:1.45rem; font-weight:800; color:#0f172a;">Mass Casualty Incident Simulation</div>
            <div style="color:#64748b; font-size:0.9rem;">Earthquake Response Dashboard • NOAA Data Intelligence</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ============================================================
# 5) FILTERS (Control panel on main page)
# ============================================================
def classify_severity(total: int) -> str:
    if total >= 1000:
        return "Severe"
    if total >= 100:
        return "Moderate"
    return "Minor"

df["severity"] = df["total_casualties"].apply(classify_severity)

controls_col, display_col = st.columns([1, 3], gap="medium")

with controls_col:
    st.markdown("<p style='font-weight:800; margin:0 0 10px 0; color:#1e293b;'>CONTROL PANEL</p>", unsafe_allow_html=True)

    magnitude = st.slider("Target Magnitude", 3.0, 9.0, 6.5, 0.1)

    year_range = st.slider(
        "Year Span",
        int(df["year"].min()),
        int(df["year"].max()),
        (1900, int(df["year"].max()))
    )

    has_tsunami = "tsunami_flag" in df.columns
    event_type = st.selectbox(
        "Event Category",
        ["All earthquakes", "Earthquake only", "Earthquake + Tsunami"] if has_tsunami else ["All earthquakes", "Earthquake only"]
    )

    st.write("Severity Impact")
    options = ["Minor", "Moderate", "Severe"]

    # Initialize state once
    for opt in options:
        st.session_state.setdefault(opt, True)

    def enforce_one(selected_opt):
        # Prevent all being unchecked
        if sum(st.session_state[o] for o in options) == 0:
            st.session_state[selected_opt] = True

    for opt in options:
        st.checkbox(opt, key=opt, on_change=enforce_one, args=(opt,))

    severity_filter = [o for o in options if st.session_state[o]]

    only_with_casualties = st.toggle("Filter: Casualties > 0", value=True)

    metric = st.selectbox("Primary Metric", ["total_casualties", "deaths", "injuries"]) 

# ============================================================
# 6) FILTER LOGIC (Create f)
# ============================================================
mag_window = 0.25

f = df[
    (df["year"] >= year_range[0]) & (df["year"] <= year_range[1]) &
    (df["magnitude"].between(magnitude - mag_window, magnitude + mag_window)) &
    (df["severity"].isin(severity_filter))
].copy()

if only_with_casualties:
    f = f[f["total_casualties"] > 0]

if has_tsunami:
    if "only" in event_type.lower():
        f = f[f["tsunami_flag"] == 0]
    elif "tsunami" in event_type.lower():
        f = f[f["tsunami_flag"] == 1]

# ============================================================
# 7) MAIN DISPLAY AREA (Always render; if empty -> show zeros)
# ============================================================
with display_col:
    if f.empty:
        st.warning("No data matches these filters. Showing zeroed visuals.")
        f = pd.DataFrame({
            "year": [year_range[0]],
            "latitude": [0],
            "longitude": [0],
            "magnitude": [magnitude],
            "depth_km": [0],
            "deaths": [0],
            "injuries": [0],
            "total_casualties": [0],
            "severity": ["Minor"],
            "location_name": ["No data"]
        })
        if has_tsunami:
            f["tsunami_flag"] = 0

    selected_severities = set(severity_filter)

    if len(selected_severities) > 1:
        alert_class = "alert"
        severity_color = "#334155"
        sev_label = "MIXED"
    elif "Severe" in selected_severities:
        alert_class = "alert-severe"
        severity_color = "#b91c1c"
        sev_label = "SEVERE"
    elif "Moderate" in selected_severities:
        alert_class = "alert-moderate"
        severity_color = "#92400e"
        sev_label = "MODERATE"
    else:
        alert_class = "alert-low"
        severity_color = "#166534"
        sev_label = "LOW"

    peak = f.loc[f[metric].idxmax()]

    st.markdown(
        f"""
        <div class="alert {alert_class}">
          <div style="font-size:22px; line-height:1;">⚠️</div>
          <div>
            <div style="font-weight:800; font-size:1.05rem; color:{severity_color};">
              Dominant Threat Level: {sev_label}
            </div>
            <div style="font-size:0.9rem; opacity:0.9;">
              Peak Event: {int(peak[metric]):,} {metric.replace('_',' ')} in {peak.get('location_name','Unknown')} ({int(peak['year'])})
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # KPIs
    k1, k2, k3, k4 = st.columns(4)

    def render_kpi(col, title, val, sub, accent):
        with col:
            st.markdown(
                f"""
                <div class="card {accent}" style="padding:14px;">
                  <div class="kpi-title">{title}</div>
                  <div class="kpi-value">{val}</div>
                  <div class="kpi-sub">{sub}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    render_kpi(k1, "Total Casualties", f"{int(f['total_casualties'].sum()):,}", "Combined impact", "accent-red")
    render_kpi(k2, "Fatalities", f"{int(f['deaths'].sum()):,}", "Confirmed deaths", "accent-dark")
    render_kpi(k3, "Injuries", f"{int(f['injuries'].sum()):,}", "Medical cases", "accent-blue")
    render_kpi(k4, "Event Count", f"{len(f):,}", "Filtered records", "accent-blue")

    # Visuals Row 1
    v1, v2 = st.columns([1.5, 1], gap="medium")

    with v1:
        st.markdown('<div class="card"><strong>Geographic Impact</strong>', unsafe_allow_html=True)

        map_df = f.copy()
        map_df["bubble"] = map_df[metric].clip(lower=0) + 2

        fig_map = px.scatter_geo(
            map_df,
            lat="latitude",
            lon="longitude",
            size="bubble",
            color="magnitude",
            projection="natural earth",
            template="plotly_white",
            hover_name="location_name"
        )
        fig_map.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=440)
        st.plotly_chart(fig_map, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    with v2:
        st.markdown('<div class="card"><strong>Casualty Mix</strong>', unsafe_allow_html=True)

        dist = pd.DataFrame({"Cat": ["Deaths", "Injuries"], "Val": [f["deaths"].sum(), f["injuries"].sum()]})
        fig_pie = px.pie(
            dist,
            names="Cat",
            values="Val",
            hole=0.6,
            color_discrete_sequence=["#1e293b", "#ef4444"]
        )
        fig_pie.update_layout(margin=dict(l=18, r=18, t=18, b=18), height=420, showlegend=False)
        st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # Visuals Row 2
    v3, v4 = st.columns(2, gap="medium")

    with v3:
        st.markdown('<div class="card"><strong>Temporal Trend</strong>', unsafe_allow_html=True)

        trend = f.groupby("year")[metric].sum().reset_index()
        fig_line = px.area(
            trend,
            x="year",
            y=metric,
            template="plotly_white",
            color_discrete_sequence=["#3b82f6"]
        )
        fig_line.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=330)
        st.plotly_chart(fig_line, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    with v4:
        st.markdown('<div class="card"><strong>Magnitude Correlation</strong>', unsafe_allow_html=True)

        fig_scatter = px.scatter(
            f,
            x="magnitude",
            y=metric,
            color="severity",
            template="plotly_white",
            color_discrete_map={"Severe": "#ef4444", "Moderate": "#f59e0b", "Minor": "#10b981"}
        )
        fig_scatter.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=330)
        st.plotly_chart(fig_scatter, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# 8) DATA TABLE & EXPORT
# ============================================================
st.markdown('<div class="card"><strong>Filtered Data Preview</strong><p> limit to 25 event, download if you want to see more.</p></div>', unsafe_allow_html=True)

cols_to_show = [
    "date", "location_name", "magnitude", "depth_km",
    "deaths", "injuries", "total_casualties", "severity"
]
if "tsunami_flag" in f.columns:
    cols_to_show.append("tsunami_flag")

# --- Guard: if no records or missing date column ---
if f.empty or ("date" not in f.columns):
    st.info("No record found for the selected filters.")
else:
    # Only select columns that actually exist (extra safety)
    safe_cols = [c for c in cols_to_show if c in f.columns]

    table_df = f[safe_cols].sort_values("date", ascending=False).head(25)
    st.dataframe(table_df, use_container_width=True, hide_index=True)

    st.download_button(
        "Export Current Filtered Results",
        data=f.to_csv(index=False).encode("utf-8"),
        file_name="mci_export.csv",
        mime="text/csv",
        use_container_width=True
    )

# ============================================================
# 9) FOOTER
# ============================================================
st.markdown(
    """
    <div class="card">
      <div style="display:grid; grid-template-columns: 1.2fr 1fr; gap: 16px;">
        <div>
          <div style="font-weight:900;">About</div>
          <div style="color:#64748b; font-size:0.9rem; margin-top:6px;">
            Single-page dashboard prototype for Health Informatics (ITE3). It supports drill-down analysis of earthquake events
            by time, magnitude, depth, and geographic location to highlight potential mass-casualty patterns.
            Severity levels are simulated from total casualties (Minor/Moderate/Severe).
          </div>
        </div>
        <div>
          <div style="font-weight:900;">Data Source</div>
          <div style="color:#64748b; font-size:0.9rem; margin-top:6px;">
            Kaggle — Major Earthquakes (NOAA) by shekpaul<br/>
            <a href="https://www.kaggle.com/datasets/shekpaul/major-earthquakes-noaa" target="_blank">
                https://www.kaggle.com/datasets/shekpaul/major-earthquakes-noaa
            </a><br/><br/>
            <b>Original Provider</b><br/>
            NOAA NCEI — Global Significant Earthquake Database (DOI: 10.7289/V5TD9V7K)
          </div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True
)

