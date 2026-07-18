from __future__ import annotations
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import os
from app.auth import *
from app.auth import login_user, signup_user, reset_password
# --------------------------------------------------------------------------------------
# Paths & constants
# --------------------------------------------------------------------------------------


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "processed" / "processed_data.csv"
FORECAST_PATH = ROOT_DIR / "outputs" / "Demand_forecast_output.csv"
PRODUCT_PATH = ROOT_DIR / "outputs" / "product_intelligence.csv"
CUSTOMER_PATH = ROOT_DIR / "outputs" / "customer_insights.csv"
METRICS_PATH = ROOT_DIR / "outputs" / "model_metrics.csv"

API_URL = os.getenv("API_URL", "http://localhost:8000")
CACHE_TTL_DATA = 600
CACHE_TTL_HEALTH = 15

TABLER_ICONS_CSS = "https://cdn.jsdelivr.net/npm/@tabler/icons-webfont@2.47.0/tabler-icons.min.css"

BRAND = {

    # Brand
    "primary": "#F64708",
    "secondary": "#FF9F43",

    # Text
    "text": "#E5E7EB",
    "muted": "#94A3B8",
    "heading": "#FFFFFF",

    # Background
    "bg": "#05070A",
    "card": "#111827",
    "card_hover": "#1B263E",
    "border": "rgba(255,255,255,0.10)",

    # Status
    "good": "#22C55E",
    "warn": "#F59E0B",
    "bad": "#EF4444",

    # Charts
    "blue": "#3B82F6",
    "amber": "#F59E0B",
    "purple": "#A855F7",
    "green": "#22C55E",
}
HEAT_SCALE = ["#241A1A", "#4A2A26", "#7A3B2E", "#B84D34", "#F5643A"]
MAPE_SCALE = ["#3DD68C", "#F5B942", "#FF6B6B"]

PAGES = [
    ("Executive Overview", "📊"),
    ("Demand Intelligence", "📈"),
    ("Customer Hub", "👥"),
    ("Inventory Health", "📦"),
    ("MLOps Monitor", "🛠️"),
    ("Regional Sales", "🌍"),
    ("Revenue Analysis", "💰"),
    ("Order & Transaction", "🧾"),
    ("Sales Trend Analysis", "📉"),
]

PAGE_NAMES = [f"{icon}  {name}" for name, icon in PAGES]

st.set_page_config(
    page_title="NeuralRetail Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


#--------
#login page
#------

def login_theme():

    st.markdown(
        f"""
<style>

html,body,[data-testid="stAppViewContainer"]{{
    background:{BRAND["bg"]};
}}

header[data-testid="stHeader"]{{
    display:none;
}}

#MainMenu,footer{{
    visibility:hidden;
}}

.block-container{{
    padding-top:0rem;
"""
    )

# (rest of file unchanged)
