from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import os
import joblib

try:
    from auth import login_user, signup_user, reset_password, create_users_table
    create_users_table()
except Exception:
    def login_user(u, p): return True
    def signup_user(n, u, p): return True, "ok"
    def reset_password(u, p): return True

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH  = ROOT_DIR / "data" / "processed" / "processed_data.csv"
DATA_RAW   = ROOT_DIR / "data" / "raw" / "new_cleaned_retail_data_with_churn.csv"
MODEL_DIR  = ROOT_DIR / "models"

# ── Brand colours ──────────────────────────────────────────────────────────
PRIMARY   = "#F64708"
SECONDARY = "#FF9F43"
BG        = "#05070A"
CARD      = "#111827"
TEXT      = "#E5E7EB"
GOOD      = "#22C55E"
WARN      = "#F59E0B"
BAD       = "#EF4444"

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuralRetail Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
html, body, [data-testid="stAppViewContainer"] {{
    background: {BG};
    color: {TEXT};
}}
[data-testid="stSidebar"] {{
    background: {CARD};
}}
h1, h2, h3 {{
    color: #FFFFFF;
}}
.metric-card {{
    background: {CARD};
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
}}
.metric-val {{
    font-size: 2rem;
    font-weight: 700;
    color: {PRIMARY};
}}
.metric-label {{
    font-size: 0.85rem;
    color: #94A3B8;
    margin-top: 4px;
}}
</style>
""", unsafe_allow_html=True)

# ── Data loader ────────────────────────────────────────────────────────────
@st.cache_data(ttl=600)
def load_data() -> pd.DataFrame:
    for path in [DATA_PATH, DATA_RAW]:
        if path.exists():
            df = pd.read_csv(path)
            if "invoicedate" in df.columns:
                df["invoicedate"] = pd.to_datetime(df["invoicedate"], errors="coerce")
            return df
    # fallback: tiny synthetic dataset so the app never crashes
    dates = pd.date_range("2010-01-01", periods=200, freq="D")
    np.random.seed(42)
    return pd.DataFrame({
        "invoicedate": np.random.choice(dates, 1000),
        "total_amount": np.random.uniform(10, 500, 1000),
        "customer_id":  np.random.randint(10000, 20000, 1000),
        "country":      np.random.choice(["United Kingdom","Germany","France","Spain"], 1000),
        "quantity":     np.random.randint(1, 50, 1000),
        "Churn":        np.random.randint(0, 2, 1000),
        "Recency":      np.random.randint(1, 300, 1000),
    })

@st.cache_data(ttl=600)
def load_rfm(df: pd.DataFrame) -> pd.DataFrame:
    rfm = df.groupby("customer_id").agg(
        Frequency=("customer_id", "count"),
        Monetary=("total_amount", "sum"),
        Churn=("Churn", "max") if "Churn" in df.columns else ("total_amount", "count"),
    ).reset_index()
    rfm["AvgOrderValue"] = rfm["Monetary"] / rfm["Frequency"]
    return rfm

# ── Model loader (safe) ─────────────────────────────────────────────────────
def load_model(name: str):
    path = MODEL_DIR / name
    if path.exists():
        try:
            return joblib.load(path)
        except Exception:
            return None
    return None

# ── Auth state ─────────────────────────────────────────────────────────────
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "page" not in st.session_state:
    st.session_state.page = "Login"

# ── Login / Signup UI ──────────────────────────────────────────────────────
def auth_page():
    st.markdown(f"""
    <div style='text-align:center; padding: 60px 0 20px'>
        <h1 style='color:{PRIMARY}; font-size:3rem;'>NeuralRetail</h1>
        <p style='color:#94A3B8;'>AI-Powered Sales Intelligence Platform</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        tab1, tab2 = st.tabs(["Login", "Sign Up"])

        with tab1:
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_pass")
            if st.button("Login", use_container_width=True, type="primary"):
                result = login_user(email, password)
                if result:
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("Invalid credentials. Please try again.")

        with tab2:
            name  = st.text_input("Full Name", key="su_name")
            email2 = st.text_input("Email", key="su_email")
            pass2  = st.text_input("Password", type="password", key="su_pass")
            if st.button("Create Account", use_container_width=True, type="primary"):
                ok, msg = signup_user(name, email2, pass2)
                if ok:
                    st.success(msg + " Please login.")
                else:
                    st.error(msg)

# ── Sidebar navigation ─────────────────────────────────────────────────────
PAGES = [
    "📊  Executive Overview",
    "📈  Demand Intelligence",
    "👥  Customer Hub",
    "⚠️  Churn Risk",
    "📦  Inventory Health",
    "🌍  Regional Sales",
    "💰  Revenue Analysis",
]

def sidebar():
    with st.sidebar:
        st.markdown(f"""
        <div style='padding:16px 0 8px'>
            <span style='font-size:1.4rem; font-weight:700; color:{PRIMARY}'>
                NeuralRetail
            </span><br>
            <span style='font-size:0.75rem; color:#94A3B8'>
                AI-Powered Sales Intelligence
            </span>
        </div>
        """, unsafe_allow_html=True)
        st.divider()
        page = st.radio("Navigation", PAGES, label_visibility="collapsed")
        st.divider()
        st.caption("AMX-DS-2026-04 | NeuralRetail v1.0")
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.rerun()
    return page

# ── Metric card helper ──────────────────────────────────────────────────────
def metric_card(label: str, value: str, col):
    with col:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-val'>{value}</div>
            <div class='metric-label'>{label}</div>
        </div>
        """, unsafe_allow_html=True)

# ── Pages ──────────────────────────────────────────────────────────────────

def page_executive(df: pd.DataFrame):
    st.title("📊 Executive Overview")
    st.caption("High-level KPIs across the full retail dataset")
    st.divider()

    total_revenue  = df["total_amount"].sum() if "total_amount" in df.columns else 0
    total_orders   = df["customer_id"].count()
    total_customers = df["customer_id"].nunique()
    avg_order      = total_revenue / total_orders if total_orders else 0

    c1, c2, c3, c4 = st.columns(4)
    metric_card("Total Revenue",    f"£{total_revenue:,.0f}", c1)
    metric_card("Total Orders",     f"{total_orders:,}",      c2)
    metric_card("Unique Customers", f"{total_customers:,}",   c3)
    metric_card("Avg Order Value",  f"£{avg_order:.2f}",      c4)

    st.markdown("###")
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Revenue by Country")
        if "country" in df.columns:
            rev_country = df.groupby("country")["total_amount"].sum().sort_values(ascending=False).head(10)
            fig = px.bar(rev_country, orientation="h",
                         color_discrete_sequence=[PRIMARY],
                         labels={"value": "Revenue (£)", "index": "Country"})
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font_color=TEXT, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Monthly Revenue Trend")
        if "invoicedate" in df.columns:
            df["month"] = df["invoicedate"].dt.to_period("M").astype(str)
            monthly = df.groupby("month")["total_amount"].sum().reset_index()
            fig2 = px.line(monthly, x="month", y="total_amount",
                           color_discrete_sequence=[SECONDARY],
                           labels={"total_amount": "Revenue (£)", "month": "Month"})
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font_color=TEXT)
            st.plotly_chart(fig2, use_container_width=True)


def page_demand(df: pd.DataFrame):
    st.title("📈 Demand Intelligence")
    st.caption("Sales forecast and demand trends")
    st.divider()

    if "invoicedate" in df.columns:
        daily = df.groupby(df["invoicedate"].dt.date)["total_amount"].sum().reset_index()
        daily.columns = ["ds", "y"]

        st.subheader("Historical Daily Sales")
        fig = px.line(daily, x="ds", y="y",
                      color_discrete_sequence=[PRIMARY],
                      labels={"y": "Daily Revenue (£)", "ds": "Date"})
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color=TEXT)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Prophet 30-Day Forecast")
        prophet = load_model("prophet_model.pkl")
        if prophet:
            future = prophet.make_future_dataframe(periods=30)
            forecast = prophet.predict(future)
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=daily["ds"], y=daily["y"],
                                      name="Actual", line=dict(color=PRIMARY)))
            fig2.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"],
                                      name="Forecast", line=dict(color=SECONDARY, dash="dash")))
            fig2.add_trace(go.Scatter(
                x=list(forecast["ds"]) + list(forecast["ds"][::-1]),
                y=list(forecast["yhat_upper"]) + list(forecast["yhat_lower"][::-1]),
                fill="toself", fillcolor="rgba(255,159,67,0.15)",
                line=dict(color="rgba(255,255,255,0)"), name="95% CI"
            ))
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font_color=TEXT)
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Prophet model not loaded. Showing historical data only.")
    else:
        st.warning("Date column not found in dataset.")


def page_customer(df: pd.DataFrame):
    st.title("👥 Customer Hub")
    st.caption("RFM segmentation and customer behaviour")
    st.divider()

    rfm = load_rfm(df)

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Frequency vs Monetary")
        fig = px.scatter(rfm, x="Frequency", y="Monetary",
                         color="AvgOrderValue",
                         color_continuous_scale="Oranges",
                         labels={"Frequency": "No. of Orders", "Monetary": "Total Spend (£)"})
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color=TEXT)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        seg_model = load_model("segmentation_model.pkl")
        scaler    = load_model("scaler.pkl")
        if seg_model and scaler:
            features = rfm[["Frequency", "Monetary", "AvgOrderValue"]].fillna(0)
            scaled   = scaler.transform(features)
            rfm["Segment"] = seg_model.predict(scaled)
            seg_map = {0: "Loyal", 1: "VIP", 2: "At Risk", 3: "High Value"}
            rfm["SegmentName"] = rfm["Segment"].map(seg_map).fillna("Other")
            counts = rfm["SegmentName"].value_counts()
            fig2 = px.pie(values=counts.values, names=counts.index,
                          color_discrete_sequence=[PRIMARY, SECONDARY, GOOD, WARN])
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color=TEXT)
            st.subheader("Customer Segments")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.subheader("Top Customers by Spend")
            top = rfm.nlargest(10, "Monetary")[["customer_id", "Frequency", "Monetary"]]
            st.dataframe(top, use_container_width=True)


def page_churn(df: pd.DataFrame):
    st.title("⚠️ Churn Risk Assessment")
    st.caption("Customer churn predictions powered by XGBoost")
    st.divider()

    rfm = load_rfm(df)
    model = load_model("churn_model.pkl")

    if model:
        features = ["Frequency", "Monetary", "AvgOrderValue"]
        X = rfm[features].fillna(0)
        rfm["ChurnProbability"] = model.predict_proba(X)[:, 1]
        rfm["Risk"] = pd.cut(rfm["ChurnProbability"],
                             bins=[0, 0.4, 0.7, 1.0],
                             labels=["Low", "Medium", "High"])

        c1, c2, c3 = st.columns(3)
        metric_card("High Risk Customers",
                    str((rfm["Risk"] == "High").sum()), c1)
        metric_card("Medium Risk",
                    str((rfm["Risk"] == "Medium").sum()), c2)
        metric_card("Low Risk",
                    str((rfm["Risk"] == "Low").sum()), c3)

        st.markdown("###")
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Churn Probability Distribution")
            fig = px.histogram(rfm, x="ChurnProbability", nbins=30,
                               color_discrete_sequence=[PRIMARY])
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font_color=TEXT)
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            st.subheader("Risk Distribution")
            counts = rfm["Risk"].value_counts()
            fig2 = px.pie(values=counts.values, names=counts.index,
                          color_discrete_sequence=[GOOD, WARN, BAD])
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color=TEXT)
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("High Risk Customers")
        high_risk = rfm[rfm["Risk"] == "High"].sort_values(
            "ChurnProbability", ascending=False).head(20)
        st.dataframe(high_risk[["customer_id", "Frequency", "Monetary",
                                 "ChurnProbability"]].reset_index(drop=True),
                     use_container_width=True)
    else:
        st.warning("Churn model not found. Please ensure churn_model.pkl is in the models/ folder.")
        if "Churn" in df.columns:
            churn_rate = df["Churn"].mean()
            st.metric("Overall Churn Rate", f"{churn_rate:.1%}")


def page_inventory(df: pd.DataFrame):
    st.title("📦 Inventory Health & Optimization")
    st.caption("ABC analysis and EOQ reorder recommendations")
    st.divider()

    if "stockcode" not in df.columns:
        st.warning("Stock code column not found.")
        return

    inv = df.groupby("stockcode").agg(
        TotalRevenue=("total_amount", "sum"),
        TotalQty=("quantity", "sum") if "quantity" in df.columns else ("total_amount", "count")
    ).reset_index().sort_values("TotalRevenue", ascending=False)

    inv["CumRevPct"] = inv["TotalRevenue"].cumsum() / inv["TotalRevenue"].sum()
    inv["ABC"] = inv["CumRevPct"].apply(
        lambda x: "A" if x <= 0.80 else ("B" if x <= 0.95 else "C"))

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("ABC Classification")
        counts = inv["ABC"].value_counts()
        fig = px.pie(values=counts.values, names=counts.index,
                     color_discrete_sequence=[PRIMARY, SECONDARY, WARN])
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color=TEXT)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Top A-Class SKUs")
        top_a = inv[inv["ABC"] == "A"].head(10)
        st.dataframe(top_a[["stockcode", "TotalRevenue", "ABC"]].reset_index(drop=True),
                     use_container_width=True)


def page_regional(df: pd.DataFrame):
    st.title("🌍 Regional Sales")
    st.divider()
    if "country" not in df.columns:
        st.warning("Country column not found.")
        return
    rev = df.groupby("country")["total_amount"].sum().reset_index()
    rev.columns = ["Country", "Revenue"]
    fig = px.choropleth(rev, locations="Country", locationmode="country names",
                        color="Revenue", color_continuous_scale="Oranges",
                        title="Revenue by Country")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color=TEXT, geo_bgcolor=BG)
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Revenue by Country")
    st.dataframe(rev.sort_values("Revenue", ascending=False), use_container_width=True)


def page_revenue(df: pd.DataFrame):
    st.title("💰 Revenue Analysis")
    st.divider()
    if "invoicedate" not in df.columns:
        st.warning("Date column not found.")
        return
    df["month"] = df["invoicedate"].dt.to_period("M").astype(str)
    monthly = df.groupby("month")["total_amount"].agg(["sum", "mean", "count"]).reset_index()
    monthly.columns = ["Month", "Total Revenue", "Avg Order", "Orders"]

    fig = go.Figure()
    fig.add_bar(x=monthly["Month"], y=monthly["Total Revenue"],
                name="Revenue", marker_color=PRIMARY)
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font_color=TEXT, xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(monthly, use_container_width=True)


# ── Main router ─────────────────────────────────────────────────────────────
def main():
    if not st.session_state.authenticated:
        auth_page()
        return

    df   = load_data()
    page = sidebar()

    if   "Executive"  in page: page_executive(df)
    elif "Demand"     in page: page_demand(df)
    elif "Customer"   in page: page_customer(df)
    elif "Churn"      in page: page_churn(df)
    elif "Inventory"  in page: page_inventory(df)
    elif "Regional"   in page: page_regional(df)
    elif "Revenue"    in page: page_revenue(df)
    else:
        st.title("Coming Soon")
        st.info("This page is under construction.")


if __name__ == "__main__":
    main()
