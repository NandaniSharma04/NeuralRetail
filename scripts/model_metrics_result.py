import os
from datetime import datetime

import numpy as np
import pandas as pd

FORECAST_PATH = "outputs/Demand_forecast_output.csv"
PRODUCT_PATH = "outputs/product_intelligence.csv"
CUSTOMER_PATH = "outputs/customer_insights.csv"
OUTPUT_PATH = "outputs/model_metrics.csv"


def add_metric(rows, model_name, metric_name, metric_value, status, notes):
    rows.append({
        "model_name": model_name,
        "metric_name": metric_name,
        "metric_value": metric_value,
        "last_updated": datetime.now().strftime("%Y-%m-%d"),
        "status": status,
        "notes": notes,
    })


def main():
    rows = []

    if os.path.exists(FORECAST_PATH):
        forecast_df = pd.read_csv(FORECAST_PATH)

        if "forecast_type" in forecast_df.columns:
            demand_df = forecast_df[forecast_df["forecast_type"] == "demand"]
            revenue_df = forecast_df[forecast_df["forecast_type"] == "revenue"]
        else:
            demand_df = forecast_df
            revenue_df = pd.DataFrame()

        if "prophet_mape" in forecast_df.columns:
            add_metric(
                rows,
                "Prophet Forecast",
                "MAPE",
                round(forecast_df["prophet_mape"].dropna().mean(), 2),
                "Active",
                "Average Prophet MAPE from forecast output"
            )

        if "lstm_mape" in forecast_df.columns:
            lstm_values = pd.to_numeric(forecast_df["lstm_mape"], errors="coerce").dropna()
            add_metric(
                rows,
                "LSTM Forecast",
                "MAPE",
                round(lstm_values.mean(), 2) if len(lstm_values) else np.nan,
                "Active" if len(lstm_values) else "Not Available",
                "Average LSTM MAPE where LSTM was trained"
            )

        if "ensemble_mape" in forecast_df.columns:
            add_metric(
                rows,
                "Prophet + LSTM Ensemble",
                "MAPE",
                round(forecast_df["ensemble_mape"].dropna().mean(), 2),
                "Active",
                "Average ensemble MAPE from forecast output"
            )

        add_metric(
            rows,
            "Demand Forecast Output",
            "Rows",
            len(demand_df),
            "Ready",
            "Rows available for Demand Intelligence dashboard"
        )

        add_metric(
            rows,
            "Revenue Forecast Output",
            "Rows",
            len(revenue_df),
            "Ready" if len(revenue_df) else "Not Available",
            "Rows available for revenue forecast dashboard"
        )

    else:
        add_metric(
            rows,
            "Forecast Models",
            "Output Status",
            np.nan,
            "Missing",
            "forecast_results.csv not found"
        )

    if os.path.exists(PRODUCT_PATH):
        product_df = pd.read_csv(PRODUCT_PATH)

        add_metric(
            rows,
            "Price Elasticity",
            "Products Covered",
            len(product_df),
            "Ready",
            "Product-level elasticity results available"
        )

        if "elasticity" in product_df.columns:
            elasticity_values = pd.to_numeric(product_df["elasticity"], errors="coerce").dropna()
            add_metric(
                rows,
                "Price Elasticity",
                "Average Elasticity",
                round(elasticity_values.mean(), 4) if len(elasticity_values) else np.nan,
                "Ready" if len(elasticity_values) else "Limited",
                "Average observed product elasticity"
            )

        if "eoq" in product_df.columns:
            add_metric(
                rows,
                "Inventory EOQ",
                "Products Covered",
                len(product_df),
                "Ready",
                "EOQ and reorder point results available"
            )

    else:
        add_metric(
            rows,
            "Product Intelligence",
            "Output Status",
            np.nan,
            "Missing",
            "product_intelligence.csv not found"
        )

    if os.path.exists(CUSTOMER_PATH):
        customer_df = pd.read_csv(CUSTOMER_PATH)

        add_metric(
            rows,
            "Customer Segmentation",
            "Customers Covered",
            len(customer_df),
            "Ready",
            "Customer RFM segmentation results available"
        )

        if "gmm_confidence" in customer_df.columns:
            add_metric(
                rows,
                "Customer Segmentation",
                "Average GMM Confidence",
                round(customer_df["gmm_confidence"].dropna().mean(), 4),
                "Ready",
                "Average confidence from GMM segmentation"
            )

        if "churn" in customer_df.columns:
            churn_rate = customer_df["churn"].mean() * 100
            add_metric(
                rows,
                "Churn Model",
                "Observed Churn Rate",
                round(churn_rate, 2),
                "Ready",
                "Churn rate from processed dataset"
            )

        if "churn_probability" in customer_df.columns:
            churn_prob = pd.to_numeric(customer_df["churn_probability"], errors="coerce").dropna()
            add_metric(
                rows,
                "Churn Model",
                "Avg Churn Probability",
                round(churn_prob.mean(), 4) if len(churn_prob) else np.nan,
                "Active" if len(churn_prob) else "Fallback",
                "Uses saved churn model if compatible, otherwise dataset churn labels"
            )

    else:
        add_metric(
            rows,
            "Customer Insights",
            "Output Status",
            np.nan,
            "Missing",
            "customer_insights.csv not found"
        )

    model_metrics = pd.DataFrame(rows)

    os.makedirs("outputs", exist_ok=True)
    model_metrics.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved: {OUTPUT_PATH}")
    print(model_metrics)


if __name__ == "__main__":
    main()