import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

DATA_PATH = "data/processed/processed_data.csv"
OUTPUT_PATH = "outputs/customer_insights.csv"
CHURN_MODEL_PATH = "models/churn_model.pkl"

RANDOM_STATE = 42


def assign_persona(row):
    if row["recency"] < 30 and row["frequency"] > 5 and row["monetary"] > 2000:
        return "Champions"
    if row["recency"] > 180 and row["frequency"] <= 2:
        return "Lost / Hibernating"
    if row["recency"] > 90 and row["frequency"] > 5:
        return "At Risk Loyal"
    if row["recency"] < 60 and row["frequency"] <= 2:
        return "Recent Customers"
    return "Average Customers"


def risk_segment_from_probability(probability):
    if pd.isna(probability):
        return "Unknown"
    if probability >= 0.70:
        return "High Risk"
    if probability >= 0.40:
        return "Medium Risk"
    return "Low Risk"


def risk_segment_from_churn(churn):
    return "High Risk" if int(churn) == 1 else "Low Risk"


def calculate_customer_features(df):
    current_date = df["invoicedate"].max() + pd.Timedelta(days=1)

    customer_base = (
        df.groupby("customer_id")
        .agg(
            recency=("invoicedate", lambda x: (current_date - x.max()).days),
            frequency=("invoice", "nunique"),
            monetary=("total_amount", "sum"),
            churn=("churn", "max"),
            total_quantity=("quantity", "sum"),
            unique_products=("stockcode", "nunique"),
            first_purchase_date=("invoicedate", "min"),
            last_purchase_date=("invoicedate", "max"),
            country=("country", lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown"),
        )
        .reset_index()
    )

    customer_base = customer_base[customer_base["monetary"] > 0].copy()

    customer_base["avg_order_value"] = (
        customer_base["monetary"] / customer_base["frequency"]
    )

    customer_base["tenure"] = (
        customer_base["last_purchase_date"] - customer_base["first_purchase_date"]
    ).dt.days

    customer_base["avg_days_between_purchases"] = (
        customer_base["tenure"] / customer_base["frequency"]
    )

    customer_base["avg_days_between_purchases"] = (
        customer_base["avg_days_between_purchases"]
        .replace([np.inf, -np.inf], 0)
        .fillna(0)
    )

    return customer_base


def add_customer_segments(customer_df):
    rfm_cols = ["recency", "frequency", "monetary"]

    rfm_log = customer_df[rfm_cols].copy()
    rfm_log["recency"] = np.log1p(rfm_log["recency"])
    rfm_log["frequency"] = np.log1p(rfm_log["frequency"])
    rfm_log["monetary"] = np.log1p(rfm_log["monetary"])

    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_log)

    n_customers = len(customer_df)

    if n_customers < 10:
        customer_df["kmeans_cluster"] = 0
        customer_df["segment"] = "Average Customers"
        customer_df["dbscan_cluster"] = 0
        customer_df["gmm_cluster"] = 0
        customer_df["gmm_confidence"] = 1.0
        return customer_df

    n_clusters = min(6, n_customers)

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=RANDOM_STATE,
        n_init=10
    )

    customer_df["kmeans_cluster"] = kmeans.fit_predict(rfm_scaled)

    cluster_summary = (
        customer_df.groupby("kmeans_cluster")
        .agg(
            recency=("recency", "mean"),
            frequency=("frequency", "mean"),
            monetary=("monetary", "mean"),
            customer_count=("customer_id", "count"),
        )
        .reset_index()
    )

    cluster_summary["segment"] = cluster_summary.apply(assign_persona, axis=1)
    segment_map = cluster_summary.set_index("kmeans_cluster")["segment"].to_dict()

    customer_df["segment"] = customer_df["kmeans_cluster"].map(segment_map)

    dbscan = DBSCAN(eps=0.5, min_samples=10)
    customer_df["dbscan_cluster"] = dbscan.fit_predict(rfm_scaled)

    gmm = GaussianMixture(
        n_components=n_clusters,
        random_state=RANDOM_STATE
    )

    gmm.fit(rfm_scaled)
    customer_df["gmm_cluster"] = gmm.predict(rfm_scaled)
    customer_df["gmm_confidence"] = gmm.predict_proba(rfm_scaled).max(axis=1)

    return customer_df


def add_churn_predictions(customer_df):
    customer_df["churn_probability"] = np.nan
    customer_df["churn_prediction"] = customer_df["churn"]

    if not os.path.exists(CHURN_MODEL_PATH):
        customer_df["risk_segment"] = customer_df["churn"].apply(risk_segment_from_churn)
        return customer_df

    try:
        model = joblib.load(CHURN_MODEL_PATH)

        feature_cols = [
            "frequency",
            "monetary",
            "avg_order_value",
            "total_quantity",
            "unique_products",
            "tenure",
            "avg_days_between_purchases",
        ]

        X = customer_df[feature_cols].copy()

        if hasattr(model, "predict_proba"):
            customer_df["churn_probability"] = model.predict_proba(X)[:, 1]
            customer_df["churn_prediction"] = (
                customer_df["churn_probability"] >= 0.5
            ).astype(int)
        else:
            customer_df["churn_prediction"] = model.predict(X)
            customer_df["churn_probability"] = customer_df["churn_prediction"]

        customer_df["risk_segment"] = customer_df["churn_probability"].apply(
            risk_segment_from_probability
        )

    except Exception as error:
        print(f"Could not use saved churn model: {error}")
        customer_df["risk_segment"] = customer_df["churn"].apply(risk_segment_from_churn)

    return customer_df


def main():
    print("Loading processed dataset...")
    df = pd.read_csv(DATA_PATH, parse_dates=["invoicedate"])

    df = df[(df["quantity"] > 0) & (df["price"] > 0)].copy()

    print("Calculating customer features...")
    customer_df = calculate_customer_features(df)

    print("Adding customer segmentation...")
    customer_df = add_customer_segments(customer_df)

    print("Adding churn predictions/status...")
    customer_df = add_churn_predictions(customer_df)

    customer_df["first_purchase_date"] = customer_df["first_purchase_date"].dt.strftime("%Y-%m-%d")
    customer_df["last_purchase_date"] = customer_df["last_purchase_date"].dt.strftime("%Y-%m-%d")

    final_columns = [
        "customer_id",
        "recency",
        "frequency",
        "monetary",
        "avg_order_value",
        "total_quantity",
        "unique_products",
        "tenure",
        "avg_days_between_purchases",
        "segment",
        "kmeans_cluster",
        "dbscan_cluster",
        "gmm_cluster",
        "gmm_confidence",
        "churn",
        "churn_probability",
        "churn_prediction",
        "risk_segment",
        "country",
        "first_purchase_date",
        "last_purchase_date",
    ]

    customer_insights = customer_df[final_columns].copy()

    numeric_cols = customer_insights.select_dtypes(include=[np.number]).columns
    customer_insights[numeric_cols] = customer_insights[numeric_cols].round(4)

    os.makedirs("outputs", exist_ok=True)
    customer_insights.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved: {OUTPUT_PATH}")
    print(f"Rows: {len(customer_insights)}")
    print(customer_insights.head())


if __name__ == "__main__":
    main()