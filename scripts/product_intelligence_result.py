import os
import numpy as np
import pandas as pd

DATA_PATH = "data/processed/processed_data.csv"
OUTPUT_PATH = "outputs/product_intelligence.csv"

TOP_N_PRODUCTS = 100
SERVICE_LEVEL_Z = 1.65
HOLDING_RATE = 0.20
DEFAULT_ORDER_COST_MIN = 10
DEFAULT_ORDER_COST_MAX = 50


def classify_elasticity(elasticity):
    if pd.isna(elasticity):
        return "Unknown"
    if elasticity < -1:
        return "Elastic"
    if -1 <= elasticity < 0:
        return "Inelastic"
    return "Positive/Unusual"


def revenue_impact_label(elasticity):
    if pd.isna(elasticity):
        return "Unknown"
    if elasticity <= -1.5:
        return "High"
    if elasticity <= -0.8:
        return "Medium"
    return "Low"


def recommendation_label(elasticity):
    if pd.isna(elasticity):
        return "Not enough price variation"
    if elasticity <= -1.5:
        return "Avoid price increase"
    if elasticity <= -0.8:
        return "Test small price changes"
    if elasticity < 0:
        return "Can test price increase"
    return "Review pricing data"


def inventory_risk_label(row):
    if row["reorder_point"] > row["avg_daily_demand"] * 14:
        return "High"
    if row["reorder_point"] > row["avg_daily_demand"] * 7:
        return "Medium"
    return "Low"


def calculate_arc_elasticity(product_weekly):
    product_weekly = product_weekly.sort_values("week").copy()

    product_weekly["price_pct_change"] = product_weekly["avg_price"].pct_change()
    product_weekly["quantity_pct_change"] = product_weekly["quantity"].pct_change()

    valid = product_weekly[
        (product_weekly["price_pct_change"].abs() > 0.01)
        & (product_weekly["quantity_pct_change"].notna())
        & (product_weekly["price_pct_change"].notna())
    ].copy()

    if len(valid) < 2:
        return np.nan

    elasticity_values = valid["quantity_pct_change"] / valid["price_pct_change"]

    elasticity_values = elasticity_values.replace([np.inf, -np.inf], np.nan).dropna()

    if len(elasticity_values) == 0:
        return np.nan

    elasticity_values = elasticity_values.clip(lower=-10, upper=10)

    return float(elasticity_values.median())


def main():
    print("Loading processed data...")
    df = pd.read_csv(DATA_PATH, parse_dates=["invoicedate"])

    df = df[(df["quantity"] > 0) & (df["price"] > 0)].copy()

    df["date"] = df["invoicedate"].dt.date
    df["week"] = df["invoicedate"].dt.to_period("W").dt.start_time

    print("Selecting top products...")
    top_products = (
        df.groupby("stockcode")["total_amount"]
        .sum()
        .sort_values(ascending=False)
        .head(TOP_N_PRODUCTS)
        .index
    )

    df_top = df[df["stockcode"].isin(top_products)].copy()

    print("Creating product sales summary...")
    product_summary = (
        df_top.groupby("stockcode")
        .agg(
            description=("description", lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown"),
            avg_price=("price", "mean"),
            min_price=("price", "min"),
            max_price=("price", "max"),
            total_quantity=("quantity", "sum"),
            total_revenue=("total_amount", "sum"),
            transaction_count=("invoice", "nunique"),
            first_sale_date=("invoicedate", "min"),
            last_sale_date=("invoicedate", "max"),
        )
        .reset_index()
    )

    product_summary["price_range"] = product_summary["max_price"] - product_summary["min_price"]

    print("Calculating price elasticity without training...")
    weekly = (
        df_top.groupby(["stockcode", "week"])
        .agg(
            quantity=("quantity", "sum"),
            avg_price=("price", "mean"),
        )
        .reset_index()
    )

    elasticity_rows = []

    for stockcode, group in weekly.groupby("stockcode"):
        elasticity = calculate_arc_elasticity(group)
        elasticity_rows.append({
            "stockcode": stockcode,
            "elasticity": elasticity,
        })

    elasticity_df = pd.DataFrame(elasticity_rows)

    product_summary = product_summary.merge(elasticity_df, on="stockcode", how="left")

    product_summary["elasticity"] = product_summary["elasticity"].round(4)
    product_summary["elasticity_interpretation"] = product_summary["elasticity"].apply(classify_elasticity)
    product_summary["revenue_impact"] = product_summary["elasticity"].apply(revenue_impact_label)
    product_summary["recommendation"] = product_summary["elasticity"].apply(recommendation_label)

    print("Calculating inventory EOQ metrics...")
    daily_demand = (
        df_top.groupby(["stockcode", "date"])["quantity"]
        .sum()
        .reset_index()
    )

    demand_stats = (
        daily_demand.groupby("stockcode")["quantity"]
        .agg(["mean", "std"])
        .reset_index()
    )

    demand_stats.columns = ["stockcode", "avg_daily_demand", "demand_std"]
    demand_stats["demand_std"] = demand_stats["demand_std"].fillna(0)

    demand_stats["annual_demand"] = demand_stats["avg_daily_demand"] * 365

    np.random.seed(42)
    demand_stats["lead_time_days"] = np.random.randint(3, 14, len(demand_stats))
    demand_stats["ordering_cost"] = np.random.uniform(
        DEFAULT_ORDER_COST_MIN,
        DEFAULT_ORDER_COST_MAX,
        len(demand_stats)
    )

    product_summary = product_summary.merge(demand_stats, on="stockcode", how="left")

    product_summary["holding_cost"] = (product_summary["avg_price"] * HOLDING_RATE).clip(lower=0.5)

    product_summary["eoq"] = np.sqrt(
        (2 * product_summary["annual_demand"] * product_summary["ordering_cost"])
        / product_summary["holding_cost"]
    )

    product_summary["eoq"] = np.ceil(product_summary["eoq"])

    product_summary["safety_stock"] = np.ceil(
        SERVICE_LEVEL_Z
        * product_summary["demand_std"]
        * np.sqrt(product_summary["lead_time_days"])
    )

    product_summary["reorder_point"] = np.ceil(
        (product_summary["avg_daily_demand"] * product_summary["lead_time_days"])
        + product_summary["safety_stock"]
    )

    product_summary["inventory_risk"] = product_summary.apply(inventory_risk_label, axis=1)

    final_columns = [
        "stockcode",
        "description",
        "avg_price",
        "min_price",
        "max_price",
        "price_range",
        "total_quantity",
        "total_revenue",
        "transaction_count",
        "elasticity",
        "elasticity_interpretation",
        "revenue_impact",
        "recommendation",
        "avg_daily_demand",
        "demand_std",
        "annual_demand",
        "lead_time_days",
        "ordering_cost",
        "holding_cost",
        "eoq",
        "safety_stock",
        "reorder_point",
        "inventory_risk",
        "first_sale_date",
        "last_sale_date",
    ]

    product_intelligence = product_summary[final_columns].copy()

    numeric_cols = product_intelligence.select_dtypes(include=[np.number]).columns
    product_intelligence[numeric_cols] = product_intelligence[numeric_cols].round(2)

    os.makedirs("outputs", exist_ok=True)
    product_intelligence.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved: {OUTPUT_PATH}")
    print(f"Rows: {len(product_intelligence)}")
    print(product_intelligence.head())


if __name__ == "__main__":
    main()