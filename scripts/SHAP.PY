"""
Computes per-customer SHAP values for the churn model and saves them to
    outputs/shap_values.csv

This lets the dashboard show "why is this customer at risk" instead of just a
probability number. Uses XGBoost's own built-in SHAP computation (pred_contribs=True)
rather than the `shap` package, to avoid a known compatibility bug where shap's
TreeExplainer fails to parse newer XGBoost model files ("could not convert string
to float: '[...]'").

The model expects exactly 7 features, in this order:
    Frequency, Monetary, AvgOrderValue, TotalQuantity, UniqueProducts, Tenure, AvgDaysBetweenPurchases

customer_insights.csv is known to have: frequency, monetary, avg_order_value, unique_products,
first_purchase_date, last_purchase_date -- but NOT TotalQuantity, Tenure, or AvgDaysBetweenPurchases
directly. This script:
  1. Maps the four it can find directly.
  2. Derives Tenure = (last_purchase_date - first_purchase_date).days
  3. Derives AvgDaysBetweenPurchases = Tenure / Frequency (approximation -- if your training
      code computed this differently, results will be slightly off; tell me and I'll adjust)
  4. Tries to derive TotalQuantity by summing a 'quantity' column in processed_data.csv per
      customer_id, if that column exists.

If any of these can't be built, the script prints exactly what's missing instead of guessing,
so you can paste that back to me.

Run from the project root:
    python scripts/generate_shap_values.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "models"
DATA_PATH = ROOT_DIR / "data" / "processed" / "processed_data.csv"
CUSTOMER_PATH = ROOT_DIR / "outputs" / "customer_insights.csv"
OUTPUT_PATH = ROOT_DIR / "outputs" / "shap_values.csv"

REQUIRED_FEATURES = ["Frequency", "Monetary", "AvgOrderValue", "TotalQuantity", "UniqueProducts", "Tenure", "AvgDaysBetweenPurchases"]


def main() -> None:
    try:
        import xgboost as xgb
    except ImportError:
        sys.exit("Missing dependency. Run: pip install xgboost --break-system-packages")

    # Paths for legacy pickle and stable native JSON format
    pickle_model_path = MODELS_DIR / "churn_model.pkl"
    native_model_path = MODELS_DIR / "churn_model.json"

    if not pickle_model_path.exists():
        sys.exit(f"Missing base model file: {pickle_model_path}")

    # Initialize a clean, version-agnostic XGBoost Classifier container
    model = xgb.XGBClassifier()

    # One-time extraction conversion to bypass future pickle warnings/corruption
    if not native_model_path.exists():
        print(f"Extracting legacy pickle model and exporting to version-agnostic JSON format...")
        try:
            legacy_model = joblib.load(pickle_model_path)
            legacy_model.save_model(str(native_model_path))
            print(f"Successfully generated native baseline: {native_model_path}")
        except Exception as e:
            sys.exit(f"Failed to auto-convert legacy pickle file: {e}")

    # Load from the safe, native layout
    model.load_model(str(native_model_path))

    if not CUSTOMER_PATH.exists():
        sys.exit(f"Missing {CUSTOMER_PATH}")
    customer = pd.read_csv(CUSTOMER_PATH, parse_dates=["first_purchase_date", "last_purchase_date"])

    features = pd.DataFrame(index=customer.index)
    features["customer_id"] = customer["customer_id"]

    direct_map = {
        "Frequency": "frequency",
        "Monetary": "monetary",
        "AvgOrderValue": "avg_order_value",
        "UniqueProducts": "unique_products",
    }
    missing_direct = [src for src in direct_map.values() if src not in customer.columns]
    if missing_direct:
        sys.exit(f"customer_insights.csv is missing expected column(s): {missing_direct}")
    for target, source in direct_map.items():
        features[target] = customer[source]

    if "first_purchase_date" in customer.columns and "last_purchase_date" in customer.columns:
        features["Tenure"] = (customer["last_purchase_date"] - customer["first_purchase_date"]).dt.days.clip(lower=1)
    else:
        sys.exit("Need 'first_purchase_date' and 'last_purchase_date' in customer_insights.csv to derive Tenure.")

    features["AvgDaysBetweenPurchases"] = features["Tenure"] / features["Frequency"].replace(0, 1)

    # TotalQuantity: try to sum from processed_data.csv
    if DATA_PATH.exists():
        raw = pd.read_csv(DATA_PATH)
        qty_col = next((c for c in ["quantity", "Quantity", "qty"] if c in raw.columns), None)
        id_col = next((c for c in ["customer_id", "CustomerID"] if c in raw.columns), None)
        if qty_col and id_col:
            qty_by_customer = raw.groupby(id_col)[qty_col].sum().rename("TotalQuantity")
            features = features.merge(qty_by_customer, left_on="customer_id", right_index=True, how="left")
        else:
            sys.exit(
                f"Could not find a quantity + customer id column pair in {DATA_PATH.name} "
                f"(looked for quantity/Quantity/qty and customer_id/CustomerID). "
                f"Tell me the actual column names and I'll fix this."
            )
    else:
        sys.exit(f"Missing {DATA_PATH} -- needed to derive TotalQuantity.")

    missing_final = [c for c in REQUIRED_FEATURES if c not in features.columns or features[c].isna().all()]
    if missing_final:
        sys.exit(f"Could not build these required feature(s): {missing_final}")

    features[REQUIRED_FEATURES] = features[REQUIRED_FEATURES].fillna(0)
    X = features[REQUIRED_FEATURES]

    # Compute SHAP matrix values using native Booster mechanics
    booster = model.get_booster()
    dmatrix = xgb.DMatrix(X, feature_names=REQUIRED_FEATURES)
    contribs = booster.predict(dmatrix, pred_contribs=True)  # shape: (n_rows, n_features + 1)

    shap_df = pd.DataFrame(contribs[:, :-1], columns=[f"shap_{c}" for c in REQUIRED_FEATURES])
    shap_df.insert(0, "customer_id", features["customer_id"].values)
    shap_df["base_value"] = contribs[:, -1]
    shap_df["predicted_probability"] = model.predict_proba(X)[:, 1]

    # Identify the feature with the largest absolute SHAP contribution per customer row
    shap_cols = [f"shap_{c}" for c in REQUIRED_FEATURES]
    shap_df["top_driver"] = shap_df[shap_cols].abs().idxmax(axis=1).str.replace("shap_", "", regex=False)
    shap_df["top_driver_impact"] = shap_df.apply(lambda r: r[f"shap_{r['top_driver']}"], axis=1)

    shap_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {OUTPUT_PATH} for {len(shap_df)} customers.")
    print("Columns:", list(shap_df.columns))


if __name__ == "__main__":
    main()