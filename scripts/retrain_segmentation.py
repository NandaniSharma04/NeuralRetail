"""
Retrain segmentation models (KMeans + StandardScaler) with the current sklearn version
to eliminate InconsistentVersionWarning on server startup.
"""
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "data", "raw", "new_cleaned_retail_data_with_churn.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

print(f"Loading data from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df):,} rows")

# ── Build RFM table (same logic as the notebook) ──────────────────────────────
df['invoicedate'] = pd.to_datetime(df['invoicedate'])
current_date = df['invoicedate'].max() + pd.Timedelta(days=1)

rfm = df.groupby('customer_id').agg(
    Recency   = ('invoicedate',  lambda d: (current_date - d.max()).days),
    Frequency = ('invoice',      'nunique'),
    Monetary  = ('total_amount', 'sum')
).reset_index()

rfm = rfm[rfm['Monetary'] > 0]
print(f"Customers in RFM: {len(rfm):,}")

# ── Log-transform + scale (same as notebook) ──────────────────────────────────
rfm_log = rfm[['Recency', 'Frequency', 'Monetary']].copy()
for col in ['Recency', 'Frequency', 'Monetary']:
    rfm_log[col] = np.log1p(rfm_log[col])

scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_log)

# ── Pick best k (6-10) by Silhouette score ────────────────────────────────────
from sklearn.metrics import silhouette_score

best_k, best_score, best_model = 6, -1, None
print("Evaluating K-Means for k=6 to k=10 ...")
for k in range(6, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(rfm_scaled)
    score  = silhouette_score(rfm_scaled, labels)
    print(f"  k={k}  silhouette={score:.4f}")
    if score > best_score:
        best_score, best_k, best_model = score, k, km

print(f"\nBest k={best_k}  (silhouette={best_score:.4f})")

# ── Save models ───────────────────────────────────────────────────────────────
scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
seg_path    = os.path.join(MODELS_DIR, "segmentation_model.pkl")

joblib.dump(scaler,     scaler_path)
joblib.dump(best_model, seg_path)

print(f"\nSaved: {scaler_path}")
print(f"Saved: {seg_path}")
print("\nDone! Restart the backend server to load the updated models.")
