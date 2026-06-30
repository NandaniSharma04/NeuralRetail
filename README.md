# NeuralRetail — Churn Prediction Module

## Project Overview
NeuralRetail is an AI-powered retail analytics platform built for Amdox Technologies.
This module covers the **Churn Prediction** component — predicting which customers
are at risk of churning based on their purchase behavior.

## Status: Completed ✅

## What Was Done

### 1. Data Preparation
- Used the cleaned dataset: `new_cleaned_retail_data_with_churn.csv`
- Source: UCI Online Retail II dataset, cleaned by the data team
- Columns: invoice, stockcode, description, quantity, invoicedate, price,
  customer_id, country, total_amount, year, month, day, day_of_week, Recency, Churn

### 2. Identified and Fixed Data Leakage
The original `Churn` label was created using a hard cutoff on `Recency`
(customers inactive for 90+ days were labeled as churned). This meant any model
using `Recency` as an input feature would achieve artificially perfect accuracy
(100%) by simply re-deriving the original rule, rather than learning genuine
behavioral patterns. This was identified and `Recency` was excluded from the
final feature set to ensure an honest, generalizable model.

### 3. Feature Engineering (RFM + Behavioral Features)
Built the following features per customer:
- **Frequency** — number of unique invoices (orders)
- **Monetary** — total amount spent
- **AvgOrderValue** — average spend per order
- **TotalQuantity** — total items purchased
- **UniqueProducts** — number of distinct products purchased
- **Tenure** — days between first and last purchase
- **AvgDaysBetweenPurchases** — average gap between orders

### 4. Model Training & Evaluation
- Trained **XGBoost** and **LightGBM** classifiers
- Handled class imbalance using `scale_pos_weight`
- Hyperparameter tuning via `GridSearchCV` (5-fold cross-validation)
- Final model: XGBoost (max_depth=3, learning_rate=0.03, n_estimators=100)

**Results:**
| Metric | Score |
|---|---|
| AUC-ROC | 0.78 |
| Accuracy | 70% |
| Precision (Churn class) | 0.53 |
| Recall (Churn class) | 0.80 |

### 5. Explainability (SHAP)
- Global feature importance via SHAP beeswarm plot
- Individual prediction explanations via SHAP waterfall plot
- **Key insight:** `Tenure` (how long someone has been a customer) is the
  strongest predictor of churn, followed by `UniqueProducts` (product variety).

### 6. Model Artifact
- Final trained model saved as `models/churn_model.pkl`
- Can be loaded via `joblib.load('churn_model.pkl')` for reuse in the backend API

## Tech Stack
- Python 3.12
- pandas, numpy — data processing
- XGBoost, LightGBM — modeling
- scikit-learn — train/test split, evaluation, grid search
- SHAP — model explainability
- joblib — model persistence
- Jupyter Notebook — development environment

## Folder Structure
NeuralRetail/
├── notebooks/
│   └── churn_prediction.ipynb
├── models/
│   └── churn_model.pkl
├── data/
│   └── new_cleaned_retail_data_with_churn.csv
└── README.md
## Next Steps (Project-wide)
- [ ] Demand forecasting model (Prophet)
- [ ] Customer segmentation (K-Means)
- [ ] Backend API integration (FastAPI)
- [ ] Dashboard (Streamlit + Power BI)
