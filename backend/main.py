from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import os

# ── App Setup ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="NeuralRetail API",
    description="AI-Powered Retail Analytics — Churn Prediction, Demand Forecasting, Customer Segmentation",
    version="1.0.0"
)

# Allow Streamlit and browser to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load Models at Startup ─────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

try:
    churn_model = joblib.load(os.path.join(MODELS_DIR, "churn_model.pkl"))
    print("✅ Churn model loaded successfully")
except Exception as e:
    print(f"❌ Churn model failed to load: {e}")
    churn_model = None

try:
    segmentation_model = joblib.load(os.path.join(MODELS_DIR, "segmentation_model.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    print("✅ Segmentation model loaded successfully")
except Exception as e:
    print(f"❌ Segmentation model failed to load: {e}")
    segmentation_model = None
    scaler = None

# ── Request Models (What data each endpoint expects) ──────────────────────
class ChurnRequest(BaseModel):
    Frequency: float
    Monetary: float
    AvgOrderValue: float
    TotalQuantity: float
    UniqueProducts: float
    Tenure: float
    AvgDaysBetweenPurchases: float

class SegmentRequest(BaseModel):
    Frequency: float
    Monetary: float
    Tenure: float

class DemandRequest(BaseModel):
    product_id: str
    days: int = 30

# ── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "project": "NeuralRetail",
        "version": "1.0.0",
        "status": "running",
        "endpoints": [
            "/health",
            "/predict/churn",
            "/predict/demand",
            "/segment/score",
            "/docs"
        ]
    }

@app.get("/health")
def health():
    return {
        "status": "ok",
        "message": "NeuralRetail API is running!",
        "models_loaded": {
            "churn_model": churn_model is not None,
            "segmentation_model": segmentation_model is not None,
        }
    }

@app.post("/predict/churn")
def predict_churn(data: ChurnRequest):
    if churn_model is None:
        raise HTTPException(status_code=503, detail="Churn model not loaded")
    
    features = [[
        data.Frequency,
        data.Monetary,
        data.AvgOrderValue,
        data.TotalQuantity,
        data.UniqueProducts,
        data.Tenure,
        data.AvgDaysBetweenPurchases
    ]]
    
    prediction = churn_model.predict(features)[0]
    probability = churn_model.predict_proba(features)[0][1]
    
    if probability > 0.7:
        risk = "High Risk"
        action = "Contact customer immediately with a retention offer"
    elif probability > 0.4:
        risk = "Medium Risk"
        action = "Send re-engagement email campaign"
    else:
        risk = "Low Risk"
        action = "Maintain regular communication"

    return {
        "churn_prediction": int(prediction),
        "churn_probability": round(float(probability), 3),
        "risk_level": risk,
        "recommended_action": action
    }

@app.post("/segment/score")
def segment_customer(data: SegmentRequest):
    if segmentation_model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Segmentation model not loaded")
    
    features = [[data.Frequency, data.Monetary, data.Tenure]]
    scaled = scaler.transform(features)
    segment = segmentation_model.predict(scaled)[0]
    
    segment_map = {
        0: "Loyal Customers",
        1: "VIP Customers",
        2: "At Risk Customers",
        3: "High Value Customers"
    }
    
    return {
        "segment_id": int(segment),
        "segment_name": segment_map.get(int(segment), "Unknown"),
        "description": {
            "Loyal Customers": "Regular buyers with long history",
            "VIP Customers": "Highest spend and frequency",
            "At Risk Customers": "Low activity, needs attention",
            "High Value Customers": "High spend, frequent buyers"
        }.get(segment_map.get(int(segment), "Unknown"), "")
    }

@app.post("/predict/demand")
def predict_demand(data: DemandRequest):
    # Returns simulated demand forecast
    # Replace with real Prophet model when available
    import datetime
    today = datetime.date.today()
    forecast = []
    base = np.random.randint(50, 200)
    
    for i in range(1, data.days + 1):
        date = today + datetime.timedelta(days=i)
        quantity = max(0, base + np.random.normal(0, 10))
        forecast.append({
            "date": date.strftime("%Y-%m-%d"),
            "forecasted_quantity": round(float(quantity), 2)
        })
    
    return {
        "product_id": data.product_id,
        "forecast_days": data.days,
        "forecast": forecast
    }