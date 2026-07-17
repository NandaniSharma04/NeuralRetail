from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import os
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi import Depends, status

# Import auth module
from backend.auth import (
    UserCreate, UserLogin, Token, verify_password, get_password_hash,
    create_access_token, ACCESS_TOKEN_EXPIRE_MINUTES, timedelta, jwt, SECRET_KEY, ALGORITHM
)

from sqlalchemy.orm import Session
from backend.database import engine, Base, get_db
from backend import models

# Create database tables
models.Base.metadata.create_all(bind=engine)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

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
# In local development, models are in the parent directory 'models/'. In Docker, we can override via env var.
MODELS_DIR = os.getenv("MODELS_DIR", os.path.join(os.path.dirname(BASE_DIR), "models"))

try:
    churn_model = joblib.load(os.path.join(MODELS_DIR, "churn_model.pkl"))
    print("SUCCESS: Churn model loaded successfully")
except Exception as e:
    print(f"ERROR: Churn model failed to load: {e}")
    churn_model = None

try:
    segmentation_model = joblib.load(os.path.join(MODELS_DIR, "segmentation_model.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    print("SUCCESS: Segmentation model loaded successfully")
except Exception as e:
    print(f"ERROR: Segmentation model failed to load: {e}")
    segmentation_model = None
    scaler = None

import __main__

class CausalElasticityModel:
    def __init__(self, elasticity):
        self.default_elasticity = elasticity
        
    def predict(self, X_df):
        import numpy as np
        return np.log(100) + self.default_elasticity * X_df['log_price'].values

class InventoryCalculator:
    def __init__(self, stats_df):
        self.stats = stats_df.set_index('stockcode')
        self.avg_demand = self.stats['annual_demand'].mean()
        
    def get_reorder_info(self, stockcode, manual_demand=None, manual_order_cost=None, manual_holding_cost=None):
        import numpy as np
        if stockcode in self.stats.index:
            row = self.stats.loc[stockcode]
            return {
                "EOQ": row['eoq'],
                "SafetyStock": row['safety_stock'],
                "ReorderPoint": row['reorder_point']
            }
        else:
            D = manual_demand if manual_demand else self.avg_demand
            S = manual_order_cost if manual_order_cost else 25.0
            H = manual_holding_cost if manual_holding_cost else 2.0
            eoq = np.sqrt((2 * D * S) / H)
            return {
                "EOQ": round(eoq, 2),
                "SafetyStock": round(np.sqrt(D), 2),
                "ReorderPoint": round(D/365 * 7 + np.sqrt(D), 2)
            }

class HybridRevenueModel:
    def __init__(self, prophet, lgb, last_7_days_y):
        self.prophet = prophet
        self.lgb = lgb
        self.last_7_days_y = last_7_days_y
        
    def predict(self, date_pd):
        import pandas as pd
        df_p = pd.DataFrame({'ds': [date_pd]})
        f = self.prophet.predict(df_p)
        
        trend = f['trend'].values[0]
        weekly = f['weekly'].values[0]
        yearly = f['yearly'].values[0] if 'yearly' in f.columns else 0.0
        
        y_lag1 = self.last_7_days_y[-1]
        y_lag7 = self.last_7_days_y[0]
        
        X_pred = pd.DataFrame({
            'trend': [trend],
            'weekly': [weekly],
            'yearly': [yearly],
            'y_lag1': [y_lag1],
            'y_lag7': [y_lag7]
        })
        
        return self.lgb.predict(X_pred)[0]

__main__.CausalElasticityModel = CausalElasticityModel
__main__.InventoryCalculator = InventoryCalculator
__main__.HybridRevenueModel = HybridRevenueModel

try:
    elasticity_model = joblib.load(os.path.join(MODELS_DIR, "elasticity_model.pkl"))
    print("SUCCESS: Elasticity model loaded successfully")
except Exception as e:
    print(f"ERROR: Elasticity model failed to load: {e}")
    elasticity_model = None

try:
    eoq_model = joblib.load(os.path.join(MODELS_DIR, "inventory_model.pkl"))
    print("SUCCESS: EOQ model loaded successfully")
except Exception as e:
    print(f"ERROR: EOQ model failed to load: {e}")
    eoq_model = None

try:
    revenue_model = joblib.load(os.path.join(MODELS_DIR, "revenue_model.pkl"))
    print("SUCCESS: Revenue model loaded successfully")
except Exception as e:
    print(f"ERROR: Revenue model failed to load: {e}")
    revenue_model = None


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

class ElasticityRequest(BaseModel):
    price: float
    competitor_price: float
    ad_spend: float

class EOQRequest(BaseModel):
    stockcode: str
    demand_rate: float = None
    order_cost: float = None
    holding_cost: float = None

class RevenueRequest(BaseModel):
    date_string: str


# ── Authentication Endpoints ───────────────────────────────────────────────

@app.post("/auth/register", response_model=Token)
def register_user(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.username == user.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    hashed_password = get_password_hash(user.password)
    new_user = models.User(username=user.username, hashed_password=hashed_password)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/auth/login", response_model=Token)
def login_user(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.username == form_data.username).first()
    if not user:
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    
    if not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": form_data.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Could not validate credentials")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Could not validate credentials")
    user = db.query(models.User).filter(models.User.username == username).first()
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return user

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

from backend.inference import predict_demand as inf_predict_demand

@app.post("/predict/demand")
def predict_demand(data: DemandRequest):
    result = inf_predict_demand(data.product_id, data.days)
    if isinstance(result, dict) and "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    return {
        "product_id": data.product_id,
        "forecast_days": data.days,
        "forecast": result
    }

@app.post("/predict/elasticity")
def predict_elasticity(data: ElasticityRequest, current_user = Depends(get_current_user)):
    if elasticity_model is None:
        raise HTTPException(status_code=503, detail="Elasticity model not loaded")
    import numpy as np
    import pandas as pd
    X = pd.DataFrame({
        'log_price': [np.log(data.price)],
        'log_competitor_price': [np.log(data.competitor_price)],
        'log_ad_spend': [np.log(data.ad_spend)]
    })
    log_demand = elasticity_model.predict(X)[0]
    return {
        "price": data.price,
        "predicted_demand": round(np.exp(log_demand), 2),
        "price_elasticity": round(elasticity_model.default_elasticity, 3)
    }

@app.post("/inventory/reorder")
def calculate_eoq(data: EOQRequest, current_user = Depends(get_current_user)):
    if eoq_model is None:
        raise HTTPException(status_code=503, detail="EOQ model not loaded")
    
    info = eoq_model.get_reorder_info(
        data.stockcode, 
        manual_demand=data.demand_rate, 
        manual_order_cost=data.order_cost, 
        manual_holding_cost=data.holding_cost
    )
    return info

@app.post("/predict/revenue")
def predict_revenue(data: RevenueRequest, current_user = Depends(get_current_user)):
    if revenue_model is None:
        raise HTTPException(status_code=503, detail="Revenue model not loaded")
    import pandas as pd
    
    date_pd = pd.to_datetime(data.date_string)
    revenue = revenue_model.predict(date_pd)
    
    return {
        "date": data.date_string,
        "predicted_revenue": round(revenue, 2)
    }
