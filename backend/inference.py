import pandas as pd
import numpy as np
import mlflow
import datetime

# --- CONFIGURATION ---
# MLflow Tracking URI (Local by default, update to remote server if needed)
# mlflow.set_tracking_uri("http://your-mlflow-server:5000") 
MLFLOW_EXPERIMENT_NAME = "Demand_Forecasting"

import os
import joblib

import __main__

class SKUForecastModel:
    def __init__(self, models_dict):
        self.models_dict = models_dict
        
    def predict(self, stockcode, days_ahead=30):
        if stockcode in self.models_dict:
            m = self.models_dict[stockcode]
            future = m.make_future_dataframe(periods=days_ahead)
            forecast = m.predict(future)
            # Return sum of the predicted future days
            return forecast['yhat'].tail(days_ahead).sum()
        else:
            # Fallback for unknown SKUs
            return 150.0

__main__.SKUForecastModel = SKUForecastModel

def load_production_model(product_id):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODELS_DIR = os.getenv("MODELS_DIR", os.path.join(os.path.dirname(BASE_DIR), "models"))
    model_path = os.path.join(MODELS_DIR, "demand_model.pkl")
    try:
        model = joblib.load(model_path)
        print(f"✅ Demand model loaded from {model_path}")
        return model
    except Exception as e:
        print(f"❌ Error loading demand model for {product_id}: {e}")
        return None

def predict_demand(product_id: str, date_range: int) -> list:
    model = load_production_model(product_id)
    if model is None:
        return {"error": f"No production model found for product {product_id}"}
    
    try:
        # Our SimpleDemandModel has a custom predict method
        results = model.predict(product_id, date_range)
        return results
    except Exception as e:
        print(f"Prediction failed: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    # Example usage for Nandani
    print("--- Testing predict_demand ---")
    test_product = '85123A'
    days_to_predict = 30
    
    print(f"Required Inputs:")
    print(f"- product_id (string): e.g., '{test_product}'")
    print(f"- date_range (integer): e.g., {days_to_predict}")
    print("-" * 30)
    
    forecast = predict_demand(test_product, days_to_predict)
    print(f"Forecast for the next {days_to_predict} days for product {test_product}:")
    for f in forecast[:5]: # Showing first 5 days
        print(f)
    print("... (showing first 5 days)")
