import pandas as pd
import numpy as np
import mlflow
import datetime

# --- CONFIGURATION ---
# MLflow Tracking URI (Local by default, update to remote server if needed)
# mlflow.set_tracking_uri("http://your-mlflow-server:5000") 
MLFLOW_EXPERIMENT_NAME = "Demand_Forecasting"

def load_production_model(product_id):
    """
    Loads the registered production model for a given product from MLflow.
    In a real MLflow model registry setup, you would load via:
    model_uri = f"models:/product_{product_id}_demand_model/Production"
    model = mlflow.pyfunc.load_model(model_uri)
    """
    # Placeholder for MLflow loading logic. 
    # Because we're simulating without a running MLflow server in this script, 
    # here's how Nandani will load the model:
    try:
        # Example URI structure if using MLflow Model Registry
        model_uri = f"models:/Demand_Forecasting_{product_id}/Production"
        print(f"Attempting to load model from: {model_uri}")
        # model = mlflow.pyfunc.load_model(model_uri)
        # return model
        
        # Simulating a loaded model for demonstration
        return f"Loaded_Model_For_{product_id}" 
    except Exception as e:
        print(f"Error loading model for {product_id}: {e}")
        return None

def predict_demand(product_id: str, date_range: int) -> list:
    """
    Predicts the demand for a given product over a specified number of future days.
    
    Args:
        product_id (str): The StockCode of the product (e.g., '85123A').
        date_range (int): Number of days to forecast into the future (e.g., 30).
        
    Returns:
        list of dicts: The forecasted quantity per day.
            Format: [{'date': 'YYYY-MM-DD', 'forecasted_quantity': float}, ...]
    """
    # 1. Load the Production Model
    model = load_production_model(product_id)
    if model is None:
        return {"error": f"No production model found for product {product_id}"}
    
    # 2. Prepare the input dataframe for prediction
    # Required Input Columns for Prophet: 'ds' (datetime)
    # Required Input for LSTM: A sequence of the last 'N' days of historical data (e.g., last 30 days)
    
    # Generating future dates
    today = datetime.date.today()
    future_dates = [today + datetime.timedelta(days=i) for i in range(1, date_range + 1)]
    future_df = pd.DataFrame({'ds': future_dates})
    
    # 3. Make the prediction
    # Note: If it's a Prophet model loaded via MLflow pyfunc:
    # forecast = model.predict(future_df)
    # forecasted_quantity = forecast['yhat'].values
    
    # Simulating prediction output
    forecasted_quantity = np.random.normal(loc=20, scale=5, size=date_range)
    forecasted_quantity = np.maximum(0, forecasted_quantity) # Cannot have negative demand
    
    # 4. Format the output
    results = []
    for dt, qty in zip(future_dates, forecasted_quantity):
        results.append({
            'date': dt.strftime('%Y-%m-%d'),
            'forecasted_quantity': round(float(qty), 2)
        })
        
    return results

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
