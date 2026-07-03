import pandas as pd
import numpy as np
import joblib
import os
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

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

def train_demand_model():
    print("Loading data for Demand Forecasting...")
    df = pd.read_csv('../data/processed/processed_data.csv', parse_dates=['invoicedate'])
    
    # Top 50 SKUs by volume
    top_skus = df.groupby('stockcode')['quantity'].sum().nlargest(50).index
    df_top = df[df['stockcode'].isin(top_skus)]
    
    df_top['date'] = df_top['invoicedate'].dt.date
    
    # We will train one global Prophet model or store multiple.
    # To keep the API simple, let's create a dictionary of models for top SKUs
    models_dict = {}
    
    print(f"Training Prophet models for {len(top_skus)} SKUs...")
    for sku in top_skus:
        sku_data = df_top[df_top['stockcode'] == sku].groupby('date')['quantity'].sum().reset_index()
        sku_data.columns = ['ds', 'y']
        
        # Need at least 14 days of data to fit
        if len(sku_data) > 14:
            m = Prophet(daily_seasonality=False)
            m.fit(sku_data)
            models_dict[sku] = m
    
    final_model = SKUForecastModel(models_dict)
    
    os.makedirs('../models', exist_ok=True)
    joblib.dump(final_model, '../models/demand_model.pkl')
    print("Demand Forecast model saved to ../models/demand_model.pkl")

if __name__ == "__main__":
    train_demand_model()
