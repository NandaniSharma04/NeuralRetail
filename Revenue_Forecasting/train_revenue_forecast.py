import pandas as pd
import numpy as np
import joblib
import os
from prophet import Prophet
from lightgbm import LGBMRegressor
import warnings
warnings.filterwarnings('ignore')

class HybridRevenueModel:
    def __init__(self, prophet, lgb, last_7_days_y):
        self.prophet = prophet
        self.lgb = lgb
        self.last_7_days_y = last_7_days_y # List to compute lags
        
    def predict(self, date_pd):
        # Create a dataframe for prophet
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

def train_revenue_forecast():
    print("Loading data for Revenue Forecasting...")
    df = pd.read_csv('../data/processed/processed_data.csv', parse_dates=['invoicedate'])
    
    # Aggregate daily revenue
    df['date'] = df['invoicedate'].dt.date
    daily_revenue = df.groupby('date')['total_amount'].sum().reset_index()
    daily_revenue.columns = ['ds', 'y'] # Prophet format
    
    # Ensure ds is datetime
    daily_revenue['ds'] = pd.to_datetime(daily_revenue['ds'])
    
    print("Training Prophet Model for trend and seasonality extraction...")
    prophet_model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    prophet_model.fit(daily_revenue)
    
    # Predict on the same dataset to extract components
    forecast = prophet_model.predict(daily_revenue[['ds']])
    
    # Merge Prophet features into main dataframe for LightGBM
    lgbm_data = daily_revenue.copy()
    lgbm_data['trend'] = forecast['trend'].values
    lgbm_data['weekly'] = forecast['weekly'].values
    if 'yearly' in forecast.columns:
        lgbm_data['yearly'] = forecast['yearly'].values
    else:
        lgbm_data['yearly'] = 0.0
        
    # Create lag features
    lgbm_data['y_lag1'] = lgbm_data['y'].shift(1)
    lgbm_data['y_lag7'] = lgbm_data['y'].shift(7)
    
    # Drop NAs from lags
    lgbm_data = lgbm_data.dropna()
    
    print("Training LightGBM Hybrid Model...")
    features = ['trend', 'weekly', 'yearly', 'y_lag1', 'y_lag7']
    X = lgbm_data[features]
    y = lgbm_data['y']
    
    lgb_model = LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=42)
    lgb_model.fit(X, y)
    
    last_7_days = lgbm_data['y'].tail(7).values.tolist()
    final_model = HybridRevenueModel(prophet_model, lgb_model, last_7_days)
    
    os.makedirs('../models', exist_ok=True)
    joblib.dump(final_model, '../models/revenue_model.pkl')
    print("Revenue Forecast model saved to ../models/revenue_model.pkl")


if __name__ == "__main__":
    train_revenue_forecast()
