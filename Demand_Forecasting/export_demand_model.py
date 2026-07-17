import pandas as pd
import numpy as np
import joblib
import os

class SimpleDemandModel:
    def predict(self, product_id, days):
        # Generates a simple forecast simulating Prophet/LSTM
        np.random.seed(hash(product_id) % (2**32))
        base = np.random.randint(50, 200)
        
        forecast = []
        import datetime
        today = datetime.date.today()
        
        for i in range(1, days + 1):
            date = today + datetime.timedelta(days=i)
            # Add simple weekly seasonality
            seasonality = 20 if date.weekday() >= 5 else 0
            quantity = max(0, base + seasonality + np.random.normal(0, 10))
            forecast.append({
                "date": date.strftime("%Y-%m-%d"),
                "forecasted_quantity": round(float(quantity), 2)
            })
        return forecast

def export_demand_model():
    model = SimpleDemandModel()
    os.makedirs('../models', exist_ok=True)
    joblib.dump(model, '../models/demand_model.pkl')
    print("Demand model exported successfully.")

if __name__ == "__main__":
    export_demand_model()
