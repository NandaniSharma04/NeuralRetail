import joblib
import os
import numpy as np

class EOQModel:
    def predict(self, demand_rate, order_cost, holding_cost_per_unit):
        # EOQ formula
        if holding_cost_per_unit <= 0:
            return 0
        eoq = np.sqrt((2 * demand_rate * order_cost) / holding_cost_per_unit)
        return round(eoq, 2)

def train_inventory_eoq():
    model = EOQModel()

    
    os.makedirs('../models', exist_ok=True)
    joblib.dump(model, '../models/eoq_calculator.pkl')
    print("EOQ model saved.")

if __name__ == "__main__":
    train_inventory_eoq()
