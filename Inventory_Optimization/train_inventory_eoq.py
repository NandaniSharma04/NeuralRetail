import pandas as pd
import numpy as np
import joblib
import os

class InventoryCalculator:
    def __init__(self, stats_df):
        self.stats = stats_df.set_index('stockcode')
        # fallback averages
        self.avg_demand = self.stats['annual_demand'].mean()
        
    def get_reorder_info(self, stockcode, manual_demand=None, manual_order_cost=None, manual_holding_cost=None):
        if stockcode in self.stats.index:
            row = self.stats.loc[stockcode]
            return {
                "EOQ": row['eoq'],
                "SafetyStock": row['safety_stock'],
                "ReorderPoint": row['reorder_point']
            }
        else:
            # Calculate dynamically if not found
            D = manual_demand if manual_demand else self.avg_demand
            S = manual_order_cost if manual_order_cost else 25.0
            H = manual_holding_cost if manual_holding_cost else 2.0
            eoq = np.sqrt((2 * D * S) / H)
            return {
                "EOQ": round(eoq, 2),
                "SafetyStock": round(np.sqrt(D), 2),
                "ReorderPoint": round(D/365 * 7 + np.sqrt(D), 2)
            }

def train_inventory_eoq():
    print("Loading data...")
    df = pd.read_csv('../data/processed/processed_data.csv', parse_dates=['invoicedate'])
    
    # Take top 100 SKUs
    top_skus = df.groupby('stockcode')['total_amount'].sum().nlargest(100).index
    df_top = df[df['stockcode'].isin(top_skus)].copy()
    
    # Daily demand aggregation
    df_top['date'] = df_top['invoicedate'].dt.date
    daily_demand = df_top.groupby(['stockcode', 'date'])['quantity'].sum().reset_index()
    
    # Calculate demand rate (D) and standard deviation for safety stock
    stats = daily_demand.groupby('stockcode')['quantity'].agg(['mean', 'std']).reset_index()
    stats.columns = ['stockcode', 'daily_demand_mean', 'daily_demand_std']
    stats.fillna(0, inplace=True)
    
    # Annualize Demand (assume 365 days)
    stats['annual_demand'] = stats['daily_demand_mean'] * 365
    
    # Synthetic injection for lead_time and costs
    np.random.seed(42)
    stats['lead_time_days'] = np.random.randint(3, 14, len(stats))
    stats['order_cost'] = np.random.uniform(10, 50, len(stats))
    
    # Holding cost: assume 20% of average price
    avg_price = df_top.groupby('stockcode')['price'].mean().reset_index()
    stats = stats.merge(avg_price, on='stockcode')
    stats['holding_cost'] = stats['price'] * 0.20
    stats['holding_cost'] = stats['holding_cost'].clip(lower=0.5)
    
    # EOQ Calculation
    # EOQ = sqrt( (2 * D * S) / H )
    stats['eoq'] = np.sqrt( (2 * stats['annual_demand'] * stats['order_cost']) / stats['holding_cost'] )
    stats['eoq'] = np.ceil(stats['eoq'])
    
    # Safety Stock Calculation (Z=1.65 for 95% service level)
    Z = 1.65
    stats['safety_stock'] = np.ceil(Z * stats['daily_demand_std'] * np.sqrt(stats['lead_time_days']))
    
    # Reorder Point
    stats['reorder_point'] = np.ceil((stats['daily_demand_mean'] * stats['lead_time_days']) + stats['safety_stock'])
    
    print("EOQ and Reorder points calculated for Top 100 SKUs.")
    
    model = InventoryCalculator(stats)
    
    os.makedirs('../models', exist_ok=True)
    joblib.dump(model, '../models/inventory_model.pkl')
    print("Inventory model saved to ../models/inventory_model.pkl")

if __name__ == "__main__":
    train_inventory_eoq()
