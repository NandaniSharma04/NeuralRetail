import pandas as pd
import numpy as np
import joblib
import os
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

class CausalElasticityModel:
    def __init__(self, elasticity):
        self.default_elasticity = elasticity
        
    def predict(self, X_df):
        return np.log(100) + self.default_elasticity * X_df['log_price'].values

def train_price_elasticity():
    print("Loading data for Price Elasticity...")
    df = pd.read_csv('../data/processed/processed_data.csv', parse_dates=['invoicedate'])
    
    # Aggregate data to weekly level per product
    df['week'] = df['invoicedate'].dt.to_period('W').dt.start_time
    
    # Take Top 50 SKUs
    top_skus = df.groupby('stockcode')['total_amount'].sum().nlargest(50).index
    df_top = df[df['stockcode'].isin(top_skus)]
    
    weekly_data = df_top.groupby(['stockcode', 'week']).agg(
        quantity=('quantity', 'sum'),
        price=('price', 'mean')
    ).reset_index()
    
    weekly_data = weekly_data[(weekly_data['price'] > 0) & (weekly_data['quantity'] > 0)]
    
    print("Synthesizing missing confounders (competitor_price, ad_spend)...")
    np.random.seed(42)
    weekly_data['competitor_price'] = weekly_data['price'] + np.random.normal(0, 0.5, len(weekly_data))
    weekly_data['competitor_price'] = weekly_data['competitor_price'].clip(lower=0.1)
    weekly_data['ad_spend'] = np.random.uniform(100, 1000, len(weekly_data))
    
    # Log transform for Log-Log OLS (Elasticity)
    weekly_data['log_quantity'] = np.log(weekly_data['quantity'])
    weekly_data['log_price'] = np.log(weekly_data['price'])
    weekly_data['log_competitor_price'] = np.log(weekly_data['competitor_price'])
    weekly_data['log_ad_spend'] = np.log(weekly_data['ad_spend'])
    
    print("Training Log-Log OLS baseline for Causal Elasticity...")
    # Outcome: log_quantity, Predictors: log_price (Treatment), log_competitor_price, log_ad_spend (Confounders)
    X = weekly_data[['log_price', 'log_competitor_price', 'log_ad_spend']]
    X = sm.add_constant(X)
    Y = weekly_data['log_quantity']
    
    model = sm.OLS(Y, X).fit()
    ate = model.params['log_price']
    
    print(model.summary())
    print(f"\nOverall Average Price Elasticity (Log-Log): {ate:.3f}")
    
    final_model = CausalElasticityModel(ate)
    
    os.makedirs('../models', exist_ok=True)
    joblib.dump(final_model, '../models/elasticity_model.pkl')
    print("Price Elasticity model saved to ../models/elasticity_model.pkl")

if __name__ == "__main__":
    train_price_elasticity()
