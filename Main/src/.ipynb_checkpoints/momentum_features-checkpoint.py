import pandas as pd

def momentum_grouped(product_sales, i): 

    # Momentum Features: Rolling averages and standard dev 7 & 14 days  
    product_sales['q_roll_mean_7d'] = product_sales.groupby([i])['quantity'].rolling(7).mean().reset_index().quantity
    product_sales['q_roll_std_7d'] = product_sales.groupby([i])['quantity'].rolling(7).std().reset_index().quantity
    product_sales['q_roll_mean_14d'] = product_sales.groupby([i])['quantity'].rolling(14).mean().reset_index().quantity
    product_sales['q_roll_std_14d'] = product_sales.groupby([i])['quantity'].rolling(14).std().reset_index().quantity

    # Lag Features 1 day, 7 day, 14 day & 28 day
    product_sales['q_lag_1d'] = product_sales.groupby([i])['quantity'].shift(periods=1)
    product_sales['q_lag_7d'] = product_sales.groupby([i])['quantity'].shift(periods=7)
    product_sales['q_lag_14d'] = product_sales.groupby([i])['quantity'].shift(periods=14)
    product_sales['q_lag_28d'] = product_sales.groupby([i])['quantity'].shift(periods=28)
    product_sales["q_mean_lag_7_14_28"] = (product_sales['q_lag_7d'] + product_sales['q_lag_14d'] + product_sales['q_lag_28d'])/3
    
    return product_sales

def momentum_ungrouped(product_sales):
    # Momentum Features: Rolling averages and standard dev 7 & 14 days  
    product_sales['q_roll_mean_7d'] = product_sales["quantity"].rolling(7).mean().reset_index().quantity
    product_sales['q_roll_std_7d'] = product_sales["quantity"].rolling(7).std().reset_index().quantity
    product_sales['q_roll_mean_14d'] = product_sales["quantity"].rolling(14).mean().reset_index().quantity
    product_sales['q_roll_std_14d'] = product_sales["quantity"].rolling(14).std().reset_index().quantity

    # Lag Features 1 day, 7 day, 14 day & 28 day
    product_sales['q_lag_1d'] = product_sales["quantity"].shift(periods=1)
    product_sales['q_lag_7d'] = product_sales["quantity"].shift(periods=7)
    product_sales['q_lag_14d'] = product_sales["quantity"].shift(periods=14)
    product_sales['q_lag_28d'] = product_sales["quantity"].shift(periods=28)
    product_sales["q_mean_lag_7_14_28"] = (product_sales['q_lag_7d'] + product_sales['q_lag_14d'] + product_sales['q_lag_28d'])/3
    
    return product_sales