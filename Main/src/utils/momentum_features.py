# Note: This file contains functions to create momentum features
# Both Lag and Momentum features are shifted for 9 days because of the 9-day forecast horizon

def momentum_grouped(product_sales, i):

    # Momentum Features: Rolling averages and standard dev 9 & 14 days
    product_sales['q_roll_mean_9d'] = product_sales.groupby([i])['quantity'].shift(9).rolling(9, min_periods=1)\
        .mean().reset_index().quantity.fillna(0)
    product_sales['q_roll_std_9d'] = product_sales.groupby([i])['quantity'].shift(9).rolling(9, min_periods=1)\
        .std().reset_index().quantity.fillna(0)
    product_sales['q_roll_mean_14d'] = product_sales.groupby([i])['quantity'].shift(9).rolling(14, min_periods=1)\
        .mean().reset_index().quantity.fillna(0)
    product_sales['q_roll_std_14d'] = product_sales.groupby([i])['quantity'].shift(9).rolling(14, min_periods=1)\
        .std().reset_index().quantity.fillna(0)

    # Lag Features 9 days, 14 day, 28 days and 365 days
    product_sales['q_lag_9d'] = product_sales.groupby([i])['quantity'].shift(periods=9).fillna(0)
    product_sales['q_lag_14d'] = product_sales.groupby([i])['quantity'].shift(periods=14).fillna(0)
    product_sales['q_lag_28d'] = product_sales.groupby([i])['quantity'].shift(periods=28).fillna(0)
    product_sales['q_lag_365d'] = product_sales.groupby([i])['quantity'].shift(periods=365).fillna(0)

    # average of 9, 14 and 28 days
    product_sales["q_mean_lag_9_14_28"] = (product_sales['q_lag_9d'] + product_sales['q_lag_14d']
                                           + product_sales['q_lag_28d'])/3
    
    return product_sales

def momentum_ungrouped(product_sales):

    # Momentum Features: Rolling averages and standard dev 9 & 14 days
    product_sales['q_roll_mean_9d'] = product_sales["quantity"].shift(9).rolling(9, min_periods=1)\
        .mean().reset_index().quantity.fillna(0)
    product_sales['q_roll_std_9d'] = product_sales["quantity"].shift(9).rolling(9, min_periods=1)\
        .std().reset_index().quantity.fillna(0)
    product_sales['q_roll_mean_14d'] = product_sales["quantity"].shift(9).rolling(14, min_periods=1)\
        .mean().reset_index().quantity.fillna(0)
    product_sales['q_roll_std_14d'] = product_sales["quantity"].shift(9).rolling(14, min_periods=1)\
        .std().reset_index().quantity.fillna(0)

    # Lag Features 9 days, 14 day, 28 days and 365 days
    product_sales['q_lag_9d'] = product_sales["quantity"].shift(periods=9).fillna(0)
    product_sales['q_lag_14d'] = product_sales["quantity"].shift(periods=14).fillna(0)
    product_sales['q_lag_28d'] = product_sales["quantity"].shift(periods=28).fillna(0)
    product_sales['q_lag_365d'] = product_sales["quantity"].shift(periods=365).fillna(0)

    # average of 7, 14 and 28 days
    product_sales["q_mean_lag_9_14_28"] = (product_sales['q_lag_9d'] + product_sales['q_lag_14d']
                                           + product_sales['q_lag_28d'])/3
    
    return product_sales
