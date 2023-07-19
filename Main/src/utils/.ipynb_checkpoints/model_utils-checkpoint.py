# Description: This file contains the functions to train the LightGBM model and get the top features

import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error as mse


# Define the function to calculate the RMSE
def rmse(y_true, y_pred):

        return 'rmse', np.sqrt(mse(y_true, y_pred)), False
    

# Define the function to get the top features
def get_top_features(X_train, Y_train, X_val, Y_val, n_features=30):
    # Train the LightGBM model
    m_lgb = lgb.LGBMRegressor(random_state=384).fit(
        X_train, Y_train,
        eval_set=[(X_val, Y_val)],
        eval_metric=rmse,
        verbose=False
    )

    # Get the feature importances and names
    feature_importances = list(m_lgb.feature_importances_)
    feature_names = list(m_lgb.booster_.feature_name())

    # Create a DataFrame with feature names and importances
    df_feature_importances = pd.DataFrame({'feature_names': feature_names, 'importances': feature_importances})

    # Sort the DataFrame by importances in descending order
    df_feature_importances_sorted = df_feature_importances.sort_values(by='importances', ascending=False)

    # Get the top n_features feature names
    top_feature_names = list(df_feature_importances_sorted['feature_names'][:n_features])

    return top_feature_names