# Description: This file contains the utility functions for the model training and evaluation.

import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error as mse


# Define the function to calculate the rmse
def rmse(y_true, y_pred):
    return 'rmse', np.sqrt(mse(y_true, y_pred)), False
    

# Define the function to get the top features
def get_top_features(X_train, Y_train, X_val, Y_val, n_features=40):
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


def custom_backtesting(df, x, y, model, tscv):
    # Initialize an empty lists to store predictions, actuals and dates
    preds = []
    actuals = []
    dates = []
    group = []

    # loop over pre-defined time series split / time series cross validation (tscv)
    for i, (train_index, test_index) in enumerate(tscv.split(df)):
        # Train and test data
        x_train_fold, x_test_fold = x.iloc[train_index], x.iloc[test_index]
        y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]

        # Append Dates
        dates_fold = df.date.iloc[test_index].tolist()

        # Append Group
        group_fold = df.group.iloc[test_index].tolist()

        # Track
        print(f"Fold Nr. = {i}")

        # Fit the model on the current fold
        model.fit(x_train_fold, y_train_fold)

        # Predict on the current test fold
        y_pred_fold = model.predict(x_test_fold)

        # append scores and predictions
        preds.append(y_pred_fold.tolist())
        actuals.append(y_test_fold.tolist())
        dates.append(dates_fold)
        group.append(group_fold)

    # make df that holds predictions and actuals
    # flatten nested lists
    dates = np.concatenate(dates).tolist()
    actuals = np.concatenate(actuals).tolist()
    preds = np.concatenate(preds).tolist()
    group = np.concatenate(group).tolist()
    # Error analysis
    error = pd.DataFrame({
        "date": dates,
        "actual": actuals,
        "pred": preds,
        "group": group
    }).reset_index(drop=True)

    return error


