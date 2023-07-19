# library
import pandas as pd
import numpy as np
import pickle

from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae

from itertools import combinations

#data
L_3 = pd.read_pickle("../../data/processed/L_3_test.pkl")

# time series
ts_results = pd.read_pickle("../../data/modelling_results/ts_results_all.pickle")

lgbm_results = pd.read_pickle("../../data/modelling_results/lgbm_results.pickle")
lgbm_params = pd.read_pickle("../../data/modelling_results/lgbm_params.pickle")
lgbm_importance = pd.read_pickle("../../data/modelling_results/lgbm_importance.pickle")

# xgb
xgb_results = pd.read_pickle("../../data/modelling_results/xgb_results.pickle")
xgb_params = pd.read_pickle("../../data/modelling_results/xgb_params.pickle")
xgb_importance = pd.read_pickle("../../data/modelling_results/xgb_importance.pickle")

# LSTM
lstm_results = pd.read_pickle("../../data/modelling_results/lstm_results.pickle")

# nhits
nhits_results = pd.read_pickle("../../data/modelling_results/nhits_results.pickle")


# Custom Functions
def back_pred_df(ts_results, lgbm_results, xgb_results, lstm_results, nhits_results):
    # Lists to hold 'pred' series
    backtest_list = []
    pred_list = []

    # ts
    for key in ts_results.keys():
        for column in ts_model_names:
            backtest_series = ts_results[key]['backtest'].reset_index().groupby(["index"]).sum()[column]
            pred_series = ts_results[key]['pred'].reset_index().groupby(["index"]).sum()[column]

            # Rename the series to the current key and column
            backtest_series.name = key + '_' + column
            pred_series.name = key + '_' + column

            # Add the series to the list
            backtest_list.append(backtest_series)
            pred_list.append(pred_series)

    # lgbm
    for key in lgbm_results.keys():
        backtest_series = lgbm_results[key]['backtest'].groupby(["date"]).sum().pred
        pred_series = lgbm_results[key]["pred"].groupby(["date"]).sum().pred

        # Rename the series to the current key
        backtest_series.name = key + "_lgbm"
        pred_series.name = key + "_lgbm"

        # Add the series to the list
        backtest_list.append(backtest_series)
        pred_list.append(pred_series)

    # xgb
    for key in xgb_results.keys():
        backtest_series = xgb_results[key]['backtest'].groupby(["date"]).sum().pred
        pred_series = xgb_results[key]["pred"].groupby(["date"]).sum().pred

        # Rename the series to the current key
        backtest_series.name = key + "_xgb"
        pred_series.name = key + "_xgb"

        # Add the series to the list
        backtest_list.append(backtest_series)
        pred_list.append(pred_series)

    # lstm

    for key in lstm_results.keys():
        backtest_series = lstm_results[key]['backtest'].groupby(["date"]).sum().pred
        pred_series = lstm_results[key]["pred"].groupby(["date"]).sum().pred

        # Rename the series to the current key
        backtest_series.name = key + "_lstm"
        pred_series.name = key + "_lstm"

        # Add the series to the list
        backtest_list.append(backtest_series)
        pred_list.append(pred_series)

    # nhits
    for key in lstm_results.keys():
        backtest_series = nhits_results[key]['backtest'].groupby(["date"]).sum().pred
        pred_series = nhits_results[key]["pred"].groupby(["date"]).sum().pred

        # Rename the series to the current key
        backtest_series.name = key + "_nhits"
        pred_series.name = key + "_nhits"

        # Add the series to the list
        backtest_list.append(backtest_series)
        pred_list.append(pred_series)

    # Concatenate all series into one dataframe
    backtest_df = pd.concat(backtest_list, axis=1)
    pred_df = pd.concat(pred_list, axis=1)

    # attach actuals
    # Access actual values for backtest and prediction
    backtest_df["actual"] = lgbm_results["L_3_Time_Momentum_Lag"]["backtest"].groupby(["date"]).sum().actual
    pred_df["actual"] = lgbm_results["L_3_Time_Momentum_Lag"]["pred"].groupby(["date"]).sum().actual

    # Set index names
    backtest_df.index.name = 'date'
    pred_df.index.name = 'date'

    return backtest_df, pred_df


def create_ensemble(backtest_df, pred_df, list_of_selected_models):
    # Lists to hold 'pred' series
    backtest_list = []
    pred_list = []

    # create list of combinations
    model_pairs = list(combinations(list_of_selected_models, 2))

    for pair in model_pairs:
        # create ensembeles
        backtest_ensemble = (backtest_df[pair[0]] + backtest_df[pair[1]]) / 2
        pred_ensemble = (pred_df[pair[0]] + pred_df[pair[1]]) / 2

        # Rename the series to the current key
        backtest_ensemble.name = pair[0] + "_" + pair[1]
        pred_ensemble.name = pair[0] + "_" + pair[1]

        # Add the series to the list
        backtest_list.append(backtest_ensemble)
        pred_list.append(pred_ensemble)

    back_ens_df = pd.concat(backtest_list, axis=1)
    pred_ens_df = pd.concat(pred_list, axis=1)

    # Set index names
    back_ens_df.index.name = 'date'
    pred_ens_df.index.name = 'date'

    return back_ens_df, pred_ens_df


def ensemble_metrics(backtest_df, pred_df):
    # Initialize an empty list to store the results
    metrics_data = []

    # Iterate over the dataframes and categories

    for category_name in backtest_df.columns:
        # Access backtest data frame for the current level and category
        backtest = backtest_df[category_name]
        # Access prediction data frame for the current level and category
        pred = pred_df[category_name]

        # Calculate RMSE
        val_rmse = np.sqrt(mse(backtest_df.actual, backtest))
        pred_rmse = np.sqrt(mse(pred_df.actual, pred))

        # Calculate MAPE
        val_mape = mape(backtest_df.actual, backtest)
        pred_mape = mape(pred_df.actual, pred)

        # Calculate MAE
        val_mae = mae(backtest_df.actual, backtest)
        pred_mae = mae(pred_df.actual, pred)

        # Append the results to the metrics_data list
        metrics_data.append({
            'category': category_name,
            'val_rmse': val_rmse,
            'pred_rmse': pred_rmse,
            'val_mape': val_mape,
            'pred_mape': pred_mape,
            # 'val_mae': val_mae,
            # 'pred_mae': pred_mae
        })

        ensemble_metrics = pd.DataFrame(metrics_data)

    return ensemble_metrics

if __name__ == "__main__":

    # List TS Models
    ts_model_names = ts_results["L_3"]["pred"].columns[0:5]

    # Run Functions and get individual metrics (without ensembles)
    back_df, pred_df = back_pred_df(ts_results, lgbm_results, xgb_results, lstm_results, nhits_results)
    ind_metrics_df = ensemble_metrics(back_df, pred_df).sort_values("val_mape")

    # Get nest Models to create ensembles (to limit compexity)
    list_of_selected_models = list(ind_metrics_df[1:11].category)

    # Create ensemble and save as df
    back_ens_df, pred_ens_df = create_ensemble(back_df, pred_df, list_of_selected_models)

    # Backtest and prediction values
    full_back_df = back_df.merge(back_ens_df,  left_index=True, right_index=True, how="left")
    full_pred_df = pred_df.merge(pred_ens_df,  left_index=True, right_index=True, how="left")

    # Metrics
    ensemble_metrics_df = ensemble_metrics(full_back_df, full_pred_df).sort_values("pred_rmse")

    # Save
    with open('../../data/modelling_results/ens_back_results_v2.pickle', 'wb') as handle:
        pickle.dump(full_back_df, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('../../data/modelling_results/ens_pred_results_v2.pickle', 'wb') as handle:
        pickle.dump(full_pred_df, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # save to excel
    ensemble_metrics_df.to_excel("../../data/modelling_results/ensemble_metrics_v2.xlsx")
    ind_metrics_df.to_excel("../../data/modelling_results/ind_metrics_v2.xlsx")

