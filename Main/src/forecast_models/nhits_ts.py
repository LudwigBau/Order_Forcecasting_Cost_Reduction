# Setup

# Libraries
import numpy as np
import pandas as pd
import pickle

from darts import TimeSeries
from darts.models import NHiTSModel

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler

from darts.dataprocessing.transformers import Scaler
from darts.timeseries import TimeSeries



# Get custom utils functions
from src.utils.data_split import ml_data_date_split

# Fix Pytorch Error
import os
import logging
import warnings
# Environment set ups
warnings.filterwarnings('ignore')
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Set seed
np.random.seed(42)

# Horizon
time_horizon = 9
n_steps_in = 30
n_steps_out = time_horizon

# TimeSeriesSplit
tscv = TimeSeriesSplit(gap=0, max_train_size=None, n_splits=5, test_size=time_horizon)


# LSTM Forecast
def nhits_forecast(train_data):

    # Set up data
    train_series = TimeSeries.from_dataframe(train_data, "date", "quantity")

    # Scale the time series (Scaler from Darts is used)
    scaler = Scaler()
    train_series_scaled = scaler.fit_transform(train_series)

    # Define Model
    model = NHiTSModel(
        input_chunk_length=30,
        output_chunk_length=time_horizon,
        num_stacks=10,
        num_blocks=1,
        num_layers=2,
        layer_widths=512,
        n_epochs=100,
        nr_epochs_val_period=10,
        batch_size=32,)

    # Fit the model on the scaled time series
    model.fit(train_series_scaled)

    # Forecast
    yhat_scaled = model.predict(n=time_horizon)

    # Rescale the predictions to the original scale
    yhat = scaler.inverse_transform(yhat_scaled)

    # Return the rescaled predictions as a 1-dimensional NumPy array
    return yhat.values().flatten()


def nhits_backtesting(df, tscv):
    # Initialize an empty lists to store predictions, actuals and dates
    preds = []
    actuals = []
    dates = []

    # loop over pre-defined time series split / time series cross validation (tscv)
    for i, (train_index, test_index) in enumerate(tscv.split(df)):
        # Train and test data
        train_fold, test_fold = df.iloc[train_index], df.iloc[test_index]

        # Append Dates
        dates_fold = df.date.iloc[test_index].tolist()

        # Track
        print(f"Fold Nr. = {i}")

        # Predict on the current test fold
        y_pred_fold = nhits_forecast(train_fold)

        # append scores and predictions
        preds.append(y_pred_fold.tolist())
        actuals.append(test_fold.quantity.tolist())
        dates.append(dates_fold)

    # make df that holds predictions and actuals
    # flatten nested lists
    dates = np.concatenate(dates).tolist()
    actuals = np.concatenate(actuals).tolist()
    preds = np.concatenate(preds).tolist()

    # Error analysis
    error = pd.DataFrame({
        "date": dates,
        "actual": actuals,
        "pred": preds
    }).reset_index(drop=True)

    return error


if __name__ == "__main__":

    df_list = ["L_3", "L_4", "L_6"]
    group_list = ["warehouse_chain", "new_customer_id", "emtpy"]


    # Load data
    test_data = {}
    for i in df_list:
        test_data[i] = pd.read_pickle(f"../../data/processed/{i}_test.pkl")

    # Initialize an empty dictionary to store the results
    results = {}

    # lightGBM modelling
    for level, group in zip(df_list, group_list):

        # Initialize DataFrames to store the backtest and forecasted values
        backtest_values = pd.DataFrame(columns=["date", "actual", "pred", "level", "group"])
        forecast_values = pd.DataFrame(columns=["date", "actual", "pred", "level", "group"])

        # Select the right level
        df = test_data[level].copy()

        # Select the right group

        if level == "L_6":
            print("Group:", "L_6")

            # selected group
            group_df = df.copy()

            # Define Data
            train_df, test_df = ml_data_date_split(group_df, 8)  # split data with custom function

            print("Start backtest:")
            # Backtest
            temp_backtest_df = nhits_backtesting(train_df, tscv)
            temp_backtest_df["level"] = level
            temp_backtest_df["group"] = "empty"

            backtest_values = pd.concat([backtest_values, temp_backtest_df])

            print("Start forecast:")
            y_pred = nhits_forecast(train_df)
            print(y_pred.shape)
            y_pred = nhits_forecast(train_df).flatten() #new

            temp_forecast_df = pd.DataFrame({'date': test_df.date,
                                             'actual': test_df.quantity,
                                             'pred': y_pred})
            temp_forecast_df["level"] = level
            temp_forecast_df["group"] = "empty"

            forecast_values = pd.concat([forecast_values, temp_forecast_df])

        else:

            for i_ts in df[group].unique():
                print("Group:", i_ts)

                # seleced group
                group_df = df[df[group] == i_ts]

                # Define Data
                train_df, test_df = ml_data_date_split(group_df, 8)  # split data with custom function

                print("Start backtest:")
                # Backtest
                temp_backtest_df = nhits_backtesting(train_df, tscv)
                temp_backtest_df["level"] = level
                temp_backtest_df["group"] = i_ts

                backtest_values = pd.concat([backtest_values, temp_backtest_df])

                print("Start forecast:")
                y_pred = nhits_forecast(train_df)
                print(y_pred.shape)
                y_pred = nhits_forecast(train_df).flatten()  # new

                temp_forecast_df = pd.DataFrame({'date': test_df.date,
                                                 'actual': test_df.quantity,
                                                 'pred': y_pred})
                temp_forecast_df["level"] = level
                temp_forecast_df["group"] = i_ts

                forecast_values = pd.concat([forecast_values, temp_forecast_df])

                print(backtest_values)
                print(forecast_values)
        # Store the backtest and prediction data frames in the results dictionary
        # with the corresponding name including the category_name
        results[f"{level}"] = {
            'backtest': backtest_values,
            'pred': forecast_values
        }

    # Save dictionaries as pickle files
    with open('../../data/modelling_results/nhits_results.pickle', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
