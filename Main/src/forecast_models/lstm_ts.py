# NOTE: Script to run LSTM over selection of df's

# Libraries
import pandas as pd
import numpy as np
import pickle

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler

from src.utils.data_split import ml_data_date_split

import warnings
warnings.filterwarnings('ignore')

# Set Modelling Parameters

# Set seed
np.random.seed(42)

# Horizon
time_horizon = 9
n_steps_in = 30
n_steps_out = time_horizon
n_features = 1

# TimeSeriesSplit
tscv = TimeSeriesSplit(gap=0, max_train_size=None, n_splits=5, test_size=time_horizon)

# Model parameters
batch_size = 32
epochs = 150


# Custom Functions

# Split a univariate sequence into samples
def split_sequences(sequence, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequence)):
        # Find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # Check if we are beyond the sequence
        if out_end_ix > len(sequence):
            break
        # Gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


# LSTM Forecast
def LSTM_forecast(train_data, n_steps_in, n_steps_out, epochs, batch_size):
    # Set up data
    # Get quantity column as np array
    sequence = train_data['quantity'].values

    # Scale the sequence
    scaler = MinMaxScaler(feature_range=(0, 1))
    sequence = scaler.fit_transform(sequence.reshape(-1, 1))

    # Split into input/output
    X, y = split_sequences(sequence, n_steps_in, n_steps_out)

    # Reshape X to fit the LSTM input shape (samples, timesteps, features)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Define Model
    model = Sequential()
    model.add(LSTM(200, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)))
    model.add(LSTM(100, activation='relu'))
    model.add(Dense(n_steps_out))
    model.compile(optimizer='adam', loss='mse')

    # Train
    # Fit model
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)

    # Forecast
    # Get last n_steps_in quantities
    x_input = sequence[-n_steps_in:]
    x_input = x_input.reshape((1, n_steps_in, 1))

    # Predict
    yhat = model.predict(x_input, verbose=0)
    # Flatten yhat
    yhat_1d = yhat.flatten()

    # Rescale the predictions
    yhat_rescaled = scaler.inverse_transform(yhat_1d.reshape(-1, 1))

    return yhat_rescaled.flatten()


def LSTM_backtesting(df, tscv):
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
        y_pred_fold = LSTM_forecast(train_fold, n_steps_in, n_steps_out, epochs, batch_size)

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

    # define DFs and groups

    df_list = ["L_1", "L_2", "L_3"]
    group_list = ["new_customer_id", "warehouse_chain", "empty"]


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

        if level == "L_3":

            print("Group:", "L_3")

            # selected group
            group_df = df.copy()

            # Define Data
            train_df, test_df = ml_data_date_split(group_df, 8)  # split data with custom function

            print("Start backtest:")
            # Backtest
            temp_backtest_df = LSTM_backtesting(train_df, tscv)
            temp_backtest_df["level"] = level
            temp_backtest_df["group"] = "empty"

            backtest_values = pd.concat([backtest_values, temp_backtest_df])

            print("Start forecast:")
            y_pred = LSTM_forecast(train_df, n_steps_in, n_steps_out, epochs, batch_size)

            temp_forecast_df = pd.DataFrame({'date': test_df.date,
                                             'actual': test_df.quantity,
                                             'pred': y_pred})
            temp_forecast_df["level"] = level
            temp_forecast_df["group"] = "empty"

            forecast_values = pd.concat([forecast_values, temp_forecast_df])

        else:

            for i_ts in df[group].unique():
                print("Group:", i_ts)

                # selected group
                group_df = df[df[group] == i_ts]

                # Define Data
                train_df, test_df = ml_data_date_split(group_df, 8)  # split data with custom function

                print("Start backtest:")
                # Backtest
                temp_backtest_df = LSTM_backtesting(train_df, tscv)
                temp_backtest_df["level"] = level
                temp_backtest_df["group"] = i_ts

                backtest_values = pd.concat([backtest_values, temp_backtest_df])

                print("Start forecast:")
                y_pred = LSTM_forecast(train_df, n_steps_in, n_steps_out, epochs, batch_size)

                temp_forecast_df = pd.DataFrame({'date': test_df.date,
                                                 'actual': test_df.quantity,
                                                 'pred': y_pred})
                temp_forecast_df["level"] = level
                temp_forecast_df["group"] = i_ts

                forecast_values = pd.concat([forecast_values, temp_forecast_df])

        # Store the backtest and prediction data frames in the results dictionary
        # with the corresponding name including the category_name
        results[f"{level}"] = {
            'backtest': backtest_values,
            'pred': forecast_values
        }

    # Save dictionaries as pickle files
    with open('../../data/modelling_results/lstm_results.pickle', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


