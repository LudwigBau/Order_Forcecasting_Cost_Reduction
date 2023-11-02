# Libraries
import pandas as pd
import numpy as np
import pickle

import pmdarima as pm

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from sklearn.model_selection import TimeSeriesSplit


# backtest
def ts_backtest(train_data, tscv):
    # Initialize DataFrames to store the backtest and forecasted values
    backtest_values = pd.DataFrame(columns=['sarima', 'sarimax', 'ets', 'naive_drift', 'naive_seasonal'])

    # Loop through the time series splits
    for i, (train_index, test_index) in enumerate(tscv.split(train_data)):
        train, test = train_data.iloc[train_index], train_data.iloc[test_index]
        sales_train, sales_test = train['quantity'], test['quantity']
        exog_train, exog_test = train.drop('quantity', axis=1), test.drop('quantity', axis=1)

        print(f"Split {i + 1}")

        # Auto ARIMA
        auto_arima_model = pm.auto_arima(sales_train, seasonal=True, m=7,
                                         suppress_warnings=True,
                                         stepwise=False,
                                         n_jobs=-1)
        # SARIMA
        sarima_model = SARIMAX(sales_train,
                               order=auto_arima_model.order,
                               seasonal_order=auto_arima_model.seasonal_order)
        sarima_fit = sarima_model.fit(disp=0)
        sarima_forecast = sarima_fit.forecast(steps=len(test))

        # SARIMAX
        sarimax_model = SARIMAX(sales_train, exog=exog_train,
                                order=auto_arima_model.order,
                                seasonal_order=auto_arima_model.seasonal_order)
        sarimax_fit = sarimax_model.fit(disp=0)
        sarimax_forecast = sarimax_fit.forecast(steps=len(test), exog=exog_test)

        # Auto ETS
        ets_model = ExponentialSmoothing(sales_train, seasonal_periods=7,
                                         trend='add',
                                         seasonal='add',
                                         damped_trend=True)
        auto_ets_model = ets_model.fit()
        ets_forecast = auto_ets_model.forecast(len(test))

        # NaiveDrift
        naive_drift_forecast = sales_train.iloc[-1] + np.cumsum(sales_train.diff().mean()) * np.arange(1, len(test) + 1)

        # NaiveSeasonal
        naive_seasonal_model = SARIMAX(sales_train, order=(0, 0, 0), seasonal_order=(0, 1, 0, 7))
        naive_seasonal_fit = naive_seasonal_model.fit(disp=0)
        naive_seasonal_forecast = naive_seasonal_fit.forecast(steps=len(test))

        # Save backtest values
        temp_df = pd.DataFrame({'sarima': sarima_forecast,
                                'sarimax': sarimax_forecast,
                                'ets': ets_forecast,
                                'naive_drift': naive_drift_forecast,
                                'naive_seasonal': naive_seasonal_forecast}, index=test.index)

        backtest_values = pd.concat([backtest_values, temp_df])

    return backtest_values


# Forecast
def ts_forecast(train_data, test_data):
    # data
    sales_train = train_data["quantity"]
    exog_train = train_data.drop('quantity', axis=1)
    exog_test = test_data.drop('quantity', axis=1)

    # Auto ARIMA (get parameters)
    auto_arima_model = pm.auto_arima(sales_train, seasonal=True, m=7,
                                     suppress_warnings=True,
                                     stepwise=False,
                                     n_jobs=-1)
    print(f"auto_arima Parameters: {auto_arima_model.order}")
    print(f"auto_arima Seasonal Parameters: {auto_arima_model.seasonal_order}")

    # Train Models
    # SARIMA
    sarima_model = SARIMAX(sales_train,
                           order=auto_arima_model.order,
                           seasonal_order=auto_arima_model.seasonal_order)
    sarima_fit = sarima_model.fit(disp=0)

    # SARIMAX
    sarimax_model = SARIMAX(sales_train, exog=exog_train,
                            order=auto_arima_model.order,
                            seasonal_order=auto_arima_model.seasonal_order)
    sarimax_fit = sarimax_model.fit(disp=0)

    # Auto ETS
    ets_model = ExponentialSmoothing(sales_train, seasonal_periods=7,
                                     trend='add',
                                     seasonal='add',
                                     damped_trend=True)
    auto_ets_model = ets_model.fit()

    # NaiveDrift
    naive_drift_forecast = sales_train.iloc[-1] + np.cumsum(sales_train.diff().mean()) * np.arange(1, 10)

    # NaiveSeasonal
    naive_seasonal_model = SARIMAX(sales_train, order=(0, 0, 0), seasonal_order=(0, 1, 0, 7))
    naive_seasonal_fit = naive_seasonal_model.fit(disp=0)
    naive_seasonal_forecast = naive_seasonal_fit.forecast(steps=9)

    # Forecast the next 9 days
    sarima_forecast = sarima_fit.forecast(steps=9)
    sarimax_forecast = sarimax_fit.forecast(steps=9, exog=exog_test)
    ets_forecast = auto_ets_model.forecast(9)

    # Save forecasted values
    forecast_values = pd.DataFrame({'sarima': sarima_forecast,
                                    'sarimax': sarimax_forecast,
                                    'ets': ets_forecast,
                                    'naive_drift': naive_drift_forecast,
                                    'naive_seasonal': naive_seasonal_forecast},
                                   index=pd.date_range(train_data.index[-1], periods=10, inclusive='right'))

    return forecast_values

if __name__ == "__main__":
    # define tscv
    tscv = TimeSeriesSplit(gap=0, max_train_size=None, n_splits=5, test_size=9)

    # List Datasets
    df_list = ["L_3", "L_6", "L_4"]
    #df_list = ["L_3", "L_4", "L_6"]
    group_list = ["warehouse_chain", "empty","new_customer_id"]
    #group_list = ["warehouse_chain", "new_customer_id", "empty"]

    # Load data
    test_data_dict = {}
    for i in df_list:
        test_data_dict[i] = pd.read_pickle(f"../../data/processed/{i}_test.pkl").set_index("date")

    holiday = pd.read_csv("../../data/intermediate/calender.csv")
    holiday['date'] = pd.to_datetime(holiday['date'])

    # Start Modelling
    ts_results = {}
    for level, group in zip(df_list, group_list):

        # setup
        # Initialize DataFrames to store the backtest and forecasted values
        backtest_values = pd.DataFrame(columns=['sarima', 'sarimax', 'ets',
                                                'naive_drift', 'naive_seasonal',
                                                "level", "group"])

        forecast_values = pd.DataFrame(columns=['sarima', 'sarimax', 'ets',
                                                'naive_drift', 'naive_seasonal',
                                                "level", "group"])

        # select level
        df = test_data_dict[level]

        if level == "L_6":

            # test
            test_df = df.copy()
            test_series = test_df["quantity"]["2022-11-27":]
            test_data = test_series.reset_index().merge(holiday,
                                                        left_on="date", right_on="date", how="left").set_index("date")

            # train
            train_df = test_df[:"2022-11-26"]
            train_series = train_df['quantity']
            train_data = train_series.reset_index().merge(holiday,
                                                          left_on="date", right_on="date", how="left").set_index("date")

            # Define exogene data
            exog_train = train_data.drop('quantity', axis=1)
            exog_test = test_data.drop('quantity', axis=1)

            # Define Fequency
            train_data.index.freq = 'D'
            test_data.index.freq = 'D'

            print("start backtest")
            temp_backtest_df = ts_backtest(train_data, tscv)
            temp_backtest_df["level"] = level
            temp_backtest_df["group"] = "none"

            backtest_values = pd.concat([backtest_values, temp_backtest_df])

            print("start forecast")
            temp_forecast_df = ts_forecast(train_data, test_data)
            temp_forecast_df["level"] = level
            temp_forecast_df["group"] = "none"

            forecast_values = pd.concat([forecast_values, temp_forecast_df])

        else:

            for i_ts in df[group].unique():
                print(i_ts)

                # test
                test_df = df[df[group] == i_ts]
                test_series = test_df["quantity"]["2022-11-27":]
                test_data = test_series.reset_index().merge(holiday,
                                                            left_on="date", right_on="date", how="left").set_index("date")

                # train
                train_df = test_df[:"2022-11-26"]
                train_series = train_df['quantity']
                train_data = train_series.reset_index().merge(holiday,
                                                              left_on="date", right_on="date", how="left").set_index("date")

                # Define exogene data
                exog_train = train_data.drop('quantity', axis=1)
                exog_test = test_data.drop('quantity', axis=1)

                # Define Fequency
                train_data.index.freq = 'D'
                test_data.index.freq = 'D'

                print("start backtest")
                temp_backtest_df = ts_backtest(train_data, tscv)
                temp_backtest_df["level"] = level
                temp_backtest_df["group"] = i_ts

                backtest_values = pd.concat([backtest_values, temp_backtest_df])

                print("start forecast")
                temp_forecast_df = ts_forecast(train_data, test_data)
                temp_forecast_df["level"] = level
                temp_forecast_df["group"] = i_ts

                forecast_values = pd.concat([forecast_values, temp_forecast_df])

        # save forecast and backtest_values in dict
        ts_results[level] = {
            'backtest': backtest_values,
            'pred': forecast_values
        }

        print(ts_results[level])

    print(ts_results.keys())
    print(ts_results)
    # Save dictionaries as pickle files
    with open('../../data/modelling_results/ts_results_all.pickle', 'wb') as handle:
        pickle.dump(ts_results, handle, protocol=pickle.HIGHEST_PROTOCOL)



