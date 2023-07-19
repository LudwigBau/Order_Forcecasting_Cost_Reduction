# Library
import pandas as pd
import pickle

import gc

from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV

from sklearn.inspection import permutation_importance

import lightgbm as lgb
import xgboost as xgb

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")
warnings.simplefilter("ignore", category=RuntimeWarning)

# Get custom utils functions
from src.utils.data_split import ml_data_date_split
from src.utils.model_utils import custom_backtesting


if __name__ == "__main__":

    # set fixed variables
    seed = 42
    time_horizon = 9
    n_iter = 200

    # custom functions
    def get_test_size(df, time_horizon=time_horizon):
        nr_group = df.group.nunique()
        test_size = nr_group * time_horizon
        return test_size


    # Load Data

    # Only load warehouse_chain, customers and total to mimimize computational time
    df_list = ["L_3", "L_4", "L_6"]
    group_list = ["warehouse_chain", "new_customer_id", "empty"]


    # Load data
    test_data = {}
    for i in df_list:
        test_data[i] = pd.read_pickle(f"../../data/processed/{i}_test.pkl")

    # Change group name
    print("Change group name")
    for df_name, group_name in zip(df_list, group_list):
        if group_name != "empty":
            test_data[df_name] = test_data[df_name].rename(columns={group_name: 'group'})
        else:
            test_data[df_name]["group"] = "empty"

    def make_category_df(df_list, test_data):

        # Get all features
        all_features = set()

        for df in df_list:
            all_features.update(test_data[df].columns)

        # Define features for each category
        date_and_id = [col for col in all_features if col == 'date' or col == 'new_customer_id' or col == 'new_product_id'
                       or col == "state" or col == "warehouse_chain" or col == "group"]
        target = [col for col in all_features if col == 'quantity']
        indicators = [col for col in all_features if '_id_' in col or '_decile' in col]
        time = [col for col in all_features if 'tm_' in col]
        momentum_lag = [col for col in all_features if '_roll_' in col or '_lag_' in col]
        price = [col for col in all_features if 'price' in col or 'discount' in col]
        weather = ['precipitation_height', 'sunshine_duration', 'temperature_air_mean_200', 'sunshine_duration_h',
                   'suns_classes', 'temp_classes', 'rain_classes']
        holiday = [col for col in all_features if 'holiday' in col or 'event' in col or 'week' in col]

        # basic = features I always need
        basic = date_and_id + target + indicators + time + price

        # make catagories of features
        Time_Momentum_Lag = basic + momentum_lag
        Time_Momentum_Lag_Weather = Time_Momentum_Lag + weather
        Time_Momentum_Lag_Holiday = Time_Momentum_Lag + holiday
        Time_Momentum_Lag_Weather_Holiday = Time_Momentum_Lag + weather + holiday

        # Create a dictionary with category names as keys and lists of features as values
        categories_dict = {
            "Time_Momentum_Lag": Time_Momentum_Lag,
            "Time_Momentum_Lag_Weather": Time_Momentum_Lag_Weather,
            "Time_Momentum_Lag_Holiday": Time_Momentum_Lag_Holiday,
            "Time_Momentum_Lag_Weather_Holiday": Time_Momentum_Lag_Weather_Holiday
        }

        return categories_dict

    category_dict = make_category_df(df_list, test_data)

    # Get the names of the categories
    category_names = list(category_dict.keys())

    print("Selected categories:", category_names)

    # LIGHT GBM
    print("START LIGHT GBM")

    # Define model

    # Define the model
    regressor = lgb.LGBMRegressor(n_jobs=-1)

    # Parameter Grid for Randomized Search
    param_dist = {
        "objective": ["poisson", "tweedie"],
        "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
        'num_leaves': [31, 50, 64, 100, 128, 200],
        'max_depth': [-1, 5, 10, 15, 20],
        'subsample': [0.5, 0.7, 0.85],
        'subsample_freq': [0, 5, 10],
        "colsample_bytree": [0.5, 0.7, 0.9],
        "lambda_l1": [0, 0.1, 0.5],
        "lambda_l2": [0, 0.1, 0.5],
        "verbose": [-1],
        "metric": ["rmse"]
    }

    # Initialize an empty dictionary to store the results
    results = {}
    df_params = {}
    feature_importance = {}

    # lightGBM modelling
    for level, group in zip(df_list, group_list):
        for category_name, feature_list in category_dict.items():

            # Use features from the current category
            if level == "L_6":  # no group for L_6, tscv dependent

                # Define Data
                train_df, test_df = ml_data_date_split(test_data[level], 8)  # split data with custom function

                # Sort values (important for time series split)
                train_sorted_df = train_df.sort_values(by=['date'])
                test_sorted_df = test_df.sort_values(by=['date'])
                # train
                x_train = train_sorted_df.copy()
                y_train = train_sorted_df['quantity'].copy()  # get y / target variable
                x_train.drop(["quantity", 'date', "group"], axis=1,
                             inplace=True)  # clean data: drop target variable, date and group from dataset
                # test
                x_test = test_sorted_df.copy()
                y_test = test_sorted_df['quantity'].copy()  # get y / target variable
                x_test.drop(["quantity", 'date', "group"], axis=1,
                            inplace=True)  # clean data: drop target variable, date and group from dataset

                # Use only matching features from the current category
                matching_features = [feature for feature in feature_list if feature in x_train.columns]
                x_train = x_train[matching_features].copy()
                x_test = x_test[matching_features].copy()

                # time series split
                tscv = TimeSeriesSplit(gap=0,
                                       max_train_size=None,
                                       n_splits=5,
                                       test_size=time_horizon)

            else:

                # Define Data
                train_df, test_df = ml_data_date_split(test_data[level], 8)  # split data with custom function
                # Sort values (important for time series split)
                train_sorted_df = train_df.sort_values(by=['date', "group"])
                test_sorted_df = test_df.sort_values(by=['date', "group"])
                # train
                x_train = train_sorted_df.copy()
                y_train = train_sorted_df['quantity'].copy()  # get y / target variable
                x_train.drop(["quantity", 'date', "group"], axis=1,
                             inplace=True)  # clean data: drop target variable, date and group from dataset
                # test
                x_test = test_sorted_df.copy()
                y_test = test_sorted_df['quantity'].copy()  # get y / target variable
                x_test.drop(["quantity", 'date', "group"], axis=1,
                            inplace=True)  # clean data: drop target variable, date and group from dataset

                # Use only matching features from the current category
                matching_features = [feature for feature in feature_list if feature in x_train.columns]
                x_train = x_train[matching_features].copy()
                x_test = x_test[matching_features].copy()

                # time series split
                tscv = TimeSeriesSplit(gap=0,
                                       max_train_size=None,
                                       n_splits=5,
                                       test_size=get_test_size(train_sorted_df,
                                                               time_horizon=time_horizon))

            # Randomized Search with time series-validation
            random_search = RandomizedSearchCV(
                regressor,
                param_distributions=param_dist,
                n_iter=n_iter,  # Number of parameter settings that are sampled
                scoring='neg_root_mean_squared_error',  # 'neg_mean_absolute_error' equivilant to RMSE
                cv=tscv,  # Use TimeSeriesSplit (Cross validation for time series)
                random_state=seed,  # Set a random state for reproducibility
                n_jobs=6  # set n_jobs
            )

            # get parameters (model and random_search is defined outside the loop)
            random_search.fit(x_train, y_train)
            best_params = random_search.best_params_

            print(f"Best params for {level}_{category_name} level: {best_params}")  # log

            # Save best_params
            df_params[f"{level}_{category_name}"] = best_params

            # backtest
            lgbm_model = lgb.LGBMRegressor(**best_params)
            backtest_df = custom_backtesting(train_sorted_df, x_train, y_train, lgbm_model, tscv)

            # forecast
            lgbm_model = lgb.LGBMRegressor(**best_params)
            lgbm_model.fit(x_train, y_train)  # train model on full train set

            # Predict on the current test fold
            y_pred = lgbm_model.predict(x_test)  # use test data to test

            # save df
            pred_df = pd.DataFrame({
                "date": test_sorted_df.date,
                "actual": test_sorted_df.quantity,
                "pred": y_pred,
                "group": test_sorted_df.group
            }).reset_index(drop=True)

            # Store the backtest and prediction data frames in the results dictionary
            # with the corresponding name including the category_name
            results[f"{level}_{category_name}"] = {
                'backtest': backtest_df,
                'pred': pred_df
            }

            # Calculate permutation importance
            p_importance = permutation_importance(lgbm_model, x_train, y_train, n_repeats=10, random_state=seed,
                                                  n_jobs=6)

            df_permutation_importance = pd.DataFrame({'feature_names': x_train.columns,
                                                      'importances_mean': p_importance.importances_mean,
                                                      'importances_std': p_importance.importances_std}).sort_values(
                by='importances_mean', ascending=False)

            feature_importance[f"{level}_{category_name}"] = {
                "df": df_permutation_importance,
                "box_array": p_importance.importances.T
            }

            print(f"10 most important features for {level}_{category_name} "f"level: {df_permutation_importance.head(10).feature_names.tolist()}")  # log

            # Force garbage collection
            gc.collect()

    # Save dictionaries as pickle files
    with open('../../data/modelling_results/lgbm_results.pickle', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('../../data/modelling_results/lgbm_params.pickle', 'wb') as handle:
        pickle.dump(df_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('../../data/modelling_results/lgbm_importance.pickle', 'wb') as handle:
        pickle.dump(feature_importance, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Extreme Gradien Boosting
    print("START XGBM")

    # Define the model
    xgb_regressor = xgb.XGBRegressor(n_jobs=-1)

    # Parameter Grid for Randomized Search
    xgb_param_dist = {
        "objective": ["reg:squarederror"],  # 'reg:squarederror' is for regression problems
        "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
        'max_depth': [5, 10, 15, 20],
        'subsample': [0.5, 0.7, 0.85],
        "colsample_bytree": [0.5, 0.7, 0.9],
        "lambda": [0, 0.1, 0.5],  # L2 regularization
        "alpha": [0, 0.1, 0.5],  # L1 regularization
    }

    # Initialize an empty dictionary to store the results
    results = {}
    df_params = {}
    feature_importance = {}

    # lightGBM modelling
    for level, group in zip(df_list, group_list):
        for category_name, feature_list in category_dict.items():

            # Use features from the current category
            if level == "L_6":  # No group for top level

                # Define Data
                train_df, test_df = ml_data_date_split(test_data[level], 8)  # split data with custom function

                # Sort values (important for time series split)
                train_sorted_df = train_df.sort_values(by=['date'])
                test_sorted_df = test_df.sort_values(by=['date'])
                # train
                x_train = train_sorted_df.copy()
                y_train = train_sorted_df['quantity'].copy()  # get y / target variable
                x_train.drop(["quantity", 'date', "group"], axis=1,
                             inplace=True)  # clean data: drop target variable, date and group from dataset
                # test
                x_test = test_sorted_df.copy()
                y_test = test_sorted_df['quantity'].copy()  # get y / target variable
                x_test.drop(["quantity", 'date', "group"], axis=1,
                            inplace=True)  # clean data: drop target variable, date and group from dataset

                # Use only matching features from the current category
                matching_features = [feature for feature in feature_list if feature in x_train.columns]
                x_train = x_train[matching_features].copy()
                x_test = x_test[matching_features].copy()

                # time series split
                tscv = TimeSeriesSplit(gap=0,
                                       max_train_size=None,
                                       n_splits=5,
                                       test_size=time_horizon)

            else:

                # Define Data
                train_df, test_df = ml_data_date_split(test_data[level], 8)  # split data with custom function
                # Sort values (important for time series split)
                train_sorted_df = train_df.sort_values(by=['date', "group"])
                test_sorted_df = test_df.sort_values(by=['date', "group"])
                # train
                x_train = train_sorted_df.copy()
                y_train = train_sorted_df['quantity'].copy()  # get y / target variable
                x_train.drop(["quantity", 'date', "group"], axis=1,
                             inplace=True)  # clean data: drop target variable, date and group from dataset
                # test
                x_test = test_sorted_df.copy()
                y_test = test_sorted_df['quantity'].copy()  # get y / target variable
                x_test.drop(["quantity", 'date', "group"], axis=1,
                            inplace=True)  # clean data: drop target variable, date and group from dataset

                # Use only matching features from the current category
                matching_features = [feature for feature in feature_list if feature in x_train.columns]
                x_train = x_train[matching_features].copy()
                x_test = x_test[matching_features].copy()

                # time series split
                tscv = TimeSeriesSplit(gap=0,
                                       max_train_size=None,
                                       n_splits=5,
                                       test_size=get_test_size(train_sorted_df,
                                                               time_horizon=time_horizon))  # use custom function to get train size

            # Randomized Search with time series-validation
            random_search = RandomizedSearchCV(
                xgb_regressor,
                param_distributions=xgb_param_dist,
                n_iter=n_iter,  # Number of parameter settings that are sampled
                scoring='neg_root_mean_squared_error',  # 'neg_mean_absolute_error' equivalent to RMSE
                cv=tscv,  # Use TimeSeriesSplit (Cross validation for time series)
                random_state=seed,  # Set a random state for reproducibility
                n_jobs=6  # set n_jobs
            )

            # get parameters (model and random_search is defined outside the loop)
            random_search.fit(x_train, y_train)
            best_params = random_search.best_params_

            print(f"Best params for {level}_{category_name} level: {best_params}")  # log

            # Save best_params
            df_params[f"{level}_{category_name}"] = best_params

            # backtest
            xgb_model = xgb.XGBRegressor(**best_params)
            backtest_df = custom_backtesting(train_sorted_df, x_train, y_train, xgb_model, tscv)

            # forecast
            xgb_model = xgb.XGBRegressor(**best_params)
            xgb_model.fit(x_train, y_train)  # train model on full train set

            # Predict on the current test fold
            y_pred = xgb_model.predict(x_test)  # use test data to test

            # save df
            pred_df = pd.DataFrame({
                "date": test_sorted_df.date,
                "actual": test_sorted_df.quantity,
                "pred": y_pred,
                "group": test_sorted_df.group
            }).reset_index(drop=True)

            # Store the backtest and prediction data frames in the results dictionary
            # with the corresponding name including the category_name
            results[f"{level}_{category_name}"] = {
                'backtest': backtest_df,
                'pred': pred_df
            }

            # Calculate permutation importance
            p_importance = permutation_importance(xgb_model, x_train, y_train, n_repeats=10, random_state=seed, n_jobs=6)

            df_permutation_importance = pd.DataFrame({'feature_names': x_train.columns,
                                                      'importances_mean': p_importance.importances_mean,
                                                      'importances_std': p_importance.importances_std}).sort_values(
                by='importances_mean', ascending=False)

            feature_importance[f"{level}_{category_name}"] = {
                "df": df_permutation_importance,
                "box_array": p_importance.importances.T
            }

            print(
                f"10 most important features for {level}_{category_name} level: {df_permutation_importance.head(10).feature_names.tolist()}")  # log

            # Force garbage collection
            gc.collect()

    # Save dictionaries as pickle files
    with open('../../data/modelling_results/xgb_results.pickle', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('../../data/modelling_results/xgb_params.pickle', 'wb') as handle:
        pickle.dump(df_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('../../data/modelling_results/xgb_importance.pickle', 'wb') as handle:
        pickle.dump(feature_importance, handle, protocol=pickle.HIGHEST_PROTOCOL)