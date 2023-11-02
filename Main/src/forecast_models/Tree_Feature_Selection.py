# Library
import pandas as pd
import pickle

from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.inspection import permutation_importance

import lightgbm as lgb

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")
warnings.simplefilter("ignore", category=RuntimeWarning)

# Get custom utils functions
from src.utils.data_split import ml_data_date_split
from src.utils.model_utils import custom_backtesting
from src.utils.model_utils import get_test_size
from src.utils.model_utils import make_category_df

# drop columns
drop_columns = ['tm_y_1', 'tm_wm_cos', 'event_Thanksgiving', 'holiday_Zweiter_Weihnachtstag', 'holiday_Ostermontag',
                'holiday_Neujahr', 'holiday_Reformationstag', 'holiday_Karfreitag', 'event_Valentines_Day',
                'holiday_Pfingstmontag', 'holiday_Erster_Mai', 'holiday_Erster_Weihnachtstag',
                'holiday_Tag_der_Deutschen_Einheit', 'holiday_Christi_Himmelfahrt', 'tm_y_2']

if __name__ == "__main__":

    # set fixed variables
    seed = 42
    time_horizon = 9
    n_iter = 200

    # Load Data

    # Only load warehouse_chain, customers and total to mimimize computational time
    df_list = ["L_6"]
    group_list = ["empty"]


    test_data = {}
    for i in df_list:
        test_data[i] = pd.read_pickle(f"../../data/processed/{i}_test.pkl")

    # drop columns fro all levels in test_data
    for key, df in test_data.items():

        # if in key the name "L_4"
        df.drop(columns=[col for col in drop_columns if col in df.columns], inplace=True)
        test_data[key] = df
        # else nothing

    # Change group name
    print("Change group name")
    for df_name, group_name in zip(df_list, group_list):
        if group_name != "empty":
            test_data[df_name] = test_data[df_name].rename(columns={group_name: 'group'})
        else:
            test_data[df_name]["group"] = "empty"

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

            print(f"10 most important features for {level}_{category_name} "
                  f""f"level: {df_permutation_importance.head(10).feature_names.tolist()}")  # log

    # Save dictionaries as pickle files
    with open('../../data/modelling_results/lgbm_results_selected_f.pickle', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)