# NOTE: dfs[df_name] = {"train": train_df, "test": df} includes df not test_df so make scaling work

# Libraries
import pandas as pd
import pickle
import category_encoders as ce

from src.utils.data_split import ml_data_date_split
from src.utils.pre_processing import pre_processing

import warnings

warnings.filterwarnings('ignore')

# Define parameters
# Set up covariates

# All Columns (we need date)
list_of_columns = ["date", 'tm_y', 'tm_w_end', 'tm_dy_sin', 'tm_dy_cos', 'tm_dm_sin',
                   'tm_dm_cos', 'tm_wy_sin', 'tm_wy_cos', 'tm_my_sin', 'tm_my_cos',
                   'tm_wm_sin', 'tm_wm_cos', 'tm_dw_sin', 'tm_dw_cos', 'q_roll_mean_9d',
                   'q_roll_std_9d', 'q_roll_mean_14d', 'q_roll_std_14d', 'q_lag_365d',
                   'q_lag_9d', 'q_lag_14d', 'q_lag_28d', 'q_mean_lag_9_14_28',
                   'precipitation_height', 'sunshine_duration', 'temperature_air_mean_200',
                   'sunshine_duration_h', 'suns_classes', 'temp_classes', 'rain_classes',
                   'is_holiday', 'is_event', 'xmas_1_week', 'xmas_2_week',
                   'after_xmas_week', 'event_Black_Friday', 'event_Cyber_Monday',
                   'event_Thanksgiving', 'event_Valentines_Day',
                   'holiday_Christi_Himmelfahrt', 'holiday_Erster_Mai',
                   'holiday_Erster_Weihnachtstag', 'holiday_Karfreitag', 'holiday_Neujahr',
                   'holiday_Ostermontag', 'holiday_Pfingstmontag',
                   'holiday_Reformationstag', 'holiday_Tag_der_Deutschen_Einheit',
                   'holiday_Zweiter_Weihnachtstag', 'blackweek', 'blackweekend',
                   'aftercyberweek']

# All covariates
list_of_covariates = ['tm_y', 'tm_w_end', 'tm_dy_sin', 'tm_dy_cos', 'tm_dm_sin',
                      'tm_dm_cos', 'tm_wy_sin', 'tm_wy_cos', 'tm_my_sin', 'tm_my_cos',
                      'tm_wm_sin', 'tm_wm_cos', 'tm_dw_sin', 'tm_dw_cos', 'q_roll_mean_9d',
                      'q_roll_std_9d', 'q_roll_mean_14d', 'q_roll_std_14d', 'q_lag_365d',
                      'q_lag_9d', 'q_lag_14d', 'q_lag_28d', 'q_mean_lag_9_14_28',
                      'precipitation_height', 'sunshine_duration', 'temperature_air_mean_200',
                      'sunshine_duration_h', 'suns_classes', 'temp_classes', 'rain_classes',
                      'is_holiday', 'is_event', 'xmas_1_week', 'xmas_2_week',
                      'after_xmas_week', 'event_Black_Friday', 'event_Cyber_Monday',
                      'event_Thanksgiving', 'event_Valentines_Day',
                      'holiday_Christi_Himmelfahrt', 'holiday_Erster_Mai',
                      'holiday_Erster_Weihnachtstag', 'holiday_Karfreitag', 'holiday_Neujahr',
                      'holiday_Ostermontag', 'holiday_Pfingstmontag',
                      'holiday_Reformationstag', 'holiday_Tag_der_Deutschen_Einheit',
                      'holiday_Zweiter_Weihnachtstag', 'blackweek', 'blackweekend',
                      'aftercyberweek']

# Covariates to scale
scale_cols = ['q_roll_mean_9d', 'q_roll_std_9d', 'q_roll_mean_14d', 'q_roll_std_14d', 'q_lag_365d',
              'q_lag_9d', 'q_lag_14d', 'q_lag_28d', 'q_mean_lag_9_14_28',
              'precipitation_height', 'sunshine_duration', 'temperature_air_mean_200',
              'sunshine_duration_h', 'suns_classes', 'temp_classes', "rain_classes"]

# categorical columns
encode_cols = ["tm_y", "ratio_decile", "quantity_decile"]

# Define functions (binary encoder)
def binary_encode_with_original(data, column):
    # Create a copy of the original column
    original_column = data[column].copy()

    # Create a BinaryEncoder instance for the specified column
    encoder = ce.BinaryEncoder(cols=[column])

    # Fit and transform the data
    data_encoded = encoder.fit_transform(data)

    # Add the original column back to the DataFrame
    data_encoded[f'{column}'] = original_column

    return data_encoded

# Run Script
if __name__ == "__main__":

    # split all dfs from list and save in dictionary
    df_list = ["L_1", "L_2", "L_3", "L_4", "L_5", "L_6"]
    dfs = {}

    for df_name in df_list:

        # Add condition for L_4
        if df_name == "L_4":
            df = pd.read_pickle(f"../../data/intermediate/{df_name}.pkl").reset_index()
            # encode
            df = pre_processing(df, encode_cols=["tm_y", "quantity_decile"])

            # Create a BinaryEncoder instance for the new_customer_id column
            df = binary_encode_with_original(df, 'new_customer_id')

            # split
            train_df, val_df = ml_data_date_split(df, 8)
            # add to dictionary
            dfs[df_name] = {"train": train_df, "test": df}

        # Add condition for L_1
        elif df_name == "L_1":
            df = pd.read_pickle(f"../../data/intermediate/{df_name}.pkl").reset_index()
            # encode
            df = pre_processing(df, encode_cols=["tm_y", "ratio_decile"])

            # Create a BinaryEncoder instance for the new_product_id column
            #df = binary_encode_with_original(df, 'new_product_id')

            # split
            train_df, val_df = ml_data_date_split(df, 8)
            # add to dictionary
            dfs[df_name] = {"train": train_df, "test": df}

        # rest
        else:
            df = pd.read_pickle(f"../../data/intermediate/{df_name}.pkl").reset_index()
            # encode
            df = pre_processing(df, encode_cols=["tm_y"])
            # split
            train_df, val_df = ml_data_date_split(df, 8)
            # add to dictionary
            dfs[df_name] = {"train": train_df, "test": df}

    print("data split finished")

    # Define a new dictionary to store pre-processed data
    processed_dfs = {}

    # Loop over each dataframe in the dfs dictionary and apply the preprocessing function
    for df_name, df_dict in dfs.items():
        train_df = df_dict['train']
        test_df = df_dict['test']
        processed_train_df = pre_processing(train_df, scale_cols=scale_cols)
        processed_test_df = pre_processing(test_df, scale_cols=scale_cols)
        processed_dfs[f"{df_name}_train"] = processed_train_df
        processed_dfs[f"{df_name}_test"] = processed_test_df

    print("pre processing finished")

    # Loop over each dataframe in the processed_dfs dictionary and save it as a pickle
    for name, df in processed_dfs.items():
        with open(f'../../data/processed/{name}.pkl', 'wb') as f:
            pickle.dump(df, f)

        print(f"Saved {name} data")

    print("Script Finished")
