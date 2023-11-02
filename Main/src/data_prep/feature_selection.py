# Libraries
import pandas as pd

from src.utils.data_split import ml_data_date_split
from src.utils.model_utils import get_top_features


# Run Script
if __name__ == "__main__":

    # Load the data
    L_1 = pd.read_pickle("../../data/processed/L_1_train.pkl")
    L_2 = pd.read_pickle("../../data/processed/L_2_train.pkl")
    L_3 = pd.read_pickle("../../data/processed/L_3_train.pkl")
    L_4 = pd.read_pickle("../../data/processed/L_4_train.pkl")
    L_5 = pd.read_pickle("../../data/processed/L_5_train.pkl")
    L_6 = pd.read_pickle("../../data/processed/L_6_train.pkl")

    # Assuming your dataframes are named L_1, L_2, L_3, L_4, L_5, L_6
    df_list = [L_1, L_2, L_3, L_4, L_5, L_6]

    # Initialize a dictionary to store the top features for each dataset
    top_features_dict = {}

    # Iterate through the datasets and store the top features
    for i, df in enumerate(df_list, start=1):
        # Prepare the train and validation sets
        cols = [col for col in df.columns if col not in ["date", "state", "warehouse_chain",
                                                         'new_product_id', "new_customer_id", "quantity"]]

        # split data with custom function
        train, val = ml_data_date_split(df, 30)

        # define train and val Y and X
        Y_train = train['quantity']
        X_train = train[cols]

        Y_val = val['quantity']
        X_val = val[cols]

        # Get the top features for each dataset
        top_features = get_top_features(X_train, Y_train, X_val, Y_val)
        top_features_dict[f"L_{i}_features"] = top_features

        # Print the top features
        print(f"L_{i} features selected: {top_features}")

    # Convert the dictionary into a DataFrame
    df_top_features = pd.DataFrame(top_features_dict)

    # Save Dataframe as pickel file in data/selected_features
    df_top_features.to_pickle("../../data/selected_features/top_30_lgbm_features.pkl")
