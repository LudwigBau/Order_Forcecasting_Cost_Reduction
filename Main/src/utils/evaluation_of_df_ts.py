import pandas as pd
from utils.model_utils import rmse as RMSE
from sklearn.metrics import mean_absolute_percentage_error as MAPE


def calculate_rmse_mape(true_values, predicted_values):
    rmse_score = RMSE(true_values, predicted_values)[1]
    mape_score = MAPE(true_values, predicted_values)

    return rmse_score, mape_score


# create error dataframe pü
def create_error_dataframes(results, train, val, list_of_names, list_of_category):
    rmse_data = []
    mape_data = []
    index = []
    for model_name in list_of_names:
        model_rmse = []
        model_mape = []

        for category, result in results.items():
            # Calculate RMSE and MAPE for backtest
            true_values = train["quantity"][-188:]
            predicted_values = result["backtest"][model_name.__name__]
            rmse_train, mape_train = calculate_rmse_mape(true_values, predicted_values)

            # Calculate RMSE and mape for forecast
            true_values = val["quantity"]
            predicted_values = result["forecast"][model_name.__name__]
            rmse_val, mape_val = calculate_rmse_mape(true_values, predicted_values)

            model_rmse.extend([rmse_train, rmse_val])
            model_mape.extend([mape_train, mape_val])

        # append error score
        rmse_data.append(model_rmse)
        mape_data.append(model_mape)

        # append string of models names
        index.append(model_name.__name__)

    # Create column names
    columns = pd.Index(list_of_category + [f"{c}_val" for c in list_of_category])

    # Create the DataFrames
    rmse_df = pd.DataFrame(rmse_data, columns=columns, index=index)
    mape_df = pd.DataFrame(mape_data, columns=columns, index=index)

    return rmse_df, mape_df




