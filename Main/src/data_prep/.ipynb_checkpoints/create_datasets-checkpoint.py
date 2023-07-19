# Readme:
# Create six datasets to aggregate according to hierarchy
# L1: Product
# L2: Product Location
# L3: Warehouse
# L4: Customer
# L5: Customer Cluster
# L6: Total Sales

# Libraries
import warnings

import pandas as pd
import numpy as np
import pickle

from itertools import product

# custom utils functions
from src.utils.time_features import to_cyclical_time_features
from src.utils.momentum_features import momentum_grouped
from src.utils.momentum_features import momentum_ungrouped
from src.utils.join_weather_holiday import join_weather_holiday
from src.utils.load_data import load_data
from src.utils.reduce_mem import reduce_mem_usage

# Ignore warnings
warnings.filterwarnings('ignore')

# Run Script
if __name__ == "__main__":

    # load data
    df, weather_df, prices_df, holiday_df = load_data(double=True)

    # Create date column
    df["date"] = pd.to_datetime(df["order_datetime"]).dt.date
    df["date"] = pd.to_datetime(df.date)

    # L_1
    print("L_1 starts")

    # Get grouped dataset by new_product_id on daily bases. average price and sum quanity
    q_df = df.groupby(["new_product_id", "date"]).mean().drop(
        columns=["order_id", "item_discount", "filled_prices",
                 "order_item_id", "new_customer_id"])  # Drop useless columns
    # Add quantiy column
    q_df["quantity"] = df.groupby(["new_product_id", "date"]).quantity.sum()


    ### SALES PRODUCT LEVEL GRID BASED FOR ALL DATES
    def create_sales_grid(df, group_by):

        # Set start and end date based on original df
        start_date = df.index.get_level_values(1).min()
        end_date = df.index.get_level_values(1).max()

        # create list of all product ids
        product_ids = df.index.get_level_values(0).unique()

        # Create a date range base on start and end date
        date_range = pd.date_range(start_date, end_date)

        # create cartesian product of dates and product_variant id to get full range of dates per product
        C_product = pd.DataFrame.from_records(product(product_ids, date_range), columns=group_by)
        C_product = C_product.set_index(group_by)

        # Merge with q_df, fill NaN with 0 to get real sales data
        # Before 0 sales were simply not present in the dataset
        result_df = C_product.join(q_df, on=group_by, how="left").fillna(0)

        # Delete 0 sales up until point of first sale

        return result_df


    group_by = ['new_product_id', 'date']

    # Unfiltered Sales data
    product_sales = create_sales_grid(q_df, group_by)

    # Now delete all 0 quantity data before first entry and after last entry

    # create turple to Indicate first and last index of non zero entries in "quantity"
    # thus, now we deleted sales data before first index
    q_valid_index = product_sales.reset_index().groupby('new_product_id') \
        .apply(lambda x: (x[x['quantity'] > 0].index[0], x[x['quantity'] > 0].index[-1]))

    # Use boolean indexing
    # 1. Create mask, same length as df that is true for all indices set in the step before
    mask = np.zeros(len(product_sales), dtype=bool)
    for start, end in q_valid_index:
        mask[start:end + 1] = True

    # Add mask
    product_sales = product_sales.reset_index().loc[mask]

    # Reset Index
    # product_sales.set_index(["new_product_id", "date"], inplace=True)

    # Add Time and Momentum Features
    product_sales_ts = to_cyclical_time_features(product_sales)
    product_sales_ts_m = momentum_grouped(product_sales_ts, "new_product_id")

    # Join
    # Sales and Prices
    product_price_df = pd.merge(product_sales_ts_m, prices_df, how='left',
                                left_on=['new_product_id'], right_on=['new_product_id'])

    # Sales_Prices, weather and holiday
    L_1 = join_weather_holiday(product_price_df, weather_df, holiday_df)

    # L_2
    print("L_2 starts")
    # load data
    df, weather_df, prices_df, holiday_df = load_data(double=True)

    # Change state names
    # From get_weather_data notebook:

    # Change names of station names to avoid errors
    df['state'] = df['state'].str.replace('-', '_')
    df['state'] = df['state'].str.replace('ü', 'ue')
    df['state'] = df['state'].str.replace('/', '_')

    # test
    # check if both lists have the same entries
    a = df.state.unique().tolist()
    b = weather_df.state.unique().tolist()

    # test (16 unique states) correct!
    print("test states 16 = ", len(set(a) & set(b)))

    # group datasets on date and sum quantity

    # Create date column
    df["date"] = pd.to_datetime(df["order_datetime"]).dt.date

    # Order: Group by date and state
    state_sales_df = df.groupby(["date", "state"]).sum()

    # get cyclical and momentum features
    state_sales_ts = to_cyclical_time_features(state_sales_df.reset_index())
    state_sales_ts_m = momentum_grouped(state_sales_ts, "state")


    # Join
    def join_all(state_day_sales_df, weather_df, holiday_df):

        # make date index to merge on date
        weather_df["date"] = pd.to_datetime(weather_df["date"]).dt.date
        weather_df.set_index(["date", "state"], inplace=True)

        state_day_sales_df.set_index(["date", "state"], inplace=True)

        # Join state and weather
        weather_sales_df = pd.merge(state_day_sales_df, weather_df, left_index=True, right_index=True)

        weather_sales_df.reset_index(inplace=True)
        weather_sales_df["date"] = pd.to_datetime(weather_sales_df.date)
        weather_sales_df.set_index("date", inplace=True)

        holiday_df["date"] = pd.to_datetime(holiday_df.date)
        holiday_df.set_index("date", inplace=True)

        # merge weather_sales_df with holiday_df to get final_df
        final_df = pd.merge(weather_sales_df, holiday_df, left_index=True, right_index=True)

        # Drop columns useless columns
        drop_columns = ["filled_prices", 'order_id', 'order_item_id', 'new_customer_id',
                        'new_product_id', 'item_discount', 'station_id']

        final_df = final_df.drop(columns=drop_columns)
        return final_df


    L_2 = join_all(state_sales_ts_m, weather_df, holiday_df)

    # L_3
    print("L_3 starts")
    # load data
    df, weather_df, prices_df, holiday_df = load_data(double=True)

    warehouse_df = df.groupby(["date", "warehouse_chain"]).sum().quantity.reset_index()

    m_warehouse_df = momentum_grouped(warehouse_df, "warehouse_chain")
    ts_m_warehouse_df = to_cyclical_time_features(m_warehouse_df)

    # Join all
    L_3 = join_weather_holiday(ts_m_warehouse_df, weather_df, holiday_df)

    # L_4
    print("L_4 starts")
    # load data
    df, weather_df, prices_df, holiday_df = load_data(double=True)
    customer_df = df.groupby(["date", "new_customer_id"]).sum().quantity.reset_index()

    m_customer_df = momentum_grouped(customer_df, "new_customer_id")
    ts_m_customer_df = to_cyclical_time_features(m_customer_df)

    # Join all
    L_4 = join_weather_holiday(ts_m_customer_df, weather_df, holiday_df)

    # L_5
    print("L_5 starts")
    # load data
    df, weather_df, prices_df, holiday_df = load_data(double=True)
    cluster_df = pd.read_csv("../../data/intermediate/clusters.csv")

    ### CLUSTER DATA
    customer_df = df.groupby(["date", "new_customer_id"]).sum().quantity.reset_index()
    cluster_customer_df = pd.merge(customer_df, cluster_df[["new_customer_id", 'cluster_price', 'cluster_size']],
                                   on='new_customer_id', how='left')

    cluster_grouped = cluster_customer_df.groupby(["date", "cluster_size"]).sum().quantity.reset_index()

    m_customer_df = momentum_grouped(cluster_customer_df, "cluster_size")
    ts_m_customer_df = to_cyclical_time_features(m_customer_df)
    # Join all
    L_5 = join_weather_holiday(ts_m_customer_df, weather_df, holiday_df)

    # L_6
    print("L_6 starts")
    # load data
    df, weather_df, prices_df, holiday_df = load_data(double=True)
    # Group sales based on daily sales
    grouped_sales = df.groupby("date").sum().reset_index()
    # Select right columns
    total_sales = grouped_sales[["date", "quantity"]]

    # Add time features and lag_momentum_features from scripts
    cyclical_total_sales = to_cyclical_time_features(total_sales)
    m_cyclical_total_sales = momentum_ungrouped(cyclical_total_sales)
    # Join all
    L_6 = join_weather_holiday(m_cyclical_total_sales, weather_df, holiday_df)

    # Decrease Memory
    # Create a dictionary of dataframes
    df_dict = {'L_1': L_1, 'L_2': L_2, 'L_3': L_3, 'L_4': L_4, 'L_5': L_5, 'L_6': L_6}
    print("Decrease memory of all dataframes")

    count = 0
    for name, df in df_dict.items():
        df = reduce_mem_usage(df)
        count += 1
        print("The reduced mem of: L_", count)

    # Save all dataframes to pkl in intermediate folder
    print("Save all dataframes to pkl in intermediate folder")

    # Loop over each dataframe in the dictionary and save it as a pickle
    for name, df in df_dict.items():
        with open(f'../../data/intermediate/{name}.pkl', 'wb') as f:
            pickle.dump(df, f)
