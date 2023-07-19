# test new datasets

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
import pickle

from itertools import product

# custom utils functions
from src.utils.time_features import to_cyclical_time_features
from src.utils.momentum_features import momentum_grouped
from src.utils.momentum_features import momentum_ungrouped
from src.utils.join_weather_holiday import join_weather_holiday
from src.utils.reduce_mem import reduce_mem_usage

# Ignore warnings
warnings.filterwarnings('ignore')


# Define functions
def create_sales_grid(data, group_by):
    # copy dataset
    data = data.copy()

    # Get grouped dataset by new_product_id on daily bases. average price and sum quantity
    dfx = data.groupby(group_by).sum().drop(
        columns=["order_id", "item_discount", "filled_prices",
                 "order_item_id"])  # Drop useless columns

    # Set start and end date based on original df
    start_date = dfx.index.get_level_values(1).min()
    end_date = dfx.index.get_level_values(1).max()

    # create list of all product ids
    product_ids = dfx.index.get_level_values(0).unique()

    # Create a date range base on start and end date
    date_range = pd.date_range(start_date, end_date)

    # create cartesian product of dates and product_variant id to get full range of dates per product
    C_product = pd.DataFrame.from_records(product(product_ids, date_range), columns=group_by)
    C_product = C_product.set_index(group_by)

    # Merge with q_df, fill NaN with 0 to get real sales data
    # Before 0 sales were simply not present in the dataset
    result_df = C_product.join(dfx, on=group_by, how="left").fillna(0)

    # Delete 0 sales up until the point of first sales
    def remove_zero_sales_before_first_sale(group):
        first_sale_index = group['quantity'].ne(0).idxmax()
        return group.loc[first_sale_index:]

    result_df = result_df.reset_index().groupby(group_by[0]).apply(remove_zero_sales_before_first_sale)

    # Reset index
    result_df = result_df.reset_index(drop=True).set_index(group_by)
    # Reset index again
    result_df = result_df.reset_index()

    return result_df


# cyclical features and momentum
def add_features(df, group_by):
    # copy
    product_sales = df.copy()
    # Add Time and Momentum Features
    product_sales_ts = to_cyclical_time_features(product_sales)
    product_sales_ts_m = momentum_grouped(product_sales_ts, group_by[0])

    return product_sales_ts_m


# Run Script
if __name__ == "__main__":

    # load data
    df = pd.read_pickle("../../data/processed/b2c_orders_clean.pkl").reset_index()
    weather_df = pd.read_csv("../../data/intermediate/weather.csv")
    prices_df = pd.read_csv("../../data/intermediate/prices.csv")
    holiday_df = pd.read_csv("../../data/intermediate/calender.csv")
    customer_deciles_df = pd.read_pickle("../../data/intermediate/customer_deciles.pkl")
    product_deciles_df = pd.read_pickle("../../data/intermediate/product_deciles.pkl")

    # L_1

    # Set group by
    group_by = ['new_product_id', 'date']
    # Print group by
    print(f"{group_by} starts")
    # Unfiltered Sales data
    sales_grid = create_sales_grid(df, group_by)
    print("add features started")
    # merge product_deciles on new_product_id
    sales_grid = pd.merge(sales_grid, product_deciles_df, how='left',
                          left_on=['new_product_id'], right_on=['new_product_id'])
    # Create cyclical and momentum features
    sales_grid_ts_m = add_features(sales_grid, group_by)
    # Sales and Prices
    product_price_df = pd.merge(sales_grid_ts_m, prices_df, how='left',
                                left_on=['new_product_id'], right_on=['new_product_id'])

    # Sales_Prices, weather and holiday
    L_1 = join_weather_holiday(product_price_df, weather_df, holiday_df)

    print(L_1.head(3))

    # L_2

    # Set group by
    group_by = ['state', 'date']
    # Print group by
    print(f"{group_by} starts")
    # Unfiltered Sales data
    sales_grid = create_sales_grid(df, group_by)
    print("add features started")
    # Create cyclical and momentum features
    sales_grid_ts_m = add_features(sales_grid, group_by)
    # Sales_Prices, weather and holiday
    L_2 = join_weather_holiday(sales_grid_ts_m, weather_df, holiday_df)

    print(L_2.head(3))

    # L_3

    # Set group by
    group_by = ['warehouse_chain', 'date']
    # Print group by
    print(f"{group_by} starts")
    # Unfiltered Sales data
    sales_grid = create_sales_grid(df, group_by)
    print("add features started")
    # Create cyclical and momentum features
    sales_grid_ts_m = add_features(sales_grid, group_by)
    # Sales_Prices, weather and holiday
    L_3 = join_weather_holiday(sales_grid_ts_m, weather_df, holiday_df)

    print(L_3.head(3))

    # L_4

    # Set group by
    group_by = ['new_customer_id', 'date']
    # Print group by
    print(f"{group_by} starts")
    # Unfiltered Sales data
    sales_grid = create_sales_grid(df, group_by)
    print("add features started")
    # Create cyclical and momentum features
    sales_grid_ts_m = add_features(sales_grid, group_by)
    # Add deciles to data
    sales_grid_ts_m_d = pd.merge(sales_grid_ts_m, customer_deciles_df[["new_customer_id", "quantity_decile"]], how='left',
                               left_on=["new_customer_id"], right_on=["new_customer_id"])
    # Sales_Prices, weather and holiday
    L_4 = join_weather_holiday(sales_grid_ts_m_d, weather_df, holiday_df)

    print(L_4.head(3))

    # L_5
    print(df.columns)
    # Set group by
    group_by = ['quantity_decile', 'date']
    # Print group by
    print(f"{group_by} starts")

    print(df.columns)
    print(customer_deciles_df.columns)
    # Join customer_deciles
    decile_df = pd.merge(df, customer_deciles_df[["new_customer_id", "quantity_decile"]], how='left',
                                 left_on=["new_customer_id"], right_on=["new_customer_id"])
    # Unfiltered Sales data
    sales_grid = create_sales_grid(decile_df, group_by)
    print("add features started")
    # Create cyclical and momentum features
    sales_grid_ts_m = add_features(sales_grid, group_by)
    # Sales_Prices, weather and holiday
    L_5 = join_weather_holiday(sales_grid_ts_m, weather_df, holiday_df)

    print(L_5.head(3))

    # L_6

    print(f"L_6 starts")
    # Group by date and sum quantity to get top level sales
    grouped_sales = df.groupby("date").sum().reset_index()
    # Select right columns
    total_sales = grouped_sales[["date", "quantity"]]
    # Add time features and lag_momentum_features from scripts
    cyclical_total_sales = to_cyclical_time_features(total_sales)
    m_cyclical_total_sales = momentum_ungrouped(cyclical_total_sales)
    # Join all
    L_6 = join_weather_holiday(m_cyclical_total_sales, weather_df, holiday_df)

    print(L_6.head(3))

    # Decrease Memory

    # Create a dictionary of dataframes
    df_dict = {'L_1': L_1, 'L_2': L_2, 'L_3': L_3, 'L_4': L_4, 'L_5': L_5, 'L_6': L_6}
    print("Decrease memory of all dataframes")

    # Loop over each dataframe in the dictionary and reduce memory usage
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
