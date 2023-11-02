# Libaries
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


if __name__ == "__main__":
    # Load Data

    ### ORDER DATA ###
    df = pd.read_csv("../../data/processed/b2c_orders_clean.csv")

    ##### PRICES #####

    # Create new df holding price info
    prices_df = df.groupby(["new_product_id", "date"])["filled_prices", "item_discount"].mean()

    # Create real sell_price
    prices_df["sell_price"] = prices_df["filled_prices"] - prices_df["item_discount"]

    # Basics
    prices_df["price_max"] = prices_df.groupby(["new_product_id"])["filled_prices"].transform("max")
    prices_df["price_min"] = prices_df.groupby(['new_product_id'])['filled_prices'].transform('min')
    prices_df["price_std"] = prices_df.groupby(['new_product_id'])['filled_prices'].transform('std')
    prices_df["price_mean"] = prices_df.groupby(['new_product_id'])['filled_prices'].transform('mean')

    # How many price changes
    prices_df['price_nunique'] = prices_df.groupby(['new_product_id'])['filled_prices'].transform('nunique')

    ##### DISCOUNT #####

    # Basics
    prices_df["discount_max"] = prices_df.groupby(["new_product_id"])["item_discount"].transform("max")
    prices_df["discount_min"] = prices_df.groupby(['new_product_id'])['item_discount'].transform('min')
    prices_df["discount_std"] = prices_df.groupby(['new_product_id'])['item_discount'].transform('std')
    prices_df["discount_mean"] = prices_df.groupby(['new_product_id'])['item_discount'].transform('mean')

    # How many unique discount
    prices_df['discount_nunique'] = prices_df.groupby(['new_product_id'])['item_discount'].transform('nunique')

    # discount in percentage sell_price / item_price
    prices_df["discount_percent"] = 1 - (prices_df["sell_price"] / prices_df["filled_prices"])

    # Reset index
    prices_df.reset_index(inplace=True)

    # Group by product_id
    grouped_df = prices_df.groupby("new_product_id").mean()
    final_df = grouped_df.fillna(0)

    #Save
    final_df.to_csv("../../data/intermediate/prices.csv")