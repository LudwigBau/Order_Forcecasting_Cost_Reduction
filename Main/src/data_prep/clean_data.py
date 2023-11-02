# Libaries
import warnings
warnings.filterwarnings('ignore')

# Libraries
import pandas as pd
# Clean ZIP codes that cannot be located in Germany
import pgeocode

if __name__ == "__main__":
    # Load Data

    # Select all
    df_all = pd.read_csv("../../data/raw/orders_b2c.csv")

    # Select B2C
    df = df_all[df_all["business_type"] == "B2C"]

    # DATA CLEANING

    # Location

    # Select only orders in Germany
    df = df[df["billing_country"] == "DE"]

    # define Germany
    nomi = pgeocode.Nominatim('de')
    # get state names from zip codes and assign to new column
    df["state"] = nomi.query_postal_code(df["billing_zip"].tolist()).state_name.tolist()
    # select non na rows of state
    df = df[df['state'].notna()]

    print("Location finished")

    # Outliers

    # Clean top 1% quartile grouped by shop to catch misslabelled b2b orders
    outliers = df.groupby("customer_id").quantile(0.99).quantity.reset_index()
    outliers = outliers.rename(columns={"quantity": "outliers"})
    df_o = df.merge(outliers, left_on='customer_id', right_on='customer_id', how="left")
    df = df_o[df_o["quantity"] <= df_o["outliers"]].drop(columns=["outliers"])

    print("Outliers finished")

    # Missing Values

    # Fill missing prices with monthly median
    # Create Date
    df["date"] = pd.to_datetime(df["order_datetime"]).dt.date
    df["date"] = pd.to_datetime(df.date)

    # Create monthly median prices
    df["price_m_median"] = df.groupby([df['product_variant_id'], df['date'].dt.month]) \
        .item_price.transform("median")

    # Fill na prices with median prices in a new column
    df['filled_prices'] = df['item_price'].fillna(df['price_m_median'])

    print("Fill Price and Discount finished")

    # Drop na rows of "product_variant_id", "warehouse_chain", "billing_country", "billing_zip", "filled_prices"
    print("Drop na rows")
    # Rest of filled prices get dropped as well
    df_clean = df.dropna(subset=["product_variant_id", "warehouse_chain", "billing_country",
                                 "billing_zip", "filled_prices"])

    print("Drop 0 quantity rows")
    # drop rows with 0 quantity
    df_clean_0 = df_clean[df_clean['quantity'] != 0]

    # rename products and customer_id

    # assume your DataFrame is named 'df'
    df_copy = df_clean_0.copy()

    # create a dictionary mapping old product ids to new ones
    new_product_names = {old_id: new_id for new_id, old_id in enumerate(df_clean['product_variant_id'].unique())}

    # create a dictionary mapping old customer ids to new ones
    new_customer_names = {old_id: new_id for new_id, old_id in enumerate(df_clean['customer_id'].unique())}

    # create a data frame with the old and new ids
    product_df = pd.DataFrame({'product_variant_id': list(new_product_names.keys()),
                               'new_product_id': list(new_product_names.values())})
    customer_df = pd.DataFrame({'customer_id': list(new_customer_names.keys()),
                                'new_customer_id': list(new_customer_names.values())})

    # join the data frames to add the new ids to the original DataFrame
    df_final = df_copy.merge(product_df, on='product_variant_id', how='left').merge(customer_df, on='customer_id',
                                                                                    how='left')

    # Filter the dataframe to include data only for the last 60 days and at least 180 days history
    # to be able to train all models on shop level and to have enough data for Validations
    print("Drop shops with more than 60 days of no orders and or less least 180 days of history")

    # Convert date string into datetime
    date_60_days_ago = df_final.date.max() - pd.Timedelta(days=60)

    date_180_days_ago = df_final.date.max() - pd.Timedelta(days=180)

    
    df_last_60_days = df_final[df_final.date >= date_60_days_ago]
    df_last_180_days = df_final[df_final.date <= date_180_days_ago]

    # Get unique customers who have made sales in the last 90 days
    customers_with_sales_60 = df_last_60_days['new_customer_id'].unique()

    customers_with_sales_180 = df_last_180_days['new_customer_id'].unique()

    # Filter df_clean to only hold new_customer_id that are in df_last_90_days
    df_60 = df_final[df_final['new_customer_id'].isin(customers_with_sales_60)]
    df_final = df_60[df_60['new_customer_id'].isin(customers_with_sales_180)]

    # make new_product_id, new_customer_id, and state categorical variables
    df_final['new_product_id'] = df_final['new_product_id'].astype('category')
    df_final['new_customer_id'] = df_final['new_customer_id'].astype('category')
    df_final['state'] = df_final['state'].astype('category')

    # Drop columns with no valuable information
    df_final.drop(columns=["product_category", "billing_zip", "brand", 'is_flyer_product', 'product_category',
                           'brand', 'order_source', 'marketing_campaign', 'price_m_median', 'item_price',
                           "customer_id", "product_variant_id"], inplace=True)

    print("Drop NA and useless columns finished")

    # Save df as pickel
    df_final.set_index("date").to_pickle("../../data/processed/b2c_orders_clean.pkl")
    df_final.set_index("date").to_csv("../../data/processed/b2c_orders_clean.csv")
    print("Cleaned df saved: script finished")





