# load data
import pandas as pd

# create two path indicated by double for scripts and notebooks

def load_data(double=None):

    if double is not None:
        ### ORDER DATA ###
        df = pd.read_csv("../../data/processed/b2c_orders_clean.csv")

        ### WEATHER DATA ###
        weather_df = pd.read_csv("../../data/intermediate/weather.csv")

        ### Price DATA ###
        prices_df = pd.read_csv("../../data/intermediate/prices.csv")

        ### Holiday Data ###
        holiday_df = pd.read_csv("../../data/intermediate/calender.csv")

    else:
        ### ORDER DATA ###
        df = pd.read_csv("../data/processed/b2c_orders_clean.csv")

        ### WEATHER DATA ###
        weather_df = pd.read_csv("../data/intermediate/weather.csv")

        ### Price DATA ###
        prices_df = pd.read_csv("../data/intermediate/prices.csv")

        ### Holiday Data ###
        holiday_df = pd.read_csv("../data/intermediate/calender.csv")

    return df, weather_df, prices_df, holiday_df

# test
#df, weather_df, prices_df, holiday_df = load_data(double=True)
#print(df.head(2))