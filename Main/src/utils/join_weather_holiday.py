import pandas as pd 

# Join
def join_weather_holiday(sales_df, weather_df, holiday_df):

    # create copy of all dfs
    sales_df = sales_df.copy()
    weather_df = weather_df.copy()
    holiday_df = holiday_df.copy()

    # sales_df should have no date as index
    sales_df.set_index("date", inplace=True)
    # make date index to merge on date 
    weather_df["date"] = pd.to_datetime(weather_df["date"]).dt.date
    # select germany 
    weather_df = weather_df[weather_df["state"] == "Deutschland"]
    # drop state column
    weather_df.drop(columns=["state", "station_id"], inplace=True)
    weather_df.set_index(["date"], inplace=True)
    
    # Join sales and weather
    weather_sales_df = pd.merge(sales_df, weather_df, left_index=True, right_index=True)

    # Set Holiday date as Index
    holiday_df["date"] = pd.to_datetime(holiday_df.date)
    holiday_df.set_index("date", inplace=True)

    #merge weather_sales_df with holiday_df to get final_df
    final_df = pd.merge(weather_sales_df, holiday_df, left_index=True, right_index=True)

    return final_df

