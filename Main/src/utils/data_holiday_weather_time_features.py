import pandas as pd
from time_features import to_cyclical_time_features

if __name__ == "__main__":

    weather_df = pd.read_csv("../../data/intermediate/weather.csv")
    holiday_df = pd.read_csv("../../data/intermediate/calender.csv")


    # Join
    def join_weather_holiday(weather_df, holiday_df, state):
        # make copies
        weather = weather_df.copy()
        holiday = holiday_df.copy()

        # date columns to date time
        weather["date"] = pd.to_datetime(weather["date"]).dt.date
        holiday["date"] = pd.to_datetime(holiday["date"]).dt.date

        # Select German if state == "germany", else select all states
        if state == "germany":
            weather = weather[weather["state"] == "Deutschland"]
        else:
            weather = weather

        # merge weather and holiday on dates
        wh_df = pd.merge(weather, holiday, left_on="date", right_on="date")

        final_df = to_cyclical_time_features(wh_df)

        return final_df

    # Apply function for all states and germany
    df_all = join_weather_holiday(weather_df, holiday_df, state="all")
    df_ger = join_weather_holiday(weather_df, holiday_df, state="germany")
    
    # Save dfs to csv
    df_all.to_csv("../data/processed/weather_h_t_all.csv")
    df_ger.to_csv("../data/processed/weather_h_t_ger.csv")