# Dependencies
import pandas as pd
import numpy as np
import math
from feature_engine.creation import CyclicalFeatures

def to_cyclical_time_features(df):
    # Set date clolumn as datetime
    df["date"] = pd.to_datetime(df.date)
    df['tm_dy'] = df.date.dt.dayofyear.astype(np.int8)  # Day of year
    # Make some features from date
    df['tm_dm'] = df.date.dt.day.astype(np.int8)  # Day of month
    df['tm_wy'] = df.date.dt.isocalendar().week.astype(np.int8)  # week of year
    df['tm_my'] = df.date.dt.month.astype(np.int8)  # month of year
    df['tm_y'] = df.date.dt.year  # year
    df['tm_y'] = (df['tm_y'] - df['tm_y'].min()).astype(np.int8)  # year - min year = number of year
    df['tm_wm'] = df['tm_dm'].apply(lambda x: math.ceil(x / 7)).astype(np.int8)  # number of week in month
    df['tm_dw'] = df.date.dt.dayofweek.astype(np.int8)  # number of day in week
    df['tm_w_end'] = (df['tm_dw'] >= 5).astype(np.int8)  # indicate Weekend

    # add cyclical features
    # select cyclical variables
    # note: year and weekend indicator are not cyclical
    variables = ['tm_dy', 'tm_dm', 'tm_wy', 'tm_my', 'tm_wm', 'tm_dw']
    cyclical = CyclicalFeatures(variables=variables, drop_original=True)
    df = cyclical.fit_transform(df)

    return df



def to_time_feature(df):
    # Set date clolumn as datetime
    df["date"] = pd.to_datetime(df.date)
    df['tm_dy'] = df.date.dt.dayofyear.astype(np.int8)  # Day of year
    # Make some features from date
    df['tm_dm'] = df.date.dt.day.astype(np.int8)  # Day of month
    df['tm_wy'] = df.date.dt.isocalendar().week.astype(np.int8)  # week of year
    df['tm_my'] = df.date.dt.month.astype(np.int8)  # month of year
    df['tm_y'] = df.date.dt.year  # year
    df['tm_y'] = (df['tm_y'] - df['tm_y'].min()).astype(np.int8)  # year - min year = number of year
    df['tm_wm'] = df['tm_dm'].apply(lambda x: math.ceil(x / 7)).astype(np.int8)  # number of week in month

    df['tm_dw'] = df.date.dt.dayofweek.astype(np.int8)  # number of day in week
    df['tm_w_end'] = (df['tm_dw'] >= 5).astype(np.int8)  # indicate Weekend

    return df
