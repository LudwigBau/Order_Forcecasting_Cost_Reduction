import pandas as pd
import numpy as np

def to_time_feature(df):
    # df has date as index
    
    # Make some features from date
    df['tm_d'] = df.date.dt,day.astype(np.int8) # Day of month
    df['tm_w'] = df.date.dt.week.astype(np.int8) # week of year
    df['tm_m'] = df.date.dt.month.astype(np.int8) # month of year
    df['tm_y'] = df.date.dt.year # year
    df['tm_y'] = (df['tm_y'] - df['tm_y'].min()).astype(np.int8) # year - min year = number of year
    df['tm_wm'] = df['tm_d'].apply(lambda x: ceil(x / 7)).astype(np.int8) # number of week in month

    df['tm_dw'] = df.date.dt.dayofweek.astype(np.int8)  # number of day in week
    df['tm_w_end'] = (df['tm_dw'] >= 5).astype(np.int8)  # indicate Weekend

    return df
