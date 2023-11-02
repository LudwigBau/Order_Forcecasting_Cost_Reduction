# create test and train datasets

# Libraries
import pandas as pd

# machine learning split
def ml_data_split(df, days):
    # create df for ml split (m)
    mdf = df.copy()
    mdf["date"] = pd.to_datetime(mdf.date)

    # set cutoff (10 days)
    cutoff = mdf.date.max() - pd.to_timedelta(days, unit='D')

    # split into train and test dataset
    x_train = mdf.loc[mdf.date <= cutoff].copy()
    x_test = mdf.loc[mdf.date > cutoff].copy()

    return x_train, x_test


# time series split
def ts_data_split(df, days, target_variable):
    # create df for time series split (t)
    tdf = df.copy()
    tdf["date"] = pd.to_datetime(tdf.date)

    # set cutoff (10 days)
    cutoff = tdf.date.max() - pd.to_timedelta(days, unit='D')

    # split into train and test dataset
    train_df = tdf.loc[tdf.date <= cutoff].copy()
    test_df = tdf.loc[tdf.date > cutoff].copy()

    # make date index und keep quantity (univariate)
    x_train = train_df.set_index("date")[target_variable]
    x_test = test_df.set_index("date")[target_variable]

    return x_train, x_test


def ml_data_date_split(df, days):
    # create df for ml split (m)
    mdf = df.copy()
    mdf["date"] = pd.to_datetime(mdf.date)

    # define cutoff
    max_date = mdf["date"].max()
    cutoff_date = max_date - pd.Timedelta(days=days)

    mdf_sorted = mdf.sort_values("date")

    # split into train and test dataset
    x_train = mdf_sorted.loc[(df["date"] < cutoff_date), :]
    x_val = mdf_sorted.loc[(df["date"] >= cutoff_date), :]

    return x_train, x_val