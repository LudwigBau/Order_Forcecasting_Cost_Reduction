# libraries
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# Define Pre_Processing steps
def pre_processing(df, list_of_columns=None, scale_cols=None, encode_cols=None):
    # selected features / columns

    if list_of_columns is None:
        data = df.copy()

    else:
        data = df[list_of_columns].copy

    # Encode
    if encode_cols is None:
        print("No columns to encode")

        # Scale
        if scale_cols is None:
            print("No columns to scale")

        else:
            # copy df
            scaled_df = data.copy()
            # define scaler
            scaler = MinMaxScaler()
            # apply scaler
            scaled_df[scale_cols] = scaler.fit_transform(data[scale_cols])

            print("scaled df")
            return scaled_df

    else:
        # Make year a category
        data[encode_cols] = data[encode_cols].astype('category', copy=False)

        # Get dummies
        just_dummies = pd.get_dummies(data[encode_cols])
        encoded_df = pd.concat([data, just_dummies], axis=1)
        encoded_df.drop(encode_cols, inplace=True, axis=1)

        # Scale
        if scale_cols is None:
            print("No columns to scale")
            print("encoded df")
            return encoded_df

        else:
            # copy df
            scaled_df = encoded_df.copy()
            # define scaler
            scaler = MinMaxScaler()
            # apply scaler
            scaled_df[scale_cols] = scaler.fit_transform(encoded_df[scale_cols])

            print("scaled and encoded df")
            return scaled_df