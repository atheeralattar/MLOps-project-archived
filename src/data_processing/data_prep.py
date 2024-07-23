"""
This module prepares the data for training, edit dates, check types and feature engineering.
"""

import pandas as pd


def data_prep(df: pd.DataFrame, columns_path: list) -> pd.DataFrame:
    """
    creating a subset with columns wanted.
    """
    # creating a subset of the dataframe
    columns = pd.read_csv(columns_path).iloc[:, 0].tolist()
    df = df[columns]

    # # selecting the first data in segmentsDepartureTimeEpochSeconds
    df['segmentsDepartureTimeEpochSeconds'] = df['segmentsDepartureTimeEpochSeconds'].apply(
        lambda x: x.split("||")[0])
    
    # # Convert 'segmentsDepartureTimeEpochSeconds' column
    df['segmentsDepartureTimeEpochSeconds'] = pd.to_datetime(
        df['segmentsDepartureTimeEpochSeconds'], unit='s', errors='coerce')

    # # create hour, day of week, month columns
    df['departure_hour'] = df['segmentsDepartureTimeEpochSeconds'].dt.hour
    df['departure_day'] = df['segmentsDepartureTimeEpochSeconds'].dt.day_name()
    df['departure_month'] = df['segmentsDepartureTimeEpochSeconds'].dt.month
    df.drop('segmentsDepartureTimeEpochSeconds', axis=1, inplace=True)
    return df
