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

    # selecting the first data in segmentsDepartureTimeEpochSeconds
    # and segmentsArrivalTimeEpochSeconds
    df['segmentsDepartureTimeEpochSeconds'] = df['segmentsDepartureTimeEpochSeconds'].apply(
        lambda x: x.split("||")[0])
    df['segmentsArrivalTimeEpochSeconds'] = df['segmentsArrivalTimeEpochSeconds'].apply(
        lambda x: x.split("||")[0])

    # convert epoch time to '%Y-%m-%d %H:%M:%S'
    df['segmentsArrivalTimeEpochSeconds'] = pd.to_datetime(
        df['segmentsArrivalTimeEpochSeconds'], unit='s')

    # Convert 'segmentsDepartureTimeEpochSeconds' column
    df['segmentsDepartureTimeEpochSeconds'] = pd.to_datetime(
        df['segmentsDepartureTimeEpochSeconds'], unit='s', errors='coerce')

    # create hour, day of week, month columns
    df['departure_hour'] = df['segmentsDepartureTimeEpochSeconds'].dt.hour
    df['departure_day'] = df['segmentsDepartureTimeEpochSeconds'].dt.day_name()
    df['departure_month'] = df['segmentsDepartureTimeEpochSeconds'].dt.month

    df['arrival_hour'] = df['segmentsArrivalTimeEpochSeconds'].dt.hour
    df['arrival_day'] = df['segmentsArrivalTimeEpochSeconds'].dt.day_name()
    df['arrival_month'] = df['segmentsArrivalTimeEpochSeconds'].dt.month

    return df
