"""
This module prepares the data for training, edit dates, check types and feature engineering.
"""

import pandas as pd


def data_prep(df: pd.DataFrame, columns_path: list, dist_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    creating a subset with columns wanted.
    """
    # reading distance matrix

    dm = pd.read_csv(dist_matrix)

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
    df['searchDate'] = pd.to_datetime(df['searchDate'])
    df['flightDate'] = pd.to_datetime(df['flightDate'])
    df['days_to_departure'] = (df['flightDate'] - df['searchDate']).dt.days
    df.drop(['flightDate', 'searchDate'], axis=1, inplace=True)
    df = pd.merge(df, dm, on=['destinationAirport', 'destinationAirport'], how='left' )
    df.drop(df.columns[df.columns.str.contains('Airport')], axis =1, inplace= True)
    return df