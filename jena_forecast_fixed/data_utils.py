
import os
import numpy as np
import pandas as pd
import tensorflow as tf


def download_jena_csv() -> str:

    zip_path = tf.keras.utils.get_file(
        origin="https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip",
        fname="jena_climate_2009_2016.csv.zip",
        extract=True,
    )
    csv_path, _ = os.path.splitext(zip_path)
    return csv_path


def load_dataframe(csv_path: str, subsample_every: int = 6) -> pd.DataFrame:

    df = pd.read_csv(csv_path)
    df = df[5::subsample_every]  
    df["Date Time"] = pd.to_datetime(df["Date Time"], format="%d.%m.%Y %H:%M:%S")
    df = df.set_index("Date Time")
    return df


def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()
    wv = df["wv (m/s)"]
    df.loc[wv == -9999.0, "wv (m/s)"] = 0.0
    max_wv = df["max. wv (m/s)"]
    df.loc[max_wv == -9999.0, "max. wv (m/s)"] = 0.0

    wv = df.pop("wv (m/s)")
    max_wv = df.pop("max. wv (m/s)")
    wd_rad = df.pop("wd (deg)") * np.pi / 180.0
    df["Wx"] = wv * np.cos(wd_rad)
    df["Wy"] = wv * np.sin(wd_rad)
    df["max Wx"] = max_wv * np.cos(wd_rad)
    df["max Wy"] = max_wv * np.sin(wd_rad)
    timestamp_s = df.index.map(pd.Timestamp.timestamp)
    day = 24 * 60 * 60
    year = 365.2425 * day
    df["Day sin"] = np.sin(timestamp_s * (2 * np.pi / day))
    df["Day cos"] = np.cos(timestamp_s * (2 * np.pi / day))
    df["Year sin"] = np.sin(timestamp_s * (2 * np.pi / year))
    df["Year cos"] = np.cos(timestamp_s * (2 * np.pi / year))

    return df


def split_and_normalize(df: pd.DataFrame, train_frac=0.7, val_frac=0.2):
    n = len(df)
    train_df = df[0 : int(n * train_frac)]
    val_df = df[int(n * train_frac) : int(n * (train_frac + val_frac))]
    test_df = df[int(n * (train_frac + val_frac)) :]

    train_mean = train_df.mean()
    train_std = train_df.std()

    train_df = (train_df - train_mean) / train_std
    val_df = (val_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std

    return train_df, val_df, test_df, train_mean, train_std


def get_processed_data():
    csv_path = download_jena_csv()
    df = load_dataframe(csv_path)
    df = clean_and_engineer(df)
    train_df, val_df, test_df, mean, std = split_and_normalize(df)
    return train_df, val_df, test_df, mean, std
