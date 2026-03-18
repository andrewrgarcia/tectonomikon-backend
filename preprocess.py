# preprocess.py

import pandas as pd


def filter_missing_cols(df, max_missing=0.3):
    return df.loc[:, df.isnull().mean() <= max_missing]


def filter_zero_cols(df, max_zero_ratio=0.5):
    return df.loc[:, (df == 0).mean() <= max_zero_ratio]


def filter_low_variance_cols(df, min_variance=1e-8):
    return df.loc[:, df.var() >= min_variance]


def rolling_impute(df, window=24):
    rolled = df.rolling(window=window, min_periods=1, center=True).median()
    return df.fillna(rolled).fillna(df.median())


def preprocess(df):
    print(f"[preprocess] initial cols: {df.shape[1]}")

    df = filter_missing_cols(df)
    df = filter_zero_cols(df)
    df = filter_low_variance_cols(df)

    print(f"[preprocess] after filtering: {df.shape[1]}")

    df = rolling_impute(df)

    return df