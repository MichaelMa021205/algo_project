import pandas as pd


def trend_strength(close: pd.DataFrame, short=21, long=126):
    s = close.rolling(short).mean()
    l = close.rolling(long).mean()
    return s / l - 1.0
