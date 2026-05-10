import pandas as pd


def skewness_3m(close: pd.DataFrame, lb=63):
    ret = close.pct_change()
    return ret.rolling(lb).skew()
