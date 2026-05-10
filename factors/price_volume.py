import pandas as pd


def price_volume_corr(close: pd.DataFrame, volume: pd.DataFrame, lb=63):
    ret = close.pct_change()
    vol = volume.pct_change()
    return ret.rolling(lb).corr(vol)
