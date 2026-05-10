import pandas as pd


def vol_of_vol(close: pd.DataFrame, vol_lb=21, volvol_lb=63):
    vol = close.pct_change().rolling(vol_lb).std()
    return -vol.rolling(volvol_lb).std()
