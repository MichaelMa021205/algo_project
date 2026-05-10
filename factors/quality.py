import pandas as pd


def quality_proxy(close: pd.DataFrame, volume: pd.DataFrame, lb=63):
    ret = close.pct_change()
    vol_chg = volume.pct_change().clip(-0.5, 0.5)
    q = ret.rolling(lb).mean() * 0.7 + vol_chg.rolling(lb).mean() * 0.3
    return q
