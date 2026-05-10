import pandas as pd


def volume_trend(volume: pd.DataFrame, lb=63):
    return volume.pct_change(lb)


def turnover_stability(volume: pd.DataFrame, lb=63):
    vol = volume.pct_change().rolling(lb).std()
    return -vol
