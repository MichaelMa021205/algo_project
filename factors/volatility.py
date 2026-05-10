import numpy as np
import pandas as pd


def volatility_1m(close: pd.DataFrame, lb=21):
    vol = close.pct_change().rolling(lb).std()
    return -vol


def intraday_trend(close: pd.DataFrame, lb=21):
    ret = close.pct_change()
    pos = ret.clip(lower=0).rolling(lb).sum()
    neg = (-ret.clip(upper=0)).rolling(lb).sum()
    return pos - neg
