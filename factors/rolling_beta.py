import pandas as pd


def rolling_beta(close: pd.DataFrame, lb=126):
    ret = close.pct_change()
    mkt = ret.mean(axis=1)
    cov = ret.rolling(lb, min_periods=lb).cov(mkt)
    var = mkt.rolling(lb, min_periods=lb).var()
    beta = cov.div(var.replace(0, pd.NA), axis=0)
    return -beta
