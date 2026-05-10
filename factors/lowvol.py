import pandas as pd


def beta_lowvol(close: pd.DataFrame, mkt=None, lb=252):
    ret = close.pct_change()
    if mkt is None:
        mkt = ret.mean(axis=1)
    cov = ret.rolling(lb).cov(mkt)
    var = ret.rolling(lb).var()
    beta = cov / var
    lv = -beta
    return lv
