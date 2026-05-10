import numpy as np
import pandas as pd


def high_52w(close: pd.DataFrame, lb=252):
    high = close.rolling(lb, min_periods=lb).max()
    return close / high - 1.0


def long_term_reversal(close: pd.DataFrame, long_lb=756, skip=252):
    return -(close.shift(skip) / close.shift(long_lb + skip) - 1.0)


def max_daily_return(close: pd.DataFrame, lb=21):
    ret = close.pct_change()
    mx = ret.rolling(lb, min_periods=lb).max()
    return -mx


def amihud_illiquidity(close: pd.DataFrame, volume: pd.DataFrame, lb=63):
    ret = close.pct_change().abs()
    dollar = close * volume
    illiq = (ret / dollar.replace(0, pd.NA)).rolling(lb, min_periods=lb).mean()
    return -illiq


def idiosyncratic_vol(close: pd.DataFrame, lb=252):
    ret = close.pct_change()
    mkt = ret.mean(axis=1)
    cov = ret.rolling(lb, min_periods=lb).cov(mkt)
    var = mkt.rolling(lb, min_periods=lb).var()
    beta = cov.div(var.replace(0, pd.NA), axis=0)
    resid = ret - beta.mul(mkt, axis=0)
    resid_vol = resid.rolling(lb, min_periods=lb).std()
    return -resid_vol


def time_series_mom(close: pd.DataFrame, lb=252, skip=21):
    return close.shift(skip) / close.shift(lb + skip) - 1.0


def vol_managed_mom(close: pd.DataFrame, lb=252, skip=21, vol_lb=63):
    mom = close.shift(skip) / close.shift(lb + skip) - 1.0
    vol = close.pct_change().rolling(vol_lb, min_periods=vol_lb).std()
    return mom / vol.replace(0, pd.NA)


def return_autocorr(close: pd.DataFrame, lb=21, lag=1):
    ret = close.pct_change()
    ac = ret.apply(lambda s: s.rolling(lb, min_periods=lb).corr(s.shift(lag)))
    return -ac


def variance_ratio(close: pd.DataFrame, lb=63, k=5):
    ret_1 = close.pct_change()
    ret_k = close.pct_change(k)
    var_k = ret_k.rolling(lb, min_periods=lb).var()
    var_1 = ret_1.rolling(lb, min_periods=lb).var()
    vr = var_k / (k * var_1.replace(0, pd.NA))
    return vr - 1.0


def downside_beta(close: pd.DataFrame, lb=252):
    ret = close.pct_change()
    mkt = ret.mean(axis=1)
    mkt_down = mkt.where(mkt < 0, 0)
    ret_down = ret.where(mkt < 0, 0)
    cov = ret_down.rolling(lb, min_periods=lb // 2).cov(mkt_down)
    var = mkt_down.rolling(lb, min_periods=lb // 2).var()
    beta = cov.div(var.replace(0, pd.NA), axis=0)
    return -beta
