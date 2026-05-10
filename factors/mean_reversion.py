import pandas as pd


def short_term_reversal(close: pd.DataFrame, lb=5):
    ret = close.pct_change(lb)
    return -ret


def price_vs_ma(close: pd.DataFrame, lb=20):
    ma = close.rolling(lb).mean()
    return -(close / ma - 1.0)
