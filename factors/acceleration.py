import pandas as pd


def momentum_acceleration(close: pd.DataFrame, short=21, long=126, skip=21):
    mom_s = close.shift(skip) / close.shift(short + skip) - 1.0
    mom_l = close.shift(skip) / close.shift(long + skip) - 1.0
    return mom_s - mom_l
