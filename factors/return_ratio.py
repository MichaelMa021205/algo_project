import pandas as pd


def up_down_ratio(close: pd.DataFrame, lb=63):
    ret = close.pct_change()
    up = (ret > 0).rolling(lb).mean()
    down = (ret < 0).rolling(lb).mean()
    return up - down
