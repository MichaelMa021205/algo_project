import numpy as np
import pandas as pd


def xsection_mom(close: pd.DataFrame, lookback=126, skip=21):
    mom = close.shift(skip) / close.shift(lookback + skip) - 1.0
    mom = mom.replace([np.inf, -np.inf], np.nan)
    return mom
