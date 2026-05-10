import numpy as np
import pandas as pd
import statsmodels.api as sm


def _regress_row(row_mom, row_size):
    mask = row_mom.notna() & row_size.notna()
    if mask.sum() < 5:
        return row_mom
    x = sm.add_constant(row_size[mask].values)
    y = row_mom[mask].values
    params = sm.OLS(y, x).fit().params
    res = y - x.dot(params)
    out = pd.Series(index=row_mom.index, dtype=float)
    out.loc[mask] = res
    return out


def residual_momentum(close: pd.DataFrame, lb=126, skip=21):
    mom = close.shift(skip) / close.shift(lb + skip) - 1.0
    log_size = np.log(close.shift(1))
    rows = [_regress_row(mom.loc[d], log_size.loc[d]) for d in mom.index]
    res = pd.DataFrame(rows, index=mom.index)
    return res
