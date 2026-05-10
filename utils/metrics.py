import numpy as np
import pandas as pd


def winsorize(z, low=-3, high=3):
    return z.clip(lower=low, upper=high)


def zscore(df):
    mean = df.mean(axis=1, skipna=True).values.reshape(-1, 1)
    std = df.std(axis=1, ddof=0, skipna=True).values.reshape(-1, 1)
    return (df - mean) / std


def sharpe(returns, ann=252):
    mu = returns.mean() * ann
    sd = returns.std() * np.sqrt(ann)
    return mu / sd if sd != 0 else np.nan


def factor_corr_matrix(factors: dict):
    aligned = [f.stack().rename(name) for name, f in factors.items()]
    panel = pd.concat(aligned, axis=1, sort=False)
    return panel.corr(min_periods=100)


def select_low_corr_factors(corr, max_abs=0.5):
    corr = corr.fillna(0)
    names = list(corr.columns)
    keep = []
    for name in names:
        if not keep:
            keep.append(name)
            continue
        ok = True
        for k in keep:
            if abs(corr.loc[name, k]) > max_abs:
                ok = False
                break
        if ok:
            keep.append(name)
    return keep
