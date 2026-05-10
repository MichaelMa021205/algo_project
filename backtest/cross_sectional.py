import numpy as np
import pandas as pd

from utils.metrics import sharpe, winsorize, zscore


def prepare_factor(factor: pd.DataFrame):
    fac = factor.copy()
    fac = fac.replace([np.inf, -np.inf], np.nan)
    fac = zscore(fac)
    fac = winsorize(fac, -3, 3)
    return fac


def build_longshort_returns(
    close: pd.DataFrame,
    factor: pd.DataFrame,
    top_quant=0.2,
    rebalance="ME",
):
    fac = prepare_factor(factor)
    signal = fac.resample(rebalance).last()
    groups = signal.index
    daily_ret = close.pct_change().shift(-1)
    port_ret = []
    for i in range(len(groups) - 1):
        dt0, dt1 = groups[i], groups[i + 1]
        cross = signal.loc[dt0].dropna()
        if len(cross) < 10:
            zeros = pd.Series(0.0, index=daily_ret.loc[dt0:dt1].index)
            port_ret.append(zeros)
            continue
        q = cross.quantile([1 - top_quant, top_quant])
        longs = cross[cross >= q.iloc[1]].index
        shorts = cross[cross <= q.iloc[0]].index
        w = pd.Series(0.0, index=cross.index, dtype=float)
        if len(longs) > 0:
            w.loc[longs] = 1 / len(longs)
        if len(shorts) > 0:
            w.loc[shorts] = -1 / len(shorts)
        ret = daily_ret.loc[dt0:dt1, w.index].mul(w, axis=1).sum(axis=1)
        port_ret.append(ret)
    port = pd.concat(port_ret).dropna()
    return port


def average_factor_sharpe(close, factors: dict, force_positive=None):
    rets = {k: build_longshort_returns(close, f) for k, f in factors.items()}
    sharpes = {k: sharpe(v) for k, v in rets.items()}
    if force_positive:
        for k in force_positive:
            if k in sharpes and pd.notna(sharpes[k]) and sharpes[k] < 0:
                rets[k] = -rets[k]
                sharpes[k] = -sharpes[k]
    avg = np.nanmean(list(sharpes.values())) if sharpes else np.nan
    return sharpes, avg, rets


def orient_factor(close, factor, top_quant=0.2, rebalance="ME"):
    rets = build_longshort_returns(
        close,
        factor,
        top_quant=top_quant,
        rebalance=rebalance,
    )
    s = sharpe(rets)
    if pd.notna(s) and s < 0:
        return -factor
    return factor


def optimize_factor_returns(
    close,
    factor,
    top_quants,
    rebalance,
):
    best = None
    for tq in top_quants:
        rets = build_longshort_returns(close, factor, top_quant=tq, rebalance=rebalance)
        s = sharpe(rets)
        rets_flip = -rets
        s_flip = sharpe(rets_flip)
        if best is None or (pd.notna(s) and s > best[0]):
            best = (s, tq, rets, 1)
        if best is None or (pd.notna(s_flip) and s_flip > best[0]):
            best = (s_flip, tq, rets_flip, -1)
    if best is None:
        return np.nan, None, pd.Series(dtype=float), 1
    return best
