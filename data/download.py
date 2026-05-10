from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


def get_yahoo_prices(tickers, start="2015-01-01", end=None):
    end = end or datetime.today().strftime("%Y-%m-%d")
    df = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        group_by="ticker",
        progress=False,
        threads=True,
    )
    close, volume = {}, {}
    for t in tickers:
        tdf = df[t] if isinstance(df.columns, pd.MultiIndex) else df
        close[t] = tdf["Close"]
        volume[t] = tdf["Volume"]
    close = pd.DataFrame(close)
    volume = pd.DataFrame(volume)
    return close.sort_index(), volume.sort_index()


def sanitize_universe(close, min_history=252, min_price=1.0):
    ok_hist = close.rolling(min_history).count().min() >= min_history
    ok_price = close.median() >= min_price
    valid = ok_hist & ok_price
    return close.loc[:, valid.index[valid]]


def save_cache(close, volume, cache_dir):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    close.to_csv(cache_dir / "prices_close.csv")
    volume.to_csv(cache_dir / "prices_volume.csv")


def load_cache(cache_dir):
    cache_dir = Path(cache_dir)
    close_path = cache_dir / "prices_close.csv"
    volume_path = cache_dir / "prices_volume.csv"
    if not close_path.exists() or not volume_path.exists():
        return None, None
    close = pd.read_csv(close_path, index_col=0, parse_dates=True)
    volume = pd.read_csv(volume_path, index_col=0, parse_dates=True)
    return close, volume


def sample_prices(tickers, start="2018-01-01", end="2023-12-31", seed=7):
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start, end=end, freq="B")
    n = len(dates)
    close = pd.DataFrame(index=dates, columns=tickers, dtype=float)
    volume = pd.DataFrame(index=dates, columns=tickers, dtype=float)
    for t in tickers:
        drift = rng.normal(0.08, 0.03)
        vol = rng.normal(0.20, 0.05)
        rets = rng.normal(drift / 252, vol / np.sqrt(252), size=n)
        close[t] = 100 * np.exp(np.cumsum(rets))
        volume[t] = rng.integers(1_000_000, 8_000_000, size=n)
    return close, volume


def load_or_download(tickers, start, end, cache_dir, use_sample=False):
    if use_sample:
        return sample_prices(tickers, start=start, end=end)
    close, volume = load_cache(cache_dir)
    if close is not None and volume is not None:
        return close, volume
    close, volume = get_yahoo_prices(tickers, start=start, end=end)
    save_cache(close, volume, cache_dir)
    return close, volume


if __name__ == "__main__":
    tickers = [
        "AAPL",
        "MSFT",
        "AMZN",
        "GOOGL",
        "META",
        "NVDA",
        "NFLX",
        "TSLA",
        "AMD",
        "AVGO",
        "INTC",
        "ADBE",
        "CRM",
        "COST",
        "PEP",
        "KO",
        "WMT",
        "JNJ",
        "XOM",
        "CVX",
        "JPM",
        "BAC",
        "WFC",
        "PFE",
        "MRK",
        "T",
        "VZ",
        "V",
        "MA",
        "UNH",
        "HD",
        "LLY",
        "ORCL",
        "CSCO",
        "QCOM",
        "TXN",
        "IBM",
        "GE",
    ]
    close, volume = load_or_download(
        tickers,
        start="2015-01-01",
        end=None,
        cache_dir="data_cache",
        use_sample=False,
    )
    close = sanitize_universe(close)
    save_cache(close, volume, "data_cache")
