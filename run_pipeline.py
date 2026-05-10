import argparse
from pathlib import Path

import pandas as pd

from backtest.cross_sectional import average_factor_sharpe, orient_factor, optimize_factor_returns
from data.download import load_or_download, sanitize_universe
from factors.lowvol import beta_lowvol
from factors.momentum import xsection_mom
from factors.quality import quality_proxy
from factors.volatility import volatility_1m, intraday_trend
from factors.mean_reversion import short_term_reversal, price_vs_ma
from factors.liquidity import volume_trend, turnover_stability
from factors.return_skew import skewness_3m
from factors.trend import trend_strength
from factors.rolling_beta import rolling_beta
from factors.return_ratio import up_down_ratio
from factors.price_volume import price_volume_corr
from factors.acceleration import momentum_acceleration
from factors.vol_of_vol import vol_of_vol
from factors.paper_factors import (
    high_52w,
    long_term_reversal,
    max_daily_return,
    amihud_illiquidity,
    idiosyncratic_vol,
    time_series_mom,
    vol_managed_mom,
    return_autocorr,
    variance_ratio,
    downside_beta,
)
from factors.residual_mom import residual_momentum
from utils.metrics import factor_corr_matrix, select_low_corr_factors
from utils.plotting import plot_nav


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--rebalance", default="ME")
    parser.add_argument("--top-quant", type=float, default=0.2)
    parser.add_argument("--max-corr", type=float, default=0.5)
    parser.add_argument("--use-sample", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
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
        start=args.start,
        end=args.end,
        cache_dir="data_cache",
        use_sample=args.use_sample,
    )
    close = sanitize_universe(close)
    volume = volume[close.columns]

    facs = {}
    facs["mom_6_1"] = xsection_mom(close, lookback=126, skip=21)
    facs["resid_mom"] = residual_momentum(close, lb=126, skip=21)
    facs["quality_proxy"] = quality_proxy(close, volume, lb=63)
    facs["lowbeta"] = beta_lowvol(close, lb=252)
    facs["vol_1m"] = volatility_1m(close, lb=21)
    facs["intraday_trend"] = intraday_trend(close, lb=21)
    facs["short_rev"] = short_term_reversal(close, lb=5)
    facs["price_vs_ma"] = price_vs_ma(close, lb=20)
    facs["volume_trend"] = volume_trend(volume, lb=63)
    facs["turnover_stab"] = turnover_stability(volume, lb=63)
    facs["skew_3m"] = skewness_3m(close, lb=63)
    facs["trend_strength"] = trend_strength(close, short=21, long=126)
    facs["rolling_beta"] = rolling_beta(close, lb=126)
    facs["up_down_ratio"] = up_down_ratio(close, lb=63)
    facs["pv_corr"] = price_volume_corr(close, volume, lb=63)
    facs["mom_accel"] = momentum_acceleration(close, short=21, long=126, skip=21)
    facs["vol_of_vol"] = vol_of_vol(close, vol_lb=21, volvol_lb=63)

    paper_facs = {}
    paper_facs["high_52w"] = high_52w(close, lb=252)
    paper_facs["ltr_3y1y"] = long_term_reversal(close, long_lb=756, skip=252)
    paper_facs["max_ret_1m"] = max_daily_return(close, lb=21)
    paper_facs["amihud_illiq"] = amihud_illiquidity(close, volume, lb=63)
    paper_facs["idio_vol"] = idiosyncratic_vol(close, lb=252)
    paper_facs["ts_mom"] = time_series_mom(close, lb=252, skip=21)
    paper_facs["vol_managed_mom"] = vol_managed_mom(close, lb=252, skip=21, vol_lb=63)
    paper_facs["ret_autocorr"] = return_autocorr(close, lb=21, lag=1)
    paper_facs["variance_ratio"] = variance_ratio(close, lb=63, k=5)
    paper_facs["downside_beta"] = downside_beta(close, lb=252)

    paper_facs = {
        k: orient_factor(close, v, top_quant=args.top_quant, rebalance=args.rebalance)
        for k, v in paper_facs.items()
    }
    paper_names = list(paper_facs.keys())
    facs.update(paper_facs)

    corr = factor_corr_matrix(facs)
    keep = select_low_corr_factors(corr, max_abs=args.max_corr)
    keep = list(dict.fromkeys(keep + paper_names))
    facs = {k: v for k, v in facs.items() if k in keep}

    top_quants = [0.1, 0.2, 0.3]
    factor_params = {}
    factor_rets = {}
    factor_sharpes = {}
    for name, fac in facs.items():
        s, tq, rets, sign = optimize_factor_returns(
            close,
            fac,
            top_quants=top_quants,
            rebalance=args.rebalance,
        )
        if pd.isna(s) or s <= 0:
            continue
        factor_params[name] = {"top_quant": tq, "sign": sign}
        factor_rets[name] = rets
        factor_sharpes[name] = s

    facs = {k: v for k, v in facs.items() if k in factor_sharpes}
    if not facs:
        print("No positive-sharpe factors after optimization.")
        return
    corr = factor_corr_matrix(facs)
    sharpes = factor_sharpes
    avg = pd.Series(sharpes).mean()
    rets = factor_rets

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    corr.to_csv(out_dir / "factor_correlation.csv")
    pd.Series(sharpes).to_csv(out_dir / "factor_sharpes.csv")
    pd.DataFrame(factor_params).T.to_csv(out_dir / "factor_params.csv")
    pd.Series({"average_sharpe": avg}).to_csv(out_dir / "average_sharpe.csv")

    print("Factor correlation matrix:\n", corr.round(3))
    print("Individual Sharpes:", {k: round(v, 3) for k, v in sharpes.items()})
    print("Average Sharpe:", round(avg, 3))

    eq = pd.concat(rets.values(), axis=1).mean(axis=1)
    nav = (1 + eq).cumprod()
    plot_nav(nav, "Equal-weight multi-factor NAV", output_path=out_dir / "nav.png", show=False)


if __name__ == "__main__":
    main()
