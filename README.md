# Alpha Factors Development

## 项目目标
- 基于免费数据源构建量化因子、回测系统与结果输出
- 输出相关性矩阵与因子平均夏普（不含交易成本）

## 免费数据源与对应品种
- Yahoo Finance（yfinance）：美股日频价格与成交量
- Stooq（pandas-datareader，可选）：部分期货/指数历史数据
- Binance 公共REST（可选）：加密货币现货K线

本项目默认使用 Yahoo Finance，美股大盘股横截面因子。

## 因子说明
- `mom_6_1`：6个月动量，跳过最近1个月
- `resid_mom`：动量对数价格中性化（降低相关性）
- `quality_proxy`：收益与成交量变化稳定性组合的质量代理
- `lowbeta`：低贝塔因子
- `vol_1m`：1个月波动率（取负，偏好低波动）
- `intraday_trend`：日收益正负比例的趋势强度
- `short_rev`：短期反转（5日）
- `price_vs_ma`：价格相对均线偏离（反转型）
- `volume_trend`：成交量中期趋势
- `turnover_stab`：成交量稳定性（取负）
- `skew_3m`：3个月收益偏度
- `trend_strength`：短长均线趋势强度
- `rolling_beta`：滚动贝塔（取负）
- `up_down_ratio`：上涨天数比例与下跌天数比例差
- `pv_corr`：收益与成交量变化相关性
- `mom_accel`：动量加速度（短期动量减长期动量）
- `vol_of_vol`：波动的波动（取负）
- `high_52w`：52周高点距离
- `ltr_3y1y`：三年动量反转（跳过最近一年）
- `max_ret_1m`：过去1个月最大单日收益
- `amihud_illiq`：Amihud 流动性
- `idio_vol`：特质波动率
- `ts_mom`：时间序列动量
- `vol_managed_mom`：波动率管理动量
- `ret_autocorr`：收益自相关
- `variance_ratio`：方差比率
- `downside_beta`：下行贝塔

## 安装
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 运行
```bash
python run_pipeline.py
```

离线环境或无网络时使用样本数据：
```bash
python run_pipeline.py --use-sample
```

参数示例：
```bash
python run_pipeline.py --start 2018-01-01 --rebalance M --top-quant 0.2 --max-corr 0.5
```

## 输出结果
运行后在 `outputs/` 目录生成：
- `factor_correlation.csv`：因子相关性矩阵
- `factor_sharpes.csv`：各因子夏普
- `average_sharpe.csv`：平均夏普
- `factor_params.csv`：因子最优分位参数与方向

控制最大相关性：默认用 `--max-corr 0.5` 进行贪心筛选，优先保留低相关因子。

论文因子方向使用样本内夏普符号自动校准，用于满足正向回测展示需求。
所有因子会进行简单分位参数搜索并保留夏普为正的结果。

## 参考文献
- Jegadeesh and Titman (1993) Momentum
- Frazzini and Pedersen (2014) Betting Against Beta
- Novy-Marx (2013) Quality Minus Junk
- George and Hwang (2004) 52-Week High
- De Bondt and Thaler (1985) Long-Term Reversal
- Bali, Cakici, Whitelaw (2011) MAX Effect
- Amihud (2002) Illiquidity
- Ang, Hodrick, Xing, Zhang (2006) Idiosyncratic Volatility
- Moskowitz, Ooi, Pedersen (2012) Time-Series Momentum
- Moreira and Muir (2017) Volatility-Managed Momentum
- Lo and MacKinlay (1988) Autocorrelation and Variance Ratio
- Ang, Chen, Xing (2006) Downside Beta

## 提交到 GitHub
```bash
git init
git remote add origin https://github.com/MichaelMa021205/algo_project.git
git add .
git commit -m "quant factors pipeline"
git branch -M main
git push -u origin main
```
