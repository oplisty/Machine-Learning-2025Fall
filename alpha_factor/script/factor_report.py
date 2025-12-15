import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 报告输出目录
output_dir = "output/factor_report_plots"
os.makedirs(output_dir, exist_ok=True)

# 读取并预处理（与 factor_visual.py 保持一致）
df = pd.read_csv("data.csv")

df["timestamps"] = pd.to_datetime(df["timestamps"])
df = df.sort_values("timestamps").reset_index(drop=True)

df = df.rename(columns={
    "timestamps": "date",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume",
    "amount": "amount"
})

# 基础列
df["vwap"] = df["amount"] / df["volume"].replace(0, np.nan)
df["ret_1"] = df["close"].pct_change(1)
df["hl_range"] = df["high"] - df["low"]

# 指标函数

def calc_rsi(series, window=14):
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=series.index).rolling(window).mean()
    roll_down = pd.Series(down, index=series.index).rolling(window).mean()
    rs = roll_up / (roll_down + 1e-12)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def calc_macd(close, fast=12, slow=26, signal=9):
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    dif = ema_fast - ema_slow
    dea = ema(dif, signal)
    macd = (dif - dea) * 2
    return dif, dea, macd

# 构建候选因子

df["ret_5"] = df["close"].pct_change(5)
df["ret_10"] = df["close"].pct_change(10)
df["ret_20"] = df["close"].pct_change(20)

df["ma_5"] = df["close"].rolling(5).mean()
df["ma_20"] = df["close"].rolling(20).mean()
df["ma_5_20_diff"] = df["ma_5"] - df["ma_20"]
df["ma_slope"] = df["ma_5"].diff()

df["rsi_14"] = calc_rsi(df["close"], 14)

dif, dea, macd_val = calc_macd(df["close"])
df["macd_dif"] = dif
df["macd_dea"] = dea
df["macd"] = macd_val

df["volatility_10"] = df["ret_1"].rolling(10).std()
df["volatility_20"] = df["ret_1"].rolling(20).std()
df["hl_vol"] = df["hl_range"] / df["close"]

df["vol_roc_5"] = df["volume"].pct_change(5)
df["price_vwap_diff"] = df["close"] / df["vwap"] - 1
df["obv"] = (np.sign(df["ret_1"].fillna(0)) * df["volume"]).cumsum()

df["body"] = (df["close"] - df["open"]).abs()
df["upper_shadow"] = df["high"] - df[["open", "close"]].max(axis=1)
df["lower_shadow"] = df[["open", "close"]].min(axis=1) - df["low"]

df["body_ratio"] = df["body"] / (df["hl_range"] + 1e-12)
df["upper_ratio"] = df["upper_shadow"] / (df["hl_range"] + 1e-12)
df["lower_ratio"] = df["lower_shadow"] / (df["hl_range"] + 1e-12)

df.replace([np.inf, -np.inf], np.nan, inplace=True)

# 截断到 2023-12-31 再算未来收益
cutoff = pd.to_datetime("2023-12-31")
df_train = df[df["date"] <= cutoff].copy()

df_train["future_ret_5"] = df_train["close"].shift(-5) / df_train["close"] - 1
df_train = df_train.dropna(subset=["future_ret_5"]).reset_index(drop=True)

selected_factors = [
    "price_vwap_diff",
    "ma_20",
    "ma_5",
    "volatility_10",
    "volatility_20",
    "upper_shadow",
    "hl_vol",
    "lower_ratio"
]

# 辅助函数

def factor_layer_stats(df_in, factor, n=5):
    df_tmp = df_in[["date", factor, "future_ret_5"]].dropna().copy()
    if df_tmp.shape[0] < n * 10:
        return None
    df_tmp["factor_bin"] = df_tmp.groupby('date')[factor].transform(lambda x: pd.qcut(x, n, labels=False, duplicates='drop'))
    daily_q = df_tmp.groupby(['date', 'factor_bin'])['future_ret_5'].mean().unstack()
    return daily_q


def rolling_spearman_ic(df_in, factor, window=120):
    df_tmp = df_in[["date", factor, "future_ret_5"]].dropna().copy()
    if df_tmp.shape[0] < window:
        return None
    fac_rank = df_tmp[factor].rank()
    ret_rank = df_tmp["future_ret_5"].rank()
    roll_ic = fac_rank.rolling(window).corr(ret_rank)
    ic_ts = pd.DataFrame({
        "date": df_tmp["date"],
        "ic": roll_ic
    }).dropna()
    return ic_ts

# 绘图与保存逻辑：对每个因子生成两张图（分位累积 Long-Short、IC 时间序列），每张图保存 PDF（无标题）和 PNG（带标题）
for fac in selected_factors:
    print(f"Processing factor: {fac}")

    # 分位日度平均收益矩阵（date x q）
    daily_q = factor_layer_stats(df_train, fac, n=5)
    if daily_q is not None and not daily_q.empty:
        # 计算 long-short（top - bottom）的日收益
        top = daily_q.iloc[:, -1]
        bottom = daily_q.iloc[:, 0]
        ls_ret = (top - bottom).fillna(0)
        cum = (1 + ls_ret).cumprod()

        # 画图
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(cum.index, cum.values, label='Long-Short')
        ax.set_ylabel('Cumulative Return')
        ax.grid(True)
        # PNG 带标题
        ax.set_title(f"Long-Short Cumulative - {fac}")
        png_path = os.path.join(output_dir, f"{fac}_longshort.png")
        fig.tight_layout()
        fig.savefig(png_path, format='png', bbox_inches='tight')

        # PDF 不要小标题 -> 清除标题后保存
        ax.set_title("")
        pdf_path = os.path.join(output_dir, f"{fac}_longshort.pdf")
        fig.tight_layout()
        fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
        plt.close(fig)

    # IC 时间序列
    ic_ts = rolling_spearman_ic(df_train, fac, window=120)
    if ic_ts is not None and not ic_ts.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(ic_ts['date'], ic_ts['ic'], label='IC')
        ax.axhline(0, linestyle='--', color='gray')
        ax.set_ylabel('Rolling Spearman IC (120d)')
        ax.grid(True)
        # PNG 带标题
        ax.set_title(f"IC Time Series - {fac}")
        png_path = os.path.join(output_dir, f"{fac}_ic_timeseries.png")
        fig.tight_layout()
        fig.savefig(png_path, format='png', bbox_inches='tight')

        # PDF 无标题
        ax.set_title("")
        pdf_path = os.path.join(output_dir, f"{fac}_ic_timeseries.pdf")
        fig.tight_layout()
        fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
        plt.close(fig)

print('\nAll done. Files saved to:', output_dir)
