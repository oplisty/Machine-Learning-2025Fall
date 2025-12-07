import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ==============================
# 0. 读取数据 & 基础预处理
# ==============================

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

df["vwap"] = df["amount"] / df["volume"].replace(0, np.nan)
df["ret_1"] = df["close"].pct_change(1)
df["hl_range"] = df["high"] - df["low"]

df.replace([np.inf, -np.inf], np.nan, inplace=True)

# ==============================
# 1. 指标函数
# ==============================

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

# ==============================
# 2. 构建候选因子
# ==============================

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

# ==============================
# 3. 截断到 2023-12-31，再算未来收益（防止泄漏）
# ==============================

cutoff = pd.to_datetime("2023-12-31")
df_train = df[df["date"] <= cutoff].copy()

df_train["future_ret_5"] = df_train["close"].shift(-5) / df_train["close"] - 1
df_train = df_train.dropna(subset=["future_ret_5"]).reset_index(drop=True)

# ==============================
# 4. 选定 8 个推荐因子
# ==============================

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

# 输出目录
output_dir = "output/factor_plots"
os.makedirs(output_dir, exist_ok=True)

# ==============================
# 5. 分层收益函数
# ==============================

def factor_layer_stats(df_in, factor, n=5):
    df_tmp = df_in[[factor, "future_ret_5"]].dropna().copy()
    if df_tmp.shape[0] < n * 10:
        return None
    df_tmp["factor_bin"] = pd.qcut(
        df_tmp[factor],
        n,
        labels=False,
        duplicates="drop"
    )
    layer_ret = df_tmp.groupby("factor_bin")["future_ret_5"].mean()
    return layer_ret

# ==============================
# 6. IC 时间序列 Rolling 120-day
# ==============================

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

# ==============================
# 7. 批量绘制图像
# ==============================

for fac in selected_factors:
    print(f"绘制因子 {fac} ...")

    # ---------- 分层收益图 ----------
    layer_ret = factor_layer_stats(df_train, fac, n=5)
    if layer_ret is not None:
        plt.figure(figsize=(6, 4))
        x = np.arange(len(layer_ret))
        plt.bar(x, layer_ret.values)
        plt.xticks(x, [f"Q{i}" for i in range(len(layer_ret))])
        plt.axhline(0, linestyle="--")
        plt.ylabel("Mean future_ret_5")
        plt.title(f"Layered Return - {fac}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{fac}_layered_return.png"))
        plt.close()

    # ---------- IC 时间序列图 ----------
    ic_ts = rolling_spearman_ic(df_train, fac, window=120)
    if ic_ts is not None and not ic_ts.empty:
        plt.figure(figsize=(8, 4))
        plt.plot(ic_ts["date"], ic_ts["ic"])
        plt.axhline(0, linestyle="--")
        plt.ylabel("Rolling Spearman IC (120d)")
        plt.title(f"IC Time Series - {fac}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{fac}_ic_timeseries.png"))
        plt.close()

print(f"\n全部完成！图像已保存至：{output_dir}")
