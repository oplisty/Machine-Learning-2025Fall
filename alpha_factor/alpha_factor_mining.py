import pandas as pd
import numpy as np

# =========================================================
# 0. 读取数据 & 基础预处理
# =========================================================

# 读取你的真实数据文件 data.csv
df = pd.read_csv("data.csv")

# 时间序列排序
df["timestamps"] = pd.to_datetime(df["timestamps"])
df = df.sort_values("timestamps").reset_index(drop=True)

# 统一命名（可选）
df = df.rename(columns={
    "timestamps": "date",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume",
    "amount": "amount"
})

# 基础派生
df["vwap"] = df["amount"] / df["volume"].replace(0, np.nan)  # 避免除以 0
df["ret_1"] = df["close"].pct_change(1)                      # 日收益
df["hl_range"] = df["high"] - df["low"]                      # 高低价振幅

# 可选：把 inf 变成 NaN，方便后面 dropna
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# =========================================================
# 1. 手写技术指标函数（避免 TA-Lib 依赖）
# =========================================================

def calc_rsi(series, window=14):
    """
    简易 RSI 计算
    """
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

# =========================================================
# 2. 构建候选因子库（只用 OHLCV + amount）
# =========================================================

# --- 动量因子 ---
df["ret_5"] = df["close"].pct_change(5)
df["ret_10"] = df["close"].pct_change(10)
df["ret_20"] = df["close"].pct_change(20)

df["ma_5"] = df["close"].rolling(5).mean()
df["ma_20"] = df["close"].rolling(20).mean()
df["ma_5_20_diff"] = df["ma_5"] - df["ma_20"]     # 短长均线差
df["ma_slope"] = df["ma_5"].diff()                # 均线近似斜率

df["rsi_14"] = calc_rsi(df["close"], 14)

dif, dea, macd_val = calc_macd(df["close"])
df["macd_dif"] = dif
df["macd_dea"] = dea
df["macd"] = macd_val

# --- 波动率因子 ---
df["volatility_10"] = df["ret_1"].rolling(10).std()
df["volatility_20"] = df["ret_1"].rolling(20).std()
df["hl_vol"] = df["hl_range"] / df["close"]

# --- 量价因子 ---
df["vol_roc_5"] = df["volume"].pct_change(5)
df["price_vwap_diff"] = df["close"] / df["vwap"] - 1   # 收盘价相对均价偏离
df["obv"] = (np.sign(df["ret_1"].fillna(0)) * df["volume"]).cumsum()

# --- K 线结构因子 ---
df["body"] = (df["close"] - df["open"]).abs()
df["upper_shadow"] = df["high"] - df[["open", "close"]].max(axis=1)
df["lower_shadow"] = df[["open", "close"]].min(axis=1) - df["low"]

df["body_ratio"] = df["body"] / (df["hl_range"] + 1e-12)
df["upper_ratio"] = df["upper_shadow"] / (df["hl_range"] + 1e-12)
df["lower_ratio"] = df["lower_shadow"] / (df["hl_range"] + 1e-12)

# =========================================================
# 3. 严格避免数据泄漏：先截断日期，再算未来收益
# =========================================================

cutoff = pd.to_datetime("2023-12-31")

# 仅用 2023-12-31（含）之前的数据来做因子挖掘
df_train = df[df["date"] <= cutoff].copy()

# 在“已经截断”的数据上计算未来5日收益
df_train["future_ret_5"] = df_train["close"].shift(-5) / df_train["close"] - 1

# 最后几行未来收益为空（没有足够未来5天），丢掉
df_train = df_train.dropna(subset=["future_ret_5"]).reset_index(drop=True)

# =========================================================
# 4. 因子列表
# =========================================================

factors = [
    "ret_1","ret_5","ret_10","ret_20",
    "ma_5","ma_20","ma_5_20_diff","ma_slope",
    "rsi_14",
    "macd_dif","macd_dea","macd",
    "volatility_10","volatility_20","hl_vol",
    "vol_roc_5",
    "price_vwap_diff","obv",
    "body","upper_shadow","lower_shadow",
    "body_ratio","upper_ratio","lower_ratio"
]

# 确保没有奇怪的 inf / NaN 干扰统计
df_train.replace([np.inf, -np.inf], np.nan, inplace=True)


# =========================================================
# 4.5 因子极值处理（Winsorize）
# =========================================================

def winsorize(series, lower=0.01, upper=0.99):
    return series.clip(
        series.quantile(lower),
        series.quantile(upper)
    )

for fac in factors:
    df_train[fac] = winsorize(df_train[fac])





# =========================================================
# 5. 计算单因子 IC（Pearson / Spearman）
# =========================================================

results = []
for fac in factors:
    # 有些列可能全是 NaN 或方差过小，简单跳过
    if df_train[fac].dropna().shape[0] < 30:
        continue

    pearson_ic = df_train[fac].corr(df_train["future_ret_5"])
    spearman_ic = df_train[fac].corr(df_train["future_ret_5"], method="spearman")

    results.append({
        "factor": fac,
        "pearson_ic": pearson_ic,
        "spearman_ic": spearman_ic,
        "abs_rank_ic": abs(spearman_ic)
    })

ic_df = pd.DataFrame(results).sort_values("abs_rank_ic", ascending=False)

print("\n====================== 最终因子 IC 排名（Top 15） ======================")
print(ic_df.head(15))

# 保存到文件，方便你用 Excel / 报告查看
ic_df.to_csv("alpha_factor_ic_ranking.csv", index=False)
print("\n因子 IC 排名已保存到 alpha_factor_ic_ranking.csv")

# =========================================================
# 6. 因子分层收益检验（验证有效性）
# =========================================================

def factor_layer_test(df_in, factor, n=5):
    """
    简单分层测试：按因子值分成 n 层，计算每层的未来5日收益均值
    看是否有单调性（比如因子越大未来收益越高/越低）
    """
    df_tmp = df_in[[factor, "future_ret_5"]].dropna().copy()
    df_tmp["factor_bin"] = pd.qcut(
        df_tmp[factor],
        n,
        labels=False,
        duplicates="drop"
    )
    layer_ret = df_tmp.groupby("factor_bin")["future_ret_5"].mean()
    return layer_ret

print("\n====================== 分层收益检验（示例：Top 5 因子） ======================")
top5 = ic_df.head(5)["factor"].tolist()

for f in top5:
    print(f"\n因子 {f} 分层收益：")
    print(factor_layer_test(df_train, f))

print("\n挖掘完成：已基于 2023-12-31 之前的真实数据，选出最适合这支股票的 alpha 因子。")
