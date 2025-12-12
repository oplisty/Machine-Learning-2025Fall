"""
script/bt.py

使用 Backtrader 调用你自己的预测数据 + 因子，回测以下策略：

- Benchmark_BuyAndHold                  ：买入并持有（基准）
- StrategyLongOnly_ConvictionFilter     ：不做空、无杠杆、强过滤择时（尽量不丢上涨趋势）
- StrategyLongShort_QuantileDirectional ：允许做空、多空分位数策略（收益上限更高，环境需支持做空）

运行：
    python script/bt.py

注意：
- 只使用已有预测结果 (ml_model/output/xgboost/*.csv)
  与因子 IC 排名 (alpha_factor/alpha_factor_ic_ranking.csv)
- 不重新训练模型或挖新因子
- 不考虑手续费：FEE_RATE = 0
"""

import os
import math
import pandas as pd
import numpy as np
import backtrader as bt

# =========================
# 0. 路径 & 全局参数
# =========================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XGB_OUT_DIR = os.path.join(BASE_DIR, "ml_model", "output", "xgboost")
FACTOR_IC_PATH = os.path.join(BASE_DIR, "alpha_factor", "alpha_factor_ic_ranking.csv")

TRAIN_START = pd.Timestamp("2018-01-01")
TRAIN_END = pd.Timestamp("2023-12-31")
BACKTEST_START = pd.Timestamp("2024-01-01")
BACKTEST_END = pd.Timestamp("2025-04-24")

INITIAL_CASH = 100000.0
FEE_RATE = 0.0  # 不考虑手续费


# =========================
# 1. 数据准备：合并预测结果 + 构造因子 & alpha_score
# =========================

def load_xgb_results_and_features():
    """
    读取 train/valid/test 预测结果，并合并成一个 DataFrame，
    再构造：
        - r_true：真实日收益
        - r_pred：预测日收益
        - price_vwap_diff, ma_5, ma_20, ma_60
        - volatility_10, volatility_20
        - z_r_pred：预测收益标准化（仅用训练期均值/方差）
        - alpha_score：多因子综合打分（仅使用 IC 文件里列出的、且本脚本已构造的因子）
    """
    train_path = os.path.join(XGB_OUT_DIR, "train_results.csv")
    valid_path = os.path.join(XGB_OUT_DIR, "valid_results.csv")
    test_path = os.path.join(XGB_OUT_DIR, "test_results.csv")

    if not (os.path.exists(train_path) and os.path.exists(valid_path) and os.path.exists(test_path)):
        raise FileNotFoundError("找不到 XGBoost train/valid/test 结果，请检查 ml_model/output/xgboost 目录。")

    train_df = pd.read_csv(train_path)
    valid_df = pd.read_csv(valid_path)
    test_df = pd.read_csv(test_path)

    train_df["split"] = "train"
    valid_df["split"] = "valid"
    test_df["split"] = "test"

    df_all = pd.concat([train_df, valid_df, test_df], ignore_index=True)
    df_all["date"] = pd.to_datetime(df_all["timestamp"])
    df_all = df_all.sort_values("date").reset_index(drop=True)
    df_all = df_all.set_index("date")

    # 构造真实 & 预测收益
    df_all["true_close_prev"] = df_all["true_close"].shift(1)
    df_all["r_true"] = df_all["true_close"] / df_all["true_close_prev"] - 1.0
    df_all["r_pred"] = df_all["pred_close"] / df_all["true_close_prev"] - 1.0

    # 价格/成交数据
    df_all["open"] = df_all["true_open"]
    df_all["high"] = df_all["true_high"]
    df_all["low"] = df_all["true_low"]
    df_all["close"] = df_all["true_close"]
    df_all["volume"] = df_all["true_volume"]

    # 计算 VWAP 及因子（不新增，只做派生）
    df_all["vwap"] = df_all["true_amount"] / (df_all["true_volume"] + 1e-8)
    df_all["price_vwap_diff"] = df_all["true_close"] - df_all["vwap"]

    df_all["ma_5"] = df_all["true_close"].rolling(5, min_periods=3).mean()
    df_all["ma_20"] = df_all["true_close"].rolling(20, min_periods=5).mean()
    df_all["ma_60"] = df_all["true_close"].rolling(60, min_periods=10).mean()

    df_all["volatility_10"] = df_all["r_true"].rolling(10, min_periods=5).std()
    df_all["volatility_20"] = df_all["r_true"].rolling(20, min_periods=10).std()

    # 用训练期对 r_pred 做标准化
    mask_train = (df_all.index >= TRAIN_START) & (df_all.index <= TRAIN_END) & df_all["r_pred"].notna()
    r_pred_train = df_all.loc[mask_train, "r_pred"]
    mean_pred = float(r_pred_train.mean())
    std_pred = float(r_pred_train.std()) if r_pred_train.std() not in [0, None] else 1.0
    if std_pred == 0 or np.isnan(std_pred):
        std_pred = 1.0
    df_all["z_r_pred"] = (df_all["r_pred"] - mean_pred) / std_pred

    # 构造 alpha_score（完全基于已有 IC 文件和上述因子）
    df_all["alpha_score"] = build_alpha_score(df_all, FACTOR_IC_PATH)

    return df_all


def build_alpha_score(df: pd.DataFrame, ic_path: str = FACTOR_IC_PATH) -> pd.Series:
    """
    从 alpha_factor_ic_ranking.csv 读取因子及 IC，
    对每个因子做 z-score，再按 |IC| 加权求和，并按 IC 符号决定方向。
    """
    if not os.path.exists(ic_path):
        print("⚠️ 未找到因子 IC 文件，alpha_score 全部置为 0。")
        return pd.Series(0.0, index=df.index)

    ic_df = pd.read_csv(ic_path)

    # 只允许使用脚本里已经构造的这些因子
    usable_factors = {"price_vwap_diff", "ma_5", "ma_20", "volatility_10", "volatility_20"}
    ic_df = ic_df[ic_df["factor"].isin(usable_factors)]

    if ic_df.empty:
        print("⚠️ IC 文件中没有可用因子，alpha_score 全部置为 0。")
        return pd.Series(0.0, index=df.index)

    factor_series = {}
    weights = {}

    for _, row in ic_df.iterrows():
        name = row["factor"]
        ic_val = float(row["spearman_ic"])

        if name not in df.columns:
            continue

        s = df[name]
        std_ = float(s.std()) if s.std() not in [0, None] else 0.0
        if std_ == 0 or np.isnan(std_):
            continue

        z = (s - s.mean()) / std_
        sign = 1.0 if ic_val >= 0 else -1.0
        factor_series[name] = sign * z
        weights[name] = abs(ic_val)

    if not factor_series:
        print("⚠️ 因子构造失败，alpha_score 全部置为 0。")
        return pd.Series(0.0, index=df.index)

    wsum = sum(weights.values()) if weights else 1.0
    alpha = None
    for name, series in factor_series.items():
        w = weights.get(name, 1.0) / wsum
        alpha = w * series if alpha is None else alpha + w * series

    std = float(alpha.std()) if alpha.std() not in [0, None] else 0.0
    if std != 0 and not np.isnan(std):
        alpha = (alpha - alpha.mean()) / std

    alpha.name = "alpha_score"
    alpha.index = df.index
    return alpha


# =========================
# 2. Backtrader 数据源定义
# =========================

class MLFactorData(bt.feeds.PandasData):
    lines = (
        'r_true',
        'r_pred',
        'z_r_pred',
        'alpha_score',
        'price_vwap_diff',
        'ma_5',
        'ma_20',
        'ma_60',
        'volatility_10',
        'volatility_20',
    )

    params = dict(
        datetime=None,
        open='open',
        high='high',
        low='low',
        close='close',
        volume='volume',
        openinterest=-1,

        r_true='r_true',
        r_pred='r_pred',
        z_r_pred='z_r_pred',
        alpha_score='alpha_score',
        price_vwap_diff='price_vwap_diff',
        ma_5='ma_5',
        ma_20='ma_20',
        ma_60='ma_60',
        volatility_10='volatility_10',
        volatility_20='volatility_20',
    )


# =========================
# 3. 策略基类（记录绩效）
# =========================

class BaseMLStrategy(bt.Strategy):
    params = dict(
        name="BaseMLStrategy",
        fee_rate=FEE_RATE,
    )

    def __init__(self):
        self.data_close = self.datas[0].close
        self.r_true = self.datas[0].r_true
        self.z_r_pred = self.datas[0].z_r_pred
        self.alpha_score = self.datas[0].alpha_score

        self.equity_list = []
        self.ret_list = []
        self.last_value = None

        self.equity_peak = None
        self.max_drawdown = 0.0

        self.perf_stats = None

    def next(self):
        v = self.broker.getvalue()

        if self.last_value is not None:
            self.ret_list.append(v / self.last_value - 1.0)
        self.last_value = v

        self.equity_list.append(v)
        if self.equity_peak is None:
            self.equity_peak = v
        else:
            self.equity_peak = max(self.equity_peak, v)

        self.max_drawdown = min(self.max_drawdown, v / self.equity_peak - 1.0)

        target = self._get_target_percent()
        if target is None or (isinstance(target, float) and np.isnan(target)):
            target = 0.0

        # 无杠杆：限制在 [-1, 1]（做空版会用到 -1）
        target = float(max(-1.0, min(1.0, target)))

        self.order_target_percent(data=self.datas[0], target=target)

    def _get_target_percent(self) -> float:
        return 0.0

    def stop(self):
        v = self.broker.getvalue()
        total = v / INITIAL_CASH - 1.0
        n = len(self.equity_list)
        ann = (1 + total) ** (252 / n) - 1.0 if n > 0 else float('nan')

        vol = np.std(self.ret_list) * math.sqrt(252) if self.ret_list else float('nan')
        sharpe = ann / vol if vol and vol > 0 else float('nan')

        print(f"\n=== {self.p.name} ===")
        print(f"Final value      : {v:.2f}")
        print(f"Total return     : {total:.4f}")
        print(f"Annual return    : {ann:.4f}")
        print(f"Annual vol       : {vol:.4f}")
        print(f"Sharpe (approx)  : {sharpe:.4f}")
        print(f"Max drawdown     : {self.max_drawdown:.4f}")

        self.perf_stats = dict(
            name=self.p.name,
            final_value=float(v),
            total_return=float(total),
            annual_return=float(ann),
            annual_vol=float(vol),
            sharpe=float(sharpe),
            max_drawdown=float(self.max_drawdown),
        )


# =========================
# 4. 策略 1：基准（买入并持有）
# =========================

class BuyAndHoldStrategy(BaseMLStrategy):
    params = dict(name="Benchmark_BuyAndHold")

    def __init__(self):
        super().__init__()
        self.ordered = False

    def _get_target_percent(self):
        if not self.ordered:
            self.ordered = True
            return 1.0
        return 1.0


# =========================
# 5. 策略 A：不做空（Long-only 强过滤）
# =========================

class StrategyLongOnly_ConvictionFilter(BaseMLStrategy):
    """
    仓位 ∈ {0.0, 0.5, 1.0}
    核心：尽量保持满仓，只有在极弱信号时退出，避免上涨行情跑输基准。
    """
    params = dict(
        name="StrategyLongOnly_ConvictionFilter",
        lookback=120,
        w_pred=0.65,
        w_alpha=0.35,
        q_exit=0.05,   # bottom 5% 才空仓
        q_half=0.25,   # bottom 5~25% 半仓
    )

    def __init__(self):
        super().__init__()
        self.score_hist = []

    def _get_target_percent(self):
        z = float(self.z_r_pred[0]) if not np.isnan(self.z_r_pred[0]) else 0.0
        a = float(self.alpha_score[0]) if not np.isnan(self.alpha_score[0]) else 0.0
        score = self.p.w_pred * z + self.p.w_alpha * a
        self.score_hist.append(score)

        # 前期阈值不稳：为了不丢趋势，先满仓
        if len(self.score_hist) < max(30, self.p.lookback // 2):
            return 1.0

        hist = np.array(self.score_hist[-self.p.lookback:])
        q_exit = float(np.quantile(hist, self.p.q_exit))
        q_half = float(np.quantile(hist, self.p.q_half))

        if score < q_exit:
            return 0.0
        elif score < q_half:
            return 0.5
        else:
            return 1.0


# =========================
# 6. 策略 B：允许做空（Long-Short 分位数方向策略）
# =========================

class StrategyLongShort_QuantileDirectional(BaseMLStrategy):
    """
    允许做空、不加杠杆：仓位 ∈ {-1.0, 0.0, +1.0}
    强信号才下注：top 20% 做多，bottom 20% 做空，中间观望。
    """
    params = dict(
        name="StrategyLongShort_QuantileDirectional",
        lookback=120,
        w_pred=0.7,
        w_alpha=0.3,
        q_long=0.80,   # top 20% 做多
        q_short=0.20,  # bottom 20% 做空
    )

    def __init__(self):
        super().__init__()
        self.score_hist = []

    def _get_target_percent(self):
        z = float(self.z_r_pred[0]) if not np.isnan(self.z_r_pred[0]) else 0.0
        a = float(self.alpha_score[0]) if not np.isnan(self.alpha_score[0]) else 0.0
        score = self.p.w_pred * z + self.p.w_alpha * a
        self.score_hist.append(score)

        # 前期阈值不稳：先观望
        if len(self.score_hist) < max(30, self.p.lookback // 2):
            return 0.0

        hist = np.array(self.score_hist[-self.p.lookback:])
        q_long = float(np.quantile(hist, self.p.q_long))
        q_short = float(np.quantile(hist, self.p.q_short))

        if score >= q_long:
            return 1.0
        elif score <= q_short:
            return -1.0
        else:
            return 0.0


# =========================
# 7. 运行回测并输出对比结果
# =========================

def run_backtest():
    # 1) 准备数据
    df_all = load_xgb_results_and_features()

    mask_bt = (df_all.index >= BACKTEST_START) & (df_all.index <= BACKTEST_END)
    df_bt = df_all.loc[mask_bt].copy()

    df_bt = df_bt.dropna(subset=[
        "open", "high", "low", "close", "volume",
        "r_true", "z_r_pred", "alpha_score"
    ])

    if df_bt.empty:
        raise ValueError("回测区间内没有有效数据。")

    print("Backtest date range:", df_bt.index.min(), " ~ ", df_bt.index.max())
    print("Backtest days:", len(df_bt))

    strategies = [
        BuyAndHoldStrategy,
        StrategyLongOnly_ConvictionFilter,
        StrategyLongShort_QuantileDirectional,
    ]

    summary = []

    for strat_cls in strategies:
        cerebro = bt.Cerebro(stdstats=False)
        cerebro.broker.setcash(INITIAL_CASH)
        cerebro.broker.setcommission(commission=FEE_RATE)

        # 尝试开启做空现金（不同 backtrader 版本可能不支持此接口）
        try:
            cerebro.broker.set_shortcash(True)
        except Exception:
            pass

        data = MLFactorData(dataname=df_bt)
        cerebro.adddata(data)
        cerebro.addstrategy(strat_cls)

        print("\n==============================")
        print(f"Running: {strat_cls.__name__}")
        print("==============================")

        results = cerebro.run(maxcpus=1)
        strat = results[0]
        if getattr(strat, "perf_stats", None) is not None:
            summary.append(strat.perf_stats)

    if summary:
        df_summary = pd.DataFrame(summary)
        print("\n===== Strategy Performance Summary =====")
        print(df_summary.to_string(index=False))

        out_path = os.path.join(BASE_DIR, "backtest_results_longonly_longshort.csv")
        df_summary.to_csv(out_path, index=False)
        print(f"\n回测汇总结果已保存至: {out_path}")


if __name__ == "__main__":
    run_backtest()
