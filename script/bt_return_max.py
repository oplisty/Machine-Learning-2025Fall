"""
bt_return_max_contrarian.py

目标：在当前这段 2024-2025 的行情里，尽量“最大化纯收益”。

做法：
- 仍然使用你的 XGBoost 预测 + 多因子 alpha_score 构造综合信号
- 保留部分对照策略：Buy&Hold / SimpleScoreLinear / H / B / D / H++ / B++
- 新增：StrategyReturnMaxContrarianLongOnly
    * 思路：你的 rolling Z 在这段时期几乎一直偏负（偏空）
    * 标的是上涨行情
    * 因此我们用 “-tanh(Z)” 做加仓信号 —— 模型越悲观，我们越加仓（反向用模型）

在项目根目录执行：
    python script/bt_return_max_contrarian.py
"""

import os
import math
import pandas as pd
import numpy as np
import backtrader as bt
from datetime import datetime

print(">>> bt_return_max_contrarian.py loaded from:", __file__)

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
FEE_RATE = 0.001   # 单边手续费 0.1%


# =========================
# 1. 数据准备：合并预测结果 + 因子
# =========================

def load_xgb_results_and_features():
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

    # 收益
    df_all["true_close_prev"] = df_all["true_close"].shift(1)
    df_all["r_true"] = df_all["true_close"] / df_all["true_close_prev"] - 1.0
    df_all["r_pred"] = df_all["pred_close"] / df_all["true_close_prev"] - 1.0

    # OHLCV
    df_all["open"] = df_all["true_open"]
    df_all["high"] = df_all["true_high"]
    df_all["low"] = df_all["true_low"]
    df_all["close"] = df_all["true_close"]
    df_all["volume"] = df_all["true_volume"]

    # VWAP & 因子
    df_all["vwap"] = df_all["true_amount"] / (df_all["true_volume"] + 1e-8)
    df_all["price_vwap_diff"] = df_all["true_close"] - df_all["vwap"]

    df_all["ret_1"] = df_all["r_true"]
    df_all["ma_5"] = df_all["true_close"].rolling(5, min_periods=3).mean()
    df_all["ma_20"] = df_all["true_close"].rolling(20, min_periods=5).mean()
    df_all["ma_60"] = df_all["true_close"].rolling(60, min_periods=10).mean()

    df_all["volatility_10"] = df_all["ret_1"].rolling(10, min_periods=5).std()
    df_all["volatility_20"] = df_all["ret_1"].rolling(20, min_periods=10).std()

    # 训练期标准化 r_pred
    mask_train = (df_all.index >= TRAIN_START) & (df_all.index <= TRAIN_END) & df_all["r_pred"].notna()
    r_pred_train = df_all.loc[mask_train, "r_pred"]
    mean_pred = r_pred_train.mean()
    std_pred = r_pred_train.std() if r_pred_train.std() not in [0, None] else 1.0
    df_all["z_r_pred"] = (df_all["r_pred"] - mean_pred) / std_pred

    # alpha_score
    df_all["alpha_score"] = build_alpha_score(df_all, FACTOR_IC_PATH)

    return df_all


def build_alpha_score(df: pd.DataFrame,
                      ic_path: str = FACTOR_IC_PATH) -> pd.Series:
    if not os.path.exists(ic_path):
        print("⚠️ 未找到因子 IC 文件，alpha_score 全部置为 0。")
        return pd.Series(0.0, index=df.index)

    ic_df = pd.read_csv(ic_path)
    usable_factors = {
        "price_vwap_diff",
        "ma_5",
        "ma_20",
        "volatility_10",
        "volatility_20",
    }
    ic_df = ic_df[ic_df["factor"].isin(usable_factors)]

    if ic_df.empty:
        print("⚠️ IC 文件中没有可用因子，alpha_score 全部置为 0。")
        return pd.Series(0.0, index=df.index)

    factor_series = {}
    weights = {}

    for _, row in ic_df.iterrows():
        name = row["factor"]
        ic_val = row["spearman_ic"]

        if name not in df.columns:
            continue

        s = df[name]
        mean_ = s.mean()
        std_ = s.std()
        if std_ is None or std_ == 0 or np.isnan(std_):
            continue

        z = (s - mean_) / std_
        sign = 1.0 if ic_val >= 0 else -1.0
        factor_series[name] = sign * z
        weights[name] = abs(ic_val)

    if not factor_series:
        print("⚠️ 因子构造失败，alpha_score 全部置为 0。")
        return pd.Series(0.0, index=df.index)

    weight_sum = sum(weights.values()) if weights else 1.0
    alpha_score = None

    for name, series in factor_series.items():
        w = weights.get(name, 1.0) / weight_sum
        if alpha_score is None:
            alpha_score = w * series
        else:
            alpha_score = alpha_score + w * series

    m = alpha_score.mean()
    std = alpha_score.std()
    if std and not np.isnan(std):
        alpha_score = (alpha_score - m) / std

    alpha_score.name = "alpha_score"
    alpha_score.index = df.index
    return alpha_score


# =========================
# 2. Backtrader 数据源
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
# 3. 策略基类
# =========================

class BaseMLStrategy(bt.Strategy):
    params = dict(
        name="BaseMLStrategy",
        fee_rate=FEE_RATE,
    )

    def __init__(self):
        self.data_close = self.datas[0].close
        self.r_true = self.datas[0].r_true
        self.r_pred = self.datas[0].r_pred
        self.z_r_pred = self.datas[0].z_r_pred
        self.alpha_score = self.datas[0].alpha_score

        self.equity_list = []
        self.ret_list = []
        self.last_value = None

        self.equity_peak = None
        self.dd_list = []
        self.max_drawdown = 0.0

        self.perf_stats = None

    def next(self):
        value = self.broker.getvalue()

        if self.last_value is not None:
            daily_ret = (value / self.last_value) - 1.0
            self.ret_list.append(daily_ret)

        if self.equity_peak is None:
            self.equity_peak = value
        else:
            self.equity_peak = max(self.equity_peak, value)

        cur_dd = value / self.equity_peak - 1.0
        self.dd_list.append(cur_dd)
        self.max_drawdown = min(self.max_drawdown, cur_dd)

        self.last_value = value
        self.equity_list.append(value)

        target_percent = self._get_target_percent()
        if target_percent is None or np.isnan(target_percent):
            target_percent = 1.0

        target_percent = float(max(0.0, min(2.0, target_percent)))
        self.order_target_percent(data=self.datas[0], target=target_percent)

    def _get_target_percent(self):
        return 1.0

    def stop(self):
        value = self.broker.getvalue()
        total_ret = value / INITIAL_CASH - 1.0
        n_days = len(self.equity_list)
        if n_days > 0:
            annual_ret = (1 + total_ret) ** (252 / n_days) - 1.0
        else:
            annual_ret = float('nan')

        if self.ret_list:
            vol_annual = np.std(self.ret_list) * math.sqrt(252)
            sharpe = annual_ret / vol_annual if vol_annual > 0 else float('nan')
        else:
            vol_annual = float('nan')
            sharpe = float('nan')

        max_dd = self.max_drawdown

        print(f"\n=== {self.p.name} ===")
        print(f"Final value      : {value:.2f}")
        print(f"Total return     : {total_ret:.4f}")
        print(f"Annual return    : {annual_ret:.4f}")
        print(f"Annual vol       : {vol_annual:.4f}")
        print(f"Sharpe (approx)  : {sharpe:.4f}")
        print(f"Max drawdown     : {max_dd:.4f}")

        self.perf_stats = {
            "name": self.p.name,
            "final_value": float(value),
            "total_return": float(total_ret),
            "annual_return": float(annual_ret),
            "annual_vol": float(vol_annual),
            "sharpe": float(sharpe),
            "max_drawdown": float(max_dd),
        }


# =========================
# 3.1 对照用：SimpleScoreLinear
# =========================

class StrategySimpleScoreLinear(BaseMLStrategy):
    params = dict(
        name="StrategySimpleScoreLinear",
        k=0.2,
        z_window=200,
        warmup=60,
        fee_rate=FEE_RATE,
    )

    def __init__(self):
        super().__init__()
        self.score_hist = []

    def _get_target_percent(self):
        score = 0.6 * self.z_r_pred[0] + 0.4 * self.alpha_score[0]
        if np.isnan(score):
            return 1.0

        self.score_hist.append(score)
        if len(self.score_hist) < self.p.warmup:
            return 1.0

        window = min(self.p.z_window, len(self.score_hist))
        recent_scores = np.array(self.score_hist[-window:])
        mean_s = float(np.mean(recent_scores))
        std_s = float(np.std(recent_scores))

        if std_s == 0 or np.isnan(std_s):
            z = 0.0
        else:
            z = (score - mean_s) / std_s

        target = 1.0 + self.p.k * z
        return target


# =========================
# 3.2 纯收益最大化：反向 long-only 加仓
# =========================

class StrategyReturnMaxContrarianLongOnly(BaseMLStrategy):
    """
    纯收益优先 + 反向 long-only 策略：

    - 仍然使用统一 score = 0.6*z_r_pred + 0.4*alpha_score
    - 做时间滚动 z-score 得到 Z
    - 模型越“悲观”（Z 越负），我们越加仓：
          overlay_raw = -tanh(Z)
          pos_signal = max(overlay_raw, 0)
          overlay = overlay_max * pos_signal
    - 仓位 = base_beta + overlay，裁剪到 [base_beta, max_exposure]

    直白解释：
        在当前这段样本里，你的 rolling Z 几乎一直 <= 0，
        而标的在涨，所以我们反过来：
            * 模型说“很差”的时候，我们认为“可能是错杀/抄底机会”，加仓。
    """

    params = dict(
        name="StrategyReturnMaxContrarianLongOnly",
        z_window=200,
        warmup=60,
        base_beta=1.0,      # 至少等于买入持有
        overlay_max=1.0,    # 最大加仓幅度，最大仓位 ≈ base_beta + overlay_max
        max_exposure=2.0,   # 总仓位上限，这里允许到 2 倍
        fee_rate=FEE_RATE,
    )

    def __init__(self):
        super().__init__()
        self.score_hist = []

    def _get_target_percent(self):
        score = 0.6 * self.z_r_pred[0] + 0.4 * self.alpha_score[0]
        if np.isnan(score):
            return self.p.base_beta

        self.score_hist.append(score)
        if len(self.score_hist) < self.p.warmup:
            return self.p.base_beta

        window = min(self.p.z_window, len(self.score_hist))
        recent_scores = np.array(self.score_hist[-window:])
        mean_s = float(np.mean(recent_scores))
        std_s = float(np.std(recent_scores))
        if std_s == 0 or np.isnan(std_s):
            z = 0.0
        else:
            z = (score - mean_s) / std_s

        # 关键：反向使用 Z
        # Z 很负 -> -tanh(Z) 很正 -> 加仓
        overlay_raw = -math.tanh(z)
        pos_signal = max(overlay_raw, 0.0)
        overlay = self.p.overlay_max * pos_signal

        target = self.p.base_beta + overlay
        target = max(self.p.base_beta, min(self.p.max_exposure, target))
        return target


# =========================
# 4. 其它对照策略：H / B / D / H++ / B++
# =========================

class StrategyH_HighAlphaOverlay(BaseMLStrategy):
    params = dict(
        name="StrategyH_HighAlphaOverlay",
        fee_rate=FEE_RATE,
    )

    def __init__(self):
        super().__init__()
        self.score_hist = []

    def _get_target_percent(self):
        score = 0.6 * self.z_r_pred[0] + 0.4 * self.alpha_score[0]
        if np.isnan(score):
            return 1.0

        self.score_hist.append(score)
        if len(self.score_hist) < 30:
            return 1.0

        rank_pct = sum(s <= score for s in self.score_hist) / len(self.score_hist)
        score_scaled = 2.0 * (rank_pct - 0.5)
        alpha_max = 0.5
        alpha_overlay = alpha_max * score_scaled
        pos = 1.0 + alpha_overlay
        pos = max(0.5, min(1.5, pos))
        return pos


class StrategyB_BalancedEnhanced(BaseMLStrategy):
    params = dict(
        name="StrategyB_BalancedEnhanced",
        fee_rate=FEE_RATE,
    )

    def __init__(self):
        super().__init__()
        self.score_hist = []

    def _get_target_percent(self):
        score = 0.6 * self.z_r_pred[0] + 0.4 * self.alpha_score[0]
        if np.isnan(score):
            return 1.0

        self.score_hist.append(score)
        if len(self.score_hist) < 60:
            return 1.0

        qs = np.quantile(self.score_hist, [0.2, 0.8])
        q_low, q_high = qs[0], qs[1]

        if score < q_low:
            return 0.7
        elif score > q_high:
            return 1.3
        else:
            return 1.0


class StrategyD_DefensiveMeanRev(BaseMLStrategy):
    params = dict(
        name="StrategyD_DefensiveMeanRev",
        fee_rate=FEE_RATE,
    )

    def __init__(self):
        super().__init__()
        self.pv_hist = []

    def _get_target_percent(self):
        diff = self.datas[0].price_vwap_diff[0]
        combo = self.z_r_pred[0] + self.alpha_score[0]
        if np.isnan(diff) or np.isnan(combo):
            return 0.5

        self.pv_hist.append(diff)
        if len(self.pv_hist) < 30:
            return 0.5

        m = np.mean(self.pv_hist[-200:])
        s = np.std(self.pv_hist[-200:])
        if s == 0 or np.isnan(s):
            z = 0.0
        else:
            z = (diff - m) / s

        if z < -1.0 and combo > 0:
            return 0.8
        elif z > 1.0 and combo < 0:
            return 0.2
        else:
            return 0.5


class BuyAndHoldStrategy(BaseMLStrategy):
    params = dict(
        name="Benchmark_BuyAndHold",
        fee_rate=FEE_RATE,
    )

    def __init__(self):
        super().__init__()
        self.ordered = False

    def _get_target_percent(self):
        if not self.ordered:
            self.ordered = True
            return 1.0
        else:
            return 1.0


class StrategyHPP_QlibStyle(BaseMLStrategy):
    params = dict(
        name="StrategyHPP_QlibStyle",
        fee_rate=FEE_RATE,
    )

    def __init__(self):
        super().__init__()
        self.score_hist = []
        self.vol_hist = []
        self.last_target = None

        self.dd_soft = 0.10
        self.dd_hard = 0.18

    def _get_target_percent(self):
        score = 0.6 * self.z_r_pred[0] + 0.4 * self.alpha_score[0]
        vol = self.datas[0].volatility_20[0]
        close = self.datas[0].close[0]
        ma60 = self.datas[0].ma_60[0]

        if np.isnan(score):
            return 1.0

        self.score_hist.append(score)
        if not np.isnan(vol):
            self.vol_hist.append(vol)

        if len(self.score_hist) < 60:
            target = 1.0
        else:
            recent_scores = np.array(self.score_hist[-200:])
            mean_s = float(np.mean(recent_scores))
            std_s = float(np.std(recent_scores))
            if std_s == 0 or np.isnan(std_s):
                z = 0.0
            else:
                z = (score - mean_s) / std_s

            overlay_max = 0.8
            overlay = overlay_max * math.tanh(z)
            target = 1.0 + overlay

        if not np.isnan(ma60):
            if close < ma60:
                target = 1.0 + (target - 1.0) * 0.5
            else:
                target = 1.0 + (target - 1.0) * 1.1

        if len(self.vol_hist) >= 60 and not np.isnan(vol):
            recent_vol = np.array(self.vol_hist[-200:])
            vol_q50 = float(np.quantile(recent_vol, 0.5))
            vol_q80 = float(np.quantile(recent_vol, 0.8))

            if vol > vol_q80:
                scale = vol_q80 / vol if vol > 0 else 1.0
                target = 1.0 + (target - 1.0) * scale
            elif vol < vol_q50 and vol_q50 > 0:
                scale = min(1.2, vol_q50 / max(vol, 1e-6))
                target = 1.0 + (target - 1.0) * scale

        cur_equity = self.last_value if self.last_value is not None else self.broker.getvalue()
        if self.equity_peak is None:
            cur_dd = 0.0
        else:
            cur_dd = cur_equity / self.equity_peak - 1.0

        if cur_dd < -self.dd_hard:
            target = 0.5
        elif cur_dd < -self.dd_soft:
            target = 1.0 + (target - 1.0) * 0.3

        max_step = 0.25
        if self.last_target is not None:
            delta = target - self.last_target
            if abs(delta) > max_step:
                target = self.last_target + math.copysign(max_step, delta)

        target = max(0.2, min(1.8, target))
        self.last_target = target
        return target


class StrategyBPP_QlibStyle(BaseMLStrategy):
    params = dict(
        name="StrategyBPP_QlibStyle",
        fee_rate=FEE_RATE,
    )

    def __init__(self):
        super().__init__()
        self.score_hist = []
        self.vol_hist = []
        self.last_target = None

        self.dd_soft = 0.08
        self.dd_hard = 0.15

    def _get_target_percent(self):
        score = 0.6 * self.z_r_pred[0] + 0.4 * self.alpha_score[0]
        vol = self.datas[0].volatility_20[0]
        close = self.datas[0].close[0]
        ma60 = self.datas[0].ma_60[0]

        if np.isnan(score):
            return 1.0

        self.score_hist.append(score)
        if not np.isnan(vol):
            self.vol_hist.append(vol)

        if len(self.score_hist) < 60:
            target = 1.0
        else:
            recent_scores = np.array(self.score_hist[-200:])
            mean_s = float(np.mean(recent_scores))
            std_s = float(np.std(recent_scores))
            if std_s == 0 or np.isnan(std_s):
                z = 0.0
            else:
                z = (score - mean_s) / std_s

            overlay_max = 0.5
            overlay = overlay_max * math.tanh(z)
            target = 1.0 + overlay

        if not np.isnan(ma60):
            if close < ma60:
                target = 1.0 + (target - 1.0) * 0.3
            else:
                target = 1.0 + (target - 1.0) * 0.9

        if len(self.vol_hist) >= 60 and not np.isnan(vol):
            recent_vol = np.array(self.vol_hist[-200:])
            vol_q40 = float(np.quantile(recent_vol, 0.4))
            vol_q70 = float(np.quantile(recent_vol, 0.7))

            if vol > vol_q70:
                scale = (vol_q70 / vol) ** 1.5 if vol > 0 else 1.0
                target = 1.0 + (target - 1.0) * scale
            elif vol < vol_q40 and vol_q40 > 0:
                scale = min(1.1, vol_q40 / max(vol, 1e-6))
                target = 1.0 + (target - 1.0) * scale

        cur_equity = self.last_value if self.last_value is not None else self.broker.getvalue()
        if self.equity_peak is None:
            cur_dd = 0.0
        else:
            cur_dd = cur_equity / self.equity_peak - 1.0

        if cur_dd < -self.dd_hard:
            target = 0.6
        elif cur_dd < -self.dd_soft:
            target = 1.0 + (target - 1.0) * 0.2

        max_step = 0.15
        if self.last_target is not None:
            delta = target - self.last_target
            if abs(delta) > max_step:
                target = self.last_target + math.copysign(max_step, delta)

        target = max(0.8, min(1.5, target))
        self.last_target = target
        return target


# =========================
# 6. 回测 & 输出结果
# =========================

def run_backtest():
    df_all = load_xgb_results_and_features()

    mask_bt = (df_all.index >= BACKTEST_START) & (df_all.index <= BACKTEST_END)
    df_bt = df_all.loc[mask_bt].copy()

    df_bt = df_bt.dropna(subset=[
        "open", "high", "low", "close", "volume",
        "r_true", "r_pred", "z_r_pred", "alpha_score"
    ])
    if df_bt.empty:
        raise ValueError("回测区间内没有有效数据。")

    print("Backtest date range:", df_bt.index.min(), " ~ ", df_bt.index.max())
    print("Backtest days:", len(df_bt))

    strategies = [
        BuyAndHoldStrategy,
        StrategySimpleScoreLinear,
        StrategyReturnMaxContrarianLongOnly,  # 纯收益 + 反向 long-only
        StrategyH_HighAlphaOverlay,
        StrategyB_BalancedEnhanced,
        StrategyD_DefensiveMeanRev,
        StrategyHPP_QlibStyle,
        StrategyBPP_QlibStyle,
    ]

    summary = []

    for strat_cls in strategies:
        cerebro = bt.Cerebro()
        cerebro.broker.setcash(INITIAL_CASH)
        cerebro.broker.setcommission(commission=FEE_RATE)

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

        out_path = os.path.join(BASE_DIR, "backtest_results_return_max_contrarian_summary.csv")
        df_summary.to_csv(out_path, index=False)
        print(f"\n回测汇总结果已保存至: {out_path}")
    else:
        print("⚠️ 未获取到任何策略绩效结果，请检查策略实现。")


if __name__ == "__main__":
    run_backtest()
