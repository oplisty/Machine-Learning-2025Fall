"""
bt.py

使用 Backtrader 调用你自己的预测数据 + 因子，测试多种策略：

- Benchmark_BuyAndHold        ：买入并持有
- StrategyH_HighAlphaOverlay  ：原高收益增强策略（略微调整）
- StrategyB_BalancedEnhanced  ：原均衡增强策略
- StrategyD_DefensiveMeanRev  ：原防守型均值回归策略
- StrategyHPP_QlibStyle       ：Qlib 风格的高收益增强 H++
- StrategyBPP_QlibStyle       ：Qlib 风格的均衡增强 B++

要求：
    在项目根目录执行：
        python script/bt.py

注意：
    只使用已有的预测结果 (ml_model/output/xgboost) 和因子 IC 排名
    (alpha_factor/alpha_factor_ic_ranking.csv)，不重新训练模型或挖新因子。
"""

import os
import math
import pandas as pd
import numpy as np
import backtrader as bt
from datetime import datetime

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
        - z_r_pred：预测收益标准化
        - alpha_score：多因子综合打分
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

    # 价格 / 成交数据
    df_all["open"] = df_all["true_open"]
    df_all["high"] = df_all["true_high"]
    df_all["low"] = df_all["true_low"]
    df_all["close"] = df_all["true_close"]
    df_all["volume"] = df_all["true_volume"]

    # 计算 VWAP 及因子
    df_all["vwap"] = df_all["true_amount"] / (df_all["true_volume"] + 1e-8)
    df_all["price_vwap_diff"] = df_all["true_close"] - df_all["vwap"]

    df_all["ret_1"] = df_all["r_true"]
    df_all["ma_5"] = df_all["true_close"].rolling(5, min_periods=3).mean()
    df_all["ma_20"] = df_all["true_close"].rolling(20, min_periods=5).mean()
    df_all["ma_60"] = df_all["true_close"].rolling(60, min_periods=10).mean()

    df_all["volatility_10"] = df_all["ret_1"].rolling(10, min_periods=5).std()
    df_all["volatility_20"] = df_all["ret_1"].rolling(20, min_periods=10).std()

    # 用训练期对 r_pred 做标准化
    mask_train = (df_all.index >= TRAIN_START) & (df_all.index <= TRAIN_END) & df_all["r_pred"].notna()
    r_pred_train = df_all.loc[mask_train, "r_pred"]
    mean_pred = r_pred_train.mean()
    std_pred = r_pred_train.std() if r_pred_train.std() not in [0, None] else 1.0
    df_all["z_r_pred"] = (df_all["r_pred"] - mean_pred) / std_pred

    # 构造 alpha_score（完全基于已有 IC 文件和上述因子）
    df_all["alpha_score"] = build_alpha_score(df_all, FACTOR_IC_PATH)

    return df_all


def build_alpha_score(df: pd.DataFrame,
                      ic_path: str = FACTOR_IC_PATH) -> pd.Series:
    """
    使用 alpha_factor_ic_ranking.csv 中的因子名和 IC，
    在 df 上重建对应因子序列，并组合成 alpha_score。

    使用的因子名（必须在 df 中已经构造）：
        - price_vwap_diff
        - ma_5
        - ma_20
        - volatility_10
        - volatility_20
    """
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
# 2. Backtrader 数据源定义
# =========================

class MLFactorData(bt.feeds.PandasData):
    """
    自定义数据源，在标准 OHLCV 的基础上增加：
        - r_true
        - r_pred
        - z_r_pred
        - alpha_score
        - price_vwap_diff
        - ma_5, ma_20, ma_60
        - volatility_10, volatility_20
    """

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
    """
    通用基类：记录权益曲线、计算简单绩效指标。
    子类只需要实现 self._get_target_percent() 逻辑即可。
    """

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

        # 记录回撤，用于风险评估
        self.equity_peak = None      # 历史最高权益
        self.dd_list = []            # 每日回撤
        self.max_drawdown = 0.0      # 最大回撤 (负数)

        # 不要覆盖 backtrader 自带的 self.stats
        # 用自己的字段保存绩效指标
        self.perf_stats = None

    def next(self):
        # 当前账户总价值
        value = self.broker.getvalue()

        # 记录日收益
        if self.last_value is not None:
            daily_ret = (value / self.last_value) - 1.0
            self.ret_list.append(daily_ret)

        # 更新高点和回撤
        if self.equity_peak is None:
            self.equity_peak = value
        else:
            self.equity_peak = max(self.equity_peak, value)

        cur_dd = value / self.equity_peak - 1.0  # <= 0
        self.dd_list.append(cur_dd)
        self.max_drawdown = min(self.max_drawdown, cur_dd)

        self.last_value = value
        self.equity_list.append(value)

        # 由子类实现：根据信号决定目标仓位（0~1.8）
        target_percent = self._get_target_percent()
        if target_percent is None or np.isnan(target_percent):
            target_percent = 1.0

        # 安全裁剪，避免异常仓位
        target_percent = float(max(0.0, min(1.8, target_percent)))

        # 调整到目标仓位
        self.order_target_percent(data=self.datas[0], target=target_percent)

    def _get_target_percent(self):
        """
        子类覆盖这个方法，返回当天目标仓位（0~1.8）
        """
        return 1.0  # 默认满仓

    def stop(self):
        # 运行结束时打印简单绩效
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

        max_dd = self.max_drawdown  # 负数，-0.15 表示最大回撤 15%

        print(f"\n=== {self.p.name} ===")
        print(f"Final value      : {value:.2f}")
        print(f"Total return     : {total_ret:.4f}")
        print(f"Annual return    : {annual_ret:.4f}")
        print(f"Annual vol       : {vol_annual:.4f}")
        print(f"Sharpe (approx)  : {sharpe:.4f}")
        print(f"Max drawdown     : {max_dd:.4f}")

        # 保存到自定义字段 perf_stats，避免覆盖 backtrader 的 self.stats
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
# 4. 原有策略（保留用于对比）
# =========================

class StrategyH_HighAlphaOverlay(BaseMLStrategy):
    """
    高收益偏好的增强策略（原始 H）：
    综合信号：
        score = 0.6 * z_r_pred + 0.4 * alpha_score
    仓位：
        - 使用 score 的简单百分位 rank 映射到 [-1, 1]
        - alpha_overlay ∈ [-0.5, +0.5]
        - 总仓位 = 1.0 + alpha_overlay，裁剪到 [0.5, 1.5]
    """

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
            return 1.0  # 信号缺失时，按基准处理

        # 收集近期 score 用于估计百分位
        self.score_hist.append(score)
        if len(self.score_hist) < 30:
            # 前期样本不多时，就简单设为 1.0
            return 1.0

        rank_pct = sum(s <= score for s in self.score_hist) / len(self.score_hist)  # [0,1]
        score_scaled = 2.0 * (rank_pct - 0.5)   # [-1,1]
        alpha_max = 0.5
        alpha_overlay = alpha_max * score_scaled
        pos = 1.0 + alpha_overlay
        pos = max(0.5, min(1.5, pos))
        return pos


class StrategyB_BalancedEnhanced(BaseMLStrategy):
    """
    风险收益均衡的指数增强策略（原始 B）：
    score = 0.6 * z_r_pred + 0.4 * alpha_score
    在线估计分位数：
        - score 较低 -> 0.7
        - score 中等 -> 1.0
        - score 较高 -> 1.3
    """

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
            # 前期观望，仓位靠近 1
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
    """
    防守型均值回归策略（原始 D）：
    使用 price_vwap_diff 的 z-score + 综合信号 combo = z_r_pred + alpha_score：
        - z_pv < -1.0 且 combo > 0 -> 0.8
        - z_pv >  1.0 且 combo < 0 -> 0.2
        - 否则 -> 0.5
    """

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
    """
    基准策略：在第一根 K 线满仓买入，之后一直持有。
    """

    params = dict(
        name="Benchmark_BuyAndHold",
        fee_rate=FEE_RATE,
    )

    def __init__(self):
        super().__init__()
        self.ordered = False

    def _get_target_percent(self):
        if not self.ordered:
            # 第一次进场满仓
            self.ordered = True
            return 1.0
        else:
            # 之后保持当前仓位
            return 1.0


# =========================
# 5. Qlib 风格增强策略 H++ / B++
# =========================

class StrategyHPP_QlibStyle(BaseMLStrategy):
    """
    Strategy H++ : Qlib 风格的高收益增强策略（进攻型）

    信号：
        raw_score = 0.6 * z_r_pred + 0.4 * alpha_score
        -> 时间维度 z-score -> tanh 映射到 overlay ∈ [-overlay_max, +overlay_max]

    风控：
        1) 趋势过滤：close vs ma_60
        2) 波动率风控：volatility_20 的历史分位数
        3) 最大回撤风控：软阈值 10%，硬阈值 18%
        4) 仓位平滑：每日仓位变化不超过 0.25

    总仓位：
        base_beta = 1.0
        target = base_beta + overlay
        裁剪到 [0.2, 1.8]
    """

    params = dict(
        name="StrategyHPP_QlibStyle",
        fee_rate=FEE_RATE,
    )

    def __init__(self):
        super().__init__()
        self.score_hist = []
        self.vol_hist = []
        self.last_target = None   # 用于仓位平滑

        # 回撤风控参数
        self.dd_soft = 0.10   # 软阈值：10% 回撤开始减仓
        self.dd_hard = 0.18   # 硬阈值：18% 回撤强制防御

    def _get_target_percent(self):
        score = 0.6 * self.z_r_pred[0] + 0.4 * self.alpha_score[0]
        vol = self.datas[0].volatility_20[0]
        close = self.datas[0].close[0]
        ma60 = self.datas[0].ma_60[0]

        # 信号缺失 -> 按基准 1.0 处理
        if np.isnan(score):
            return 1.0

        # 收集历史 score / vol
        self.score_hist.append(score)
        if not np.isnan(vol):
            self.vol_hist.append(vol)

        # 前 60 天用来“学习”分布，不做激进 overlay
        if len(self.score_hist) < 60:
            target = 1.0
        else:
            # 1) 时间方向标准化 (滚动 200 日)
            recent_scores = np.array(self.score_hist[-200:])
            mean_s = float(np.mean(recent_scores))
            std_s = float(np.std(recent_scores))
            if std_s == 0 or np.isnan(std_s):
                z = 0.0
            else:
                z = (score - mean_s) / std_s

            # 2) tanh 平滑映射 -> overlay
            overlay_max = 0.8      # 最大增强 ±0.8
            overlay = overlay_max * math.tanh(z)

            base_beta = 1.0
            target = base_beta + overlay   # 大致落在 [0.2, 1.8]

        # (1) 趋势过滤：close vs ma_60
        if not np.isnan(ma60):
            if close < ma60:
                # 下跌趋势：把 overlay 强度缩小一半
                target = 1.0 + (target - 1.0) * 0.5
            else:
                # 上涨趋势：允许略微放大 overlay
                target = 1.0 + (target - 1.0) * 1.1

        # (2) 波动率风控：volatility_20 分位数
        if len(self.vol_hist) >= 60 and not np.isnan(vol):
            recent_vol = np.array(self.vol_hist[-200:])
            vol_q50 = float(np.quantile(recent_vol, 0.5))
            vol_q80 = float(np.quantile(recent_vol, 0.8))

            if vol > vol_q80:
                # 高波动：把 overlay 收缩到原来的 (q80 / vol)
                scale = vol_q80 / vol if vol > 0 else 1.0
                target = 1.0 + (target - 1.0) * scale
            elif vol < vol_q50 and vol_q50 > 0:
                # 很低波动：适度放大 overlay（最多 20%）
                scale = min(1.2, vol_q50 / max(vol, 1e-6))
                target = 1.0 + (target - 1.0) * scale

        # (3) 最大回撤风控
        cur_equity = self.last_value if self.last_value is not None else self.broker.getvalue()
        if self.equity_peak is None:
            cur_dd = 0.0
        else:
            cur_dd = cur_equity / self.equity_peak - 1.0    # <= 0

        if cur_dd < -self.dd_hard:
            # 硬回撤阈值：强制降到防御仓位（例如 0.5）
            target = 0.5
        elif cur_dd < -self.dd_soft:
            # 软回撤阈值：减弱 overlay，向 1.0 收缩
            target = 1.0 + (target - 1.0) * 0.3

        # (4) 仓位平滑：每天仓位不要跳太猛
        max_step = 0.25   # 单日仓位变化上限
        if self.last_target is not None:
            delta = target - self.last_target
            if abs(delta) > max_step:
                target = self.last_target + math.copysign(max_step, delta)

        # 最终仓位裁剪到 [0.2, 1.8]
        target = max(0.2, min(1.8, target))
        self.last_target = target
        return target


class StrategyBPP_QlibStyle(BaseMLStrategy):
    """
    Strategy B++ : Qlib 风格的均衡增强策略（更保守，主打 Sharpe）

    思路：
        - 与 H++ 一样，先用 score 做时间标准化 + tanh 映射
        - overlay_max 更小 (0.5)
        - 趋势过滤更保守：下跌趋势大幅收缩 overlay
        - 波动率风控更敏感：略高波动就明显减仓
        - 回撤阈值略低：软 8%，硬 15%
        - 单日仓位变化上限 0.15
    """

    params = dict(
        name="StrategyBPP_QlibStyle",
        fee_rate=FEE_RATE,
    )

    def __init__(self):
        super().__init__()
        self.score_hist = []
        self.vol_hist = []
        self.last_target = None

        self.dd_soft = 0.08   # 更低的软阈值
        self.dd_hard = 0.15   # 更低的硬阈值

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
            # 1) 时间标准化
            recent_scores = np.array(self.score_hist[-200:])
            mean_s = float(np.mean(recent_scores))
            std_s = float(np.std(recent_scores))
            if std_s == 0 or np.isnan(std_s):
                z = 0.0
            else:
                z = (score - mean_s) / std_s

            # 2) 更保守的 overlay
            overlay_max = 0.5
            overlay = overlay_max * math.tanh(z)

            base_beta = 1.0
            target = base_beta + overlay   # 大致 [0.5, 1.5]

        # (1) 趋势过滤：比 H++ 更谨慎
        if not np.isnan(ma60):
            if close < ma60:
                # 下跌趋势：大幅收缩 overlay
                target = 1.0 + (target - 1.0) * 0.3
            else:
                # 上涨趋势：略微保留增强
                target = 1.0 + (target - 1.0) * 0.9

        # (2) 波动率风控：更敏感
        if len(self.vol_hist) >= 60 and not np.isnan(vol):
            recent_vol = np.array(self.vol_hist[-200:])
            vol_q40 = float(np.quantile(recent_vol, 0.4))
            vol_q70 = float(np.quantile(recent_vol, 0.7))

            if vol > vol_q70:
                # 稍微高波动就明显减仓
                scale = (vol_q70 / vol) ** 1.5 if vol > 0 else 1.0
                target = 1.0 + (target - 1.0) * scale
            elif vol < vol_q40 and vol_q40 > 0:
                # 低波动：只轻微允许 10% 增强
                scale = min(1.1, vol_q40 / max(vol, 1e-6))
                target = 1.0 + (target - 1.0) * scale

        # (3) 最大回撤风控：更“胆小”
        cur_equity = self.last_value if self.last_value is not None else self.broker.getvalue()
        if self.equity_peak is None:
            cur_dd = 0.0
        else:
            cur_dd = cur_equity / self.equity_peak - 1.0

        if cur_dd < -self.dd_hard:
            # 直接缩到 0.6 防守仓位
            target = 0.6
        elif cur_dd < -self.dd_soft:
            # 强烈减弱 overlay
            target = 1.0 + (target - 1.0) * 0.2

        # (4) 仓位平滑：更小的步长
        max_step = 0.15
        if self.last_target is not None:
            delta = target - self.last_target
            if abs(delta) > max_step:
                target = self.last_target + math.copysign(max_step, delta)

        # 最终裁剪到 [0.8, 1.5]，保证是“均衡增强”风格
        target = max(0.8, min(1.5, target))
        self.last_target = target
        return target


# =========================
# 6. 运行回测并输出对比结果
# =========================

def run_backtest():
    # 1) 准备数据
    df_all = load_xgb_results_and_features()

    # 只选回测区间
    mask_bt = (df_all.index >= BACKTEST_START) & (df_all.index <= BACKTEST_END)
    df_bt = df_all.loc[mask_bt].copy()

    # 丢掉缺失太多的前几行
    df_bt = df_bt.dropna(subset=[
        "open", "high", "low", "close", "volume",
        "r_true", "r_pred", "z_r_pred", "alpha_score"
    ])
    if df_bt.empty:
        raise ValueError("回测区间内没有有效数据。")

    print("Backtest date range:", df_bt.index.min(), " ~ ", df_bt.index.max())
    print("Backtest days:", len(df_bt))

    # 2) 使用 Backtrader 回测多种策略
    strategies = [
        BuyAndHoldStrategy,
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
        cerebro.broker.setcommission(commission=FEE_RATE)  # 简单佣金：按市值百分比收取

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

    # 3) 汇总各策略绩效
    if summary:
        df_summary = pd.DataFrame(summary)
        print("\n===== Strategy Performance Summary =====")
        print(df_summary.to_string(index=False))

        out_path = os.path.join(BASE_DIR, "backtest_results_summary.csv")
        df_summary.to_csv(out_path, index=False)
        print(f"\n回测汇总结果已保存至: {out_path}")
    else:
        print("⚠️ 未获取到任何策略绩效结果，请检查策略实现。")


if __name__ == "__main__":
    run_backtest()
