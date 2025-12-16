"""
bt_schemeA_ml_trend.py

方案 A：只用机器学习预测结果 + 趋势（均线），完全不用因子文件。

包含策略：
    - Benchmark_BuyAndHold           ：基准满仓买入并持有
    - StrategyML_OnlyLinear          ：只用 ML 预测收益调仓（无因子）
    - StrategyTrend_Only             ：只用价格均线趋势调仓（无 ML）
    - StrategyML_Trend_Enhance       ：ML 决定方向 + 趋势决定仓位强度（方案 A 核心）

运行方式（在项目根目录）：
    python script/bt_schemeA_ml_trend.py
"""

import os
import math
import pandas as pd
import numpy as np
import backtrader as bt

print(">>> bt_schemeA_ml_trend.py loaded from:", __file__)

# =========================
# 0. 路径 & 全局参数
# =========================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XGB_OUT_DIR = os.path.join(BASE_DIR, "ml_model", "output", "xgboost")

TRAIN_START = pd.Timestamp("2018-01-01")
TRAIN_END = pd.Timestamp("2023-12-31")
BACKTEST_START = pd.Timestamp("2024-01-01")
BACKTEST_END = pd.Timestamp("2025-04-24")

INITIAL_CASH = 100000.0
FEE_RATE = 0.001   # 单边手续费 0.1%


# =========================
# 1. 数据准备：合并预测结果 + 构造基础特征（不使用任何因子文件）
# =========================

def load_xgb_results_and_features():
    """
    读取 train/valid/test 预测结果，并合并成一个 DataFrame，
    再构造：
        - r_true：真实日收益
        - r_pred：预测日收益
        - z_r_pred：预测收益标准化（基于训练集）
        - ma_5, ma_20, ma_60：趋势指标
        - volatility_20：20 日滚动波动率（用于波动率自适应调仓）
    """
    train_path = os.path.join(XGB_OUT_DIR, "train_results.csv")
    valid_path = os.path.join(XGB_OUT_DIR, "valid_results.csv")
    test_path = os.path.join(XGB_OUT_DIR, "test_results.csv")

    if not (os.path.exists(train_path) and os.path.exists(valid_path) and os.path.exists(test_path)):
        raise FileNotFoundError(
            "找不到 XGBoost train/valid/test 结果，请检查 ml_model/output/xgboost 目录。"
        )

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

    # OHLCV
    df_all["open"] = df_all["true_open"]
    df_all["high"] = df_all["true_high"]
    df_all["low"] = df_all["true_low"]
    df_all["close"] = df_all["true_close"]
    df_all["volume"] = df_all["true_volume"]

    # 趋势指标
    df_all["ma_5"] = df_all["true_close"].rolling(5, min_periods=3).mean()
    df_all["ma_20"] = df_all["true_close"].rolling(20, min_periods=5).mean()
    df_all["ma_60"] = df_all["true_close"].rolling(60, min_periods=10).mean()

    # 波动率指标
    df_all["volatility_20"] = df_all["r_true"].rolling(20, min_periods=10).std()

    # 用训练期对 r_pred 做标准化
    mask_train = (df_all.index >= TRAIN_START) & (df_all.index <= TRAIN_END) & df_all["r_pred"].notna()
    r_pred_train = df_all.loc[mask_train, "r_pred"]
    mean_pred = r_pred_train.mean()
    std_pred = r_pred_train.std()
    if std_pred is None or std_pred == 0 or np.isnan(std_pred):
        std_pred = 1.0
    df_all["z_r_pred"] = (df_all["r_pred"] - mean_pred) / std_pred

    return df_all


# =========================
# 2. Backtrader 数据源定义
# =========================

class MLTrendData(bt.feeds.PandasData):
    """
    自定义数据源，在标准 OHLCV 的基础上增加：
        - r_true
        - r_pred
        - z_r_pred
        - ma_20
        - ma_60
        - volatility_20
    """

    lines = (
        'r_true',
        'r_pred',
        'z_r_pred',
        'ma_20',
        'ma_60',
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
        ma_20='ma_20',
        ma_60='ma_60',
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
        self.ma_20 = self.datas[0].ma_20
        self.ma_60 = self.datas[0].ma_60
        self.volatility_20 = self.datas[0].volatility_20

        self.equity_list = []
        self.ret_list = []
        self.last_value = None

        # 回撤跟踪
        self.equity_peak = None
        self.dd_list = []
        self.max_drawdown = 0.0

        # 绩效字典
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

        # 子类实现目标仓位（0~2.0）
        target_percent = self._get_target_percent()
        if target_percent is None or np.isnan(target_percent):
            target_percent = 1.0

        # 安全裁剪
        target_percent = float(max(0.0, min(2.0, target_percent)))

        # 调整到目标仓位
        self.order_target_percent(data=self.datas[0], target=target_percent)

    def _get_target_percent(self):
        """
        子类覆盖这个方法，返回当天目标仓位（0~2.0）
        """
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
# 4. 策略们
# =========================

class Benchmark_BuyAndHold(BaseMLStrategy):
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
            self.ordered = True
            return 1.0
        return 1.0


class StrategyML_OnlyLinear(BaseMLStrategy):
    """
    只用 ML 预测收益 z_r_pred 做线性调仓：
        pos_t = 1.0 + k * z_r_pred_t
    再裁剪到 [0.5, 1.5]，不考虑趋势、不考虑因子。
    """

    params = dict(
        name="StrategyML_OnlyLinear",
        k=0.25,
        warmup=20,
        z_window=200,
        fee_rate=FEE_RATE,
    )

    def __init__(self):
        super().__init__()
        self.score_hist = []

    def _get_target_percent(self):
        score = self.z_r_pred[0]
        if np.isnan(score):
            return 1.0

        self.score_hist.append(score)
        if len(self.score_hist) < self.p.warmup:
            return 1.0

        # 时间上再做一层滚动标准化，防止模型在某段时间整体偏高/偏低
        window = min(self.p.z_window, len(self.score_hist))
        recent = np.array(self.score_hist[-window:])
        m = float(np.mean(recent))
        s = float(np.std(recent))
        if s == 0 or np.isnan(s):
            z = 0.0
        else:
            z = (score - m) / s

        target = 1.0 + self.p.k * z
        target = max(0.5, min(1.5, target))
        return target


class StrategyTrend_Only(BaseMLStrategy):
    """
    只看均线趋势调仓（不使用 ML）：

        - 强多头: close > ma20 > ma60  -> 1.5 倍仓
        - 中多头: close > ma20 and close > ma60 -> 1.3
        - 弱多头: close > ma60 -> 1.1
        - 中性:   ma20 >= ma60 and close 在 ma20 附近 -> 1.0
        - 偏空:   close < ma20 <= ma60 -> 0.8
        - 弱空:   close < ma60 -> 0.6
    """

    params = dict(
        name="StrategyTrend_Only",
        fee_rate=FEE_RATE,
    )

    def __init__(self):
        super().__init__()

    def _get_target_percent(self):
        close = self.data_close[0]
        ma20 = self.ma_20[0]
        ma60 = self.ma_60[0]

        if np.isnan(close) or np.isnan(ma20) or np.isnan(ma60):
            return 1.0

        # 强多头：close > ma20 > ma60
        if (close > ma20) and (ma20 > ma60):
            return 1.5
        # 中多头：close > ma20 且 > ma60（但 ma20 不一定> ma60）
        if (close > ma20) and (close > ma60):
            return 1.3
        # 弱多头：close > ma60
        if close > ma60:
            return 1.1
        # 中性：ma20 >= ma60 且 close 在 ma20 附近（±1%）
        if (ma20 >= ma60) and (abs(close - ma20) / ma20 < 0.01):
            return 1.0
        # 偏空：close < ma20 且 ma20 <= ma60
        if (close < ma20) and (ma20 <= ma60):
            return 0.8
        # 弱空：close < ma60
        if close < ma60:
            return 0.6

        return 1.0


class StrategyML_Trend_Enhance(BaseMLStrategy):
    """
    方案 A：ML 决定方向 + 趋势决定仓位强度（不依赖因子）。

    思路：
        1. 使用 z_r_pred 作为 ML 信号：
               score = z_r_pred
               direction = sign(score)（+1 看涨，-1 看跌，0 中性）
               如果 |score| 很小，则认为是“噪声”，当成 direction=0

        2. 使用 ma20 / ma60 / close 判断趋势强度：
               强趋势多头: close > ma20 > ma60
               中趋势多头: close > ma60
               弱趋势/震荡: 其他
           对应不同的 trend_boost

        3. 综合仓位：
               base_pos:
                   如果 ML 看涨 -> base_long (~1.2)
                   如果 ML 看跌 -> base_defensive (~0.7)
                   如果 ML 中性 -> base_neutral (~0.9)

               然后：
                   pos_raw = base_pos * trend_boost

               再加一点 score 强度 overlay：
                   overlay = k_z * z_time
                   pos = pos_raw + overlay

        4. 波动率自适应：
               如果 volatility_20 很高 -> 压缩仓位（向 1.0 收缩）
               如果 volatility_20 很低 -> 略微放大仓位

        5. 最终裁剪在 [0.5, 1.8] 之间，并做仓位平滑。
    """

    params = dict(
        name="StrategyML_Trend_Enhance",
        base_long=1.2,
        base_neutral=0.9,
        base_defensive=0.7,
        k_z=0.25,
        z_window=200,
        warmup=40,
        vol_window=200,
        direction_threshold=0.2,  # |score| < 0.2 视作“没什么信号”
        max_step=0.3,             # 单日仓位最大变化
        fee_rate=FEE_RATE,
    )

    def __init__(self):
        super().__init__()
        self.score_hist = []
        self.vol_hist = []
        self.last_target = None

    def _get_target_percent(self):
        score = self.z_r_pred[0]
        close = self.data_close[0]
        ma20 = self.ma_20[0]
        ma60 = self.ma_60[0]
        vol20 = self.volatility_20[0]

        if np.isnan(score) or np.isnan(close) or np.isnan(ma20) or np.isnan(ma60):
            return 1.0

        self.score_hist.append(score)
        if not np.isnan(vol20):
            self.vol_hist.append(vol20)

        if len(self.score_hist) < self.p.warmup:
            return 1.0

        # 1) ML 信号方向（弱信号视为 0）
        if abs(score) < self.p.direction_threshold:
            direction = 0
        else:
            direction = 1 if score > 0 else -1

        # 2) 确定基础仓位（只由 ML 决定大方向）
        if direction > 0:
            base_pos = self.p.base_long        # 看涨 -> 进攻一点
        elif direction < 0:
            base_pos = self.p.base_defensive   # 看跌 -> 防守一点
        else:
            base_pos = self.p.base_neutral     # 信号弱 -> 中性

        # 3) 趋势强度：决定 trend_boost
        trend_boost = 1.0
        if (close > ma20) and (ma20 > ma60):
            # 强多头趋势
            trend_boost = 1.3
        elif (close > ma60):
            # 中等多头趋势
            trend_boost = 1.15
        elif (close > ma20) and (ma20 >= ma60):
            # 弱多头 / 震荡偏上
            trend_boost = 1.05
        elif (close < ma60) and (ma20 < ma60):
            # 中等偏空 / 走弱
            trend_boost = 0.85
        else:
            # 其他情况默认为 1.0
            trend_boost = 1.0

        # 初始仓位：由 ML 方向 + 趋势共同决定
        pos_raw = base_pos * trend_boost

        # 4) 时间方向再做一层 z-score，决定 overlay 强度
        window = min(self.p.z_window, len(self.score_hist))
        recent_scores = np.array(self.score_hist[-window:])
        m = float(np.mean(recent_scores))
        s = float(np.std(recent_scores))
        if s == 0 or np.isnan(s):
            z_time = 0.0
        else:
            z_time = (score - m) / s

        overlay = self.p.k_z * z_time
        pos = pos_raw + overlay

        # 5) 波动率自适应：高波动收缩，低波动略放大
        if len(self.vol_hist) >= 60 and not np.isnan(vol20):
            recent_vol = np.array(self.vol_hist[-self.p.vol_window:])
            vol_q30 = float(np.quantile(recent_vol, 0.3))
            vol_q70 = float(np.quantile(recent_vol, 0.7))

            if vol20 > vol_q70 and vol20 > 0:
                # 高波动：向 1.0 收缩
                scale = (vol_q70 / vol20) ** 1.5
                pos = 1.0 + (pos - 1.0) * scale
            elif vol20 < vol_q30 and vol_q30 > 0:
                # 低波动：略放大，最多放大 25%
                scale = min(1.25, vol_q30 / max(vol20, 1e-6))
                pos = 1.0 + (pos - 1.0) * scale

        # 6) 仓位平滑：每天变化不要太猛
        if self.last_target is not None:
            delta = pos - self.last_target
            if abs(delta) > self.p.max_step:
                pos = self.last_target + math.copysign(self.p.max_step, delta)

        # 7) 最终裁剪到 [0.5, 1.8]
        pos = max(0.5, min(1.8, pos))
        self.last_target = pos
        return pos


# =========================
# 5. 运行回测并输出对比结果
# =========================

def run_backtest():
    # 1) 准备数据
    df_all = load_xgb_results_and_features()

    mask_bt = (df_all.index >= BACKTEST_START) & (df_all.index <= BACKTEST_END)
    df_bt = df_all.loc[mask_bt].copy()

    df_bt = df_bt.dropna(subset=[
        "open", "high", "low", "close", "volume",
        "r_true", "r_pred", "z_r_pred", "ma_20", "ma_60"
    ])
    if df_bt.empty:
        raise ValueError("回测区间内没有有效数据。")

    print("Backtest date range:", df_bt.index.min(), " ~ ", df_bt.index.max())
    print("Backtest days:", len(df_bt))

    strategies = [
        Benchmark_BuyAndHold,
        StrategyML_OnlyLinear,
        StrategyTrend_Only,
        StrategyML_Trend_Enhance,
    ]

    summary = []

    for strat_cls in strategies:
        cerebro = bt.Cerebro()
        cerebro.broker.setcash(INITIAL_CASH)
        cerebro.broker.setcommission(commission=FEE_RATE)

        data = MLTrendData(dataname=df_bt)
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

        out_path = os.path.join(BASE_DIR, "backtest_results_schemeA_ml_trend_summary.csv")
        df_summary.to_csv(out_path, index=False)
        print(f"\n回测汇总结果已保存至: {out_path}")
    else:
        print("⚠️ 未获取到任何策略绩效结果，请检查策略实现。")


if __name__ == "__main__":
    run_backtest()
