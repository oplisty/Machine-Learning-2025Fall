"""
bt_lever_1p2.py

目标：
    在现有单一股票（如腾讯）的基础上，构造带 1.2 倍“温和杠杆”的增强策略，
    并和以下策略做对比：

    - Benchmark_BuyAndHold          ：基准满仓持有
    - StrategySimpleScoreLinear     ：极简 signal 增强（无杠杆）
    - Levered_BuyAndHold_1p2       ：始终 1.2 倍杠杆多头
    - Levered_SimpleScoreLinear_1p2：1.2 倍为中枢 + signal 上下微调

运行方式：
    在项目根目录执行：

        python script/bt_lever_1p2.py
"""

import os
import math
import pandas as pd
import numpy as np
import backtrader as bt
from datetime import datetime

print(">>> bt_lever_1p2.py loaded from:", __file__)

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

        # 自己维护的绩效字典
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

        # 由子类实现：根据信号决定目标仓位（0~2.0）
        target_percent = self._get_target_percent()
        if target_percent is None or np.isnan(target_percent):
            target_percent = 1.0

        # 安全裁剪，避免异常仓位，这里允许到 2 倍（配合 1.2 杠杆更宽松一点）
        target_percent = float(max(0.0, min(2.0, target_percent)))

        # 调整到目标仓位
        self.order_target_percent(data=self.datas[0], target=target_percent)

    def _get_target_percent(self):
        """
        子类覆盖这个方法，返回当天目标仓位（0~2.0）
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
# 4. 策略定义
# =========================

class BuyAndHoldStrategy(BaseMLStrategy):
    """
    基准策略：在第一根 K 线满仓买入，之后一直持有 1.0 倍仓位。
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
        else:
            return 1.0


class StrategySimpleScoreLinear(BaseMLStrategy):
    """
    极简版：只用综合 score 做线性仓位映射，用来检验 alpha 质量。

    核心：
        raw_score = 0.6 * z_r_pred + 0.4 * alpha_score
        -> 时间滚动 z-score (window)
        -> position = 1 + k * Z
    """

    params = dict(
        name="StrategySimpleScoreLinear",
        k=0.2,                # Z 对仓位的放大系数
        z_window=200,         # score 时间滚动标准化窗口
        warmup=60,            # 前 warmup 天只观察，不做仓位调整
        fee_rate=FEE_RATE,
    )

    def __init__(self):
        super().__init__()
        self.score_hist = []

    def _get_target_percent(self):
        # 统一的综合 score
        score = 0.6 * self.z_r_pred[0] + 0.4 * self.alpha_score[0]
        if np.isnan(score):
            return 1.0

        self.score_hist.append(score)
        if len(self.score_hist) < self.p.warmup:
            # 前几天只用来学习分布，仓位保持 1 倍
            return 1.0

        # 时间方向滚动标准化
        window = min(self.p.z_window, len(self.score_hist))
        recent_scores = np.array(self.score_hist[-window:])
        mean_s = float(np.mean(recent_scores))
        std_s = float(np.std(recent_scores))

        if std_s == 0 or np.isnan(std_s):
            z = 0.0
        else:
            z = (score - mean_s) / std_s

        # 线性映射到仓位：1 + k * Z
        target = 1.0 + self.p.k * z
        return target


class LeveredBuyAndHold_1p2(BaseMLStrategy):
    """
    永远 1.2 倍杠杆买入并持有（不看任何 signal）：
        - 用于展示：在上涨资产上，单纯加杠杆可以提升绝对收益，但回撤也会放大。
    """

    params = dict(
        name="Levered_BuyAndHold_1p2",
        fee_rate=FEE_RATE,
    )

    def __init__(self):
        super().__init__()
        self.ordered = False

    def _get_target_percent(self):
        # 这里直接固定 1.2 倍多头
        return 1.2


class LeveredSimpleScoreLinear_1p2(BaseMLStrategy):
    """
    以 1.2 倍杠杆为中枢的 signal 增强策略：

    思路：
        - 基础仓位 base_beta = 1.2（比普通 Buy&Hold 更激进）
        - 综合 score 做时间滚动 z-score 得到 Z
        - 仓位 = base_beta + k * Z
        - 限制在 [0.4, 2.0] 范围内，避免过激

    解读：
        - 平均来看，仓位大约在 1.2 左右，比 1.0 Buy&Hold 更“贪心”；
        - 当 Z 很差时可以降到 <1.2，略微控一点风险；
        - 当 Z 很好时可以加到更高（最多 2 倍），进一步拉高收益。
    """

    params = dict(
        name="Levered_SimpleScoreLinear_1p2",
        base_beta=1.2,        # 中枢杠杆
        k=0.25,               # Z 对仓位的放大系数（比无杠杆版略大）
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

        # 以 1.2 为中枢，上下线性放大
        target = self.p.base_beta + self.p.k * z

        # 稍微宽一点的限制范围，配合 BaseMLStrategy 的 [0, 2.0]
        target = max(0.4, min(2.0, target))
        return target


# =========================
# 5. 运行回测并输出对比结果
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
        StrategySimpleScoreLinear,
        LeveredBuyAndHold_1p2,
        LeveredSimpleScoreLinear_1p2,
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

        out_path = os.path.join(BASE_DIR, "backtest_results_lever_1p2_summary.csv")
        df_summary.to_csv(out_path, index=False)
        print(f"\n回测汇总结果已保存至: {out_path}")
    else:
        print("⚠️ 未获取到任何策略绩效结果，请检查策略实现。")


if __name__ == "__main__":
    run_backtest()



'''
输出：
(25C) sin@GuanyuXindeMacBook-Air Machine-Learning-2025Fall % python ./script/bt_lever_1p2.py
>>> bt_lever_1p2.py loaded from: /Users/sin/code/Machine-Learning-2025Fall/./script/bt_lever_1p2.py
Backtest date range: 2024-01-02 00:00:00  ~  2025-04-23 00:00:00
Backtest days: 320

==============================
Running: BuyAndHoldStrategy
==============================

=== Benchmark_BuyAndHold ===
Final value      : 162216.54
Total return     : 0.6222
Annual return    : 0.4637
Annual vol       : 0.3421
Sharpe (approx)  : 1.3555
Max drawdown     : -0.2349

==============================
Running: StrategySimpleScoreLinear
==============================

=== StrategySimpleScoreLinear ===
Final value      : 157691.56
Total return     : 0.5769
Annual return    : 0.4314
Annual vol       : 0.2734
Sharpe (approx)  : 1.5782
Max drawdown     : -0.1415

==============================
Running: LeveredBuyAndHold_1p2
==============================

=== Levered_BuyAndHold_1p2 ===
Final value      : 100000.00
Total return     : 0.0000
Annual return    : 0.0000
Annual vol       : 0.0000
Sharpe (approx)  : nan
Max drawdown     : 0.0000

==============================
Running: LeveredSimpleScoreLinear_1p2
==============================

=== Levered_SimpleScoreLinear_1p2 ===
Final value      : 163674.23
Total return     : 0.6367
Annual return    : 0.4740
Annual vol       : 0.2459
Sharpe (approx)  : 1.9275
Max drawdown     : -0.1480

===== Strategy Performance Summary =====
                name  final_value  total_return  annual_return  annual_vol   sharpe  max_drawdown
    Benchmark_BuyAndHold  162216.5400      0.622165       0.463693    0.342093 1.355457     -0.234892
    StrategySimpleScoreLinear  157691.5633      0.576916       0.431443    0.273373 1.578225     -0.141457
    Levered_BuyAndHold_1p2  100000.0000      0.000000       0.000000    0.000000      NaN      0.000000
Levered_SimpleScoreLinear_1p2  163674.2312      0.636742       0.474041    0.245939 1.927474     -0.148048

回测汇总结果已保存至: /Users/sin/code/Machine-Learning-2025Fall/backtest_results_lever_1p2_summary.csv
(25C) sin@GuanyuXindeMacBook-Air Machine-Learning-2025Fall % 

出现了超过benchmark的结果,说明增强策略有效。
'''

