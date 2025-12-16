"""
script/pre/visualize_q_params.py

用于可视化 ConvictionFilterStrategy 策略中 q_exit 和 q_half 参数与总收益的关系。
固定 w_pred=0.78, w_alpha=0.22
"""

import os
import math
import pandas as pd
import numpy as np
import backtrader as bt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# =========================
# 0. 路径 & 全局参数
# =========================

# script/pre/visualize_q_params.py -> script/pre -> script -> root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
XGB_OUT_DIR = os.path.join(BASE_DIR, "ml_model", "output", "xgboost")
FACTOR_IC_PATH = os.path.join(BASE_DIR, "alpha_factor", "alpha_factor_ic_ranking.csv")

TRAIN_START = pd.Timestamp("2018-01-01")
TRAIN_END = pd.Timestamp("2023-12-31")
BACKTEST_START = pd.Timestamp("2024-01-01")
BACKTEST_END = pd.Timestamp("2025-04-24")

INITIAL_CASH = 100000.0
FEE_RATE = 0.0  # 不考虑手续费

# =========================
# 1. 数据准备
# =========================

def build_alpha_score(df: pd.DataFrame, ic_path: str = FACTOR_IC_PATH) -> pd.Series:
    if not os.path.exists(ic_path):
        return pd.Series(0.0, index=df.index)

    ic_df = pd.read_csv(ic_path)
    usable_factors = {"price_vwap_diff", "ma_5", "ma_20", "volatility_10", "volatility_20"}
    ic_df = ic_df[ic_df["factor"].isin(usable_factors)]

    if ic_df.empty:
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

def load_xgb_results_and_features():
    train_path = os.path.join(XGB_OUT_DIR, "train_results.csv")
    valid_path = os.path.join(XGB_OUT_DIR, "valid_results.csv")
    test_path = os.path.join(XGB_OUT_DIR, "test_results.csv")

    if not (os.path.exists(train_path) and os.path.exists(valid_path) and os.path.exists(test_path)):
        raise FileNotFoundError("找不到 XGBoost train/valid/test 结果")

    train_df = pd.read_csv(train_path)
    valid_df = pd.read_csv(valid_path)
    test_df = pd.read_csv(test_path)

    df_all = pd.concat([train_df, valid_df, test_df], ignore_index=True)
    df_all["date"] = pd.to_datetime(df_all["timestamp"])
    df_all = df_all.sort_values("date").reset_index(drop=True)
    df_all = df_all.set_index("date")

    df_all["true_close_prev"] = df_all["true_close"].shift(1)
    df_all["r_true"] = df_all["true_close"] / df_all["true_close_prev"] - 1.0
    df_all["r_pred"] = df_all["pred_close"] / df_all["true_close_prev"] - 1.0

    df_all["open"] = df_all["true_open"]
    df_all["high"] = df_all["true_high"]
    df_all["low"] = df_all["true_low"]
    df_all["close"] = df_all["true_close"]
    df_all["volume"] = df_all["true_volume"]

    df_all["vwap"] = df_all["true_amount"] / (df_all["true_volume"] + 1e-8)
    df_all["price_vwap_diff"] = df_all["true_close"] - df_all["vwap"]

    df_all["ma_5"] = df_all["true_close"].rolling(5, min_periods=3).mean()
    df_all["ma_20"] = df_all["true_close"].rolling(20, min_periods=5).mean()
    df_all["ma_60"] = df_all["true_close"].rolling(60, min_periods=10).mean()

    df_all["volatility_10"] = df_all["r_true"].rolling(10, min_periods=5).std()
    df_all["volatility_20"] = df_all["r_true"].rolling(20, min_periods=10).std()

    mask_train = (df_all.index >= TRAIN_START) & (df_all.index <= TRAIN_END) & df_all["r_pred"].notna()
    r_pred_train = df_all.loc[mask_train, "r_pred"]
    mean_pred = float(r_pred_train.mean())
    std_pred = float(r_pred_train.std()) if r_pred_train.std() not in [0, None] else 1.0
    if std_pred == 0 or np.isnan(std_pred):
        std_pred = 1.0
    df_all["z_r_pred"] = (df_all["r_pred"] - mean_pred) / std_pred

    df_all["alpha_score"] = build_alpha_score(df_all, FACTOR_IC_PATH)

    return df_all

# =========================
# 2. Backtrader 类
# =========================

class MLFactorData(bt.feeds.PandasData):
    lines = (
        'r_true', 'r_pred', 'z_r_pred', 'alpha_score',
        'price_vwap_diff', 'ma_5', 'ma_20', 'ma_60',
        'volatility_10', 'volatility_20',
    )
    params = dict(
        datetime=None,
        open='open', high='high', low='low', close='close', volume='volume',
        openinterest=-1,
        r_true='r_true', r_pred='r_pred', z_r_pred='z_r_pred', alpha_score='alpha_score',
        price_vwap_diff='price_vwap_diff', ma_5='ma_5', ma_20='ma_20', ma_60='ma_60',
        volatility_10='volatility_10', volatility_20='volatility_20',
    )

class BaseMLStrategy(bt.Strategy):
    params = dict(name="BaseMLStrategy", fee_rate=FEE_RATE)
    def __init__(self):
        self.data_close = self.datas[0].close
        self.z_r_pred = self.datas[0].z_r_pred
        self.alpha_score = self.datas[0].alpha_score
        self.equity_list = []
        self.last_value = None
        self.total_return = 0.0

    def next(self):
        v = self.broker.getvalue()
        self.last_value = v
        self.equity_list.append(v)
        target = self._get_target_percent()
        if target is None or (isinstance(target, float) and np.isnan(target)):
            target = 0.0
        target = float(max(-1.0, min(1.0, target)))
        self.order_target_percent(data=self.datas[0], target=target)

    def _get_target_percent(self) -> float:
        return 0.0

    def stop(self):
        v = self.broker.getvalue()
        self.total_return = v / INITIAL_CASH - 1.0

class StrategyLongOnly_ConvictionFilter(BaseMLStrategy):
    params = dict(
        name="StrategyLongOnly_ConvictionFilter",
        lookback=120,
        w_pred=0.78,
        w_alpha=0.22,
        q_exit=0.05,   # bottom 5% 才空仓
        q_half=0.11,   # bottom 5~25% 半仓
    )
    def __init__(self):
        super().__init__()
        self.score_hist = []

    def _get_target_percent(self):
        z = float(self.z_r_pred[0]) if not np.isnan(self.z_r_pred[0]) else 0.0
        a = float(self.alpha_score[0]) if not np.isnan(self.alpha_score[0]) else 0.0
        score = self.p.w_pred * z + self.p.w_alpha * a
        self.score_hist.append(score)

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
# 3. 优化与可视化
# =========================

def run_optimization():
    # 1. 加载数据
    print("Loading data...")
    df_all = load_xgb_results_and_features()
    mask_bt = (df_all.index >= BACKTEST_START) & (df_all.index <= BACKTEST_END)
    df_bt = df_all.loc[mask_bt].copy()
    df_bt = df_bt.dropna(subset=["open", "high", "low", "close", "volume", "z_r_pred", "alpha_score"])
    
    if df_bt.empty:
        print("No data for backtest.")
        return

    # 2. 定义参数范围
    # q_exit: 0.00 ~ 0.20
    # q_half: 0.00 ~ 0.40
    # Constraint: q_exit < q_half
    
    q_exit_values = np.linspace(0.0, 0.20, 21) # 0.00, 0.01, ..., 0.20
    q_half_values = np.linspace(0.0, 0.40, 41) # 0.00, 0.01, ..., 0.40
    
    results = []

    print(f"Starting optimization...")

    for q_exit in q_exit_values:
        for q_half in q_half_values:
            if q_exit >= q_half:
                continue
            
            cerebro = bt.Cerebro(stdstats=False)
            cerebro.broker.setcash(INITIAL_CASH)
            cerebro.broker.setcommission(commission=FEE_RATE)
            
            data = MLFactorData(dataname=df_bt)
            cerebro.adddata(data)
            
            cerebro.addstrategy(
                StrategyLongOnly_ConvictionFilter,
                q_exit=q_exit,
                q_half=q_half
            )
            
            strats = cerebro.run()
            strat = strats[0]
            total_ret = strat.total_return
            
            results.append({
                "q_exit": q_exit,
                "q_half": q_half,
                "total_return": total_ret
            })

    # 3. 结果分析
    df_res = pd.DataFrame(results)
    
    # Save results to CSV
    output_csv = os.path.join(BASE_DIR, "script", "pre", "q_params_optimization.csv")
    df_res.to_csv(output_csv, index=False)
    print(f"Data saved to: {output_csv}")

    best_row = df_res.loc[df_res["total_return"].idxmax()]
    print("\n=== Optimization Result ===")
    print(f"Max Return: {best_row['total_return']:.4f}")
    print(f"Best q_exit: {best_row['q_exit']:.4f}")
    print(f"Best q_half: {best_row['q_half']:.4f}")

    # 4. 3D 可视化
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Create grid for plotting
    # We need to pivot the dataframe to get a grid
    # Since we have irregular data (triangular region), we might need to interpolate or just scatter plot
    # Scatter plot is easier for irregular data
    
    sc = ax.scatter(df_res["q_exit"], df_res["q_half"], df_res["total_return"], c=df_res["total_return"], cmap='viridis', marker='o')
    
    ax.set_xlabel('q_exit')
    ax.set_ylabel('q_half')
    ax.set_zlabel('Total Return')
    ax.set_title('Total Return vs q_exit & q_half')
    
    # Mark the best point
    ax.scatter(best_row["q_exit"], best_row["q_half"], best_row["total_return"], color='red', s=100, label='Max Return')
    
    # Add colorbar
    plt.colorbar(sc, label='Total Return')
    plt.legend()

    output_plot = os.path.join(BASE_DIR, "script", "pre", "q_params_optimization_3d.pdf")
    plt.savefig(output_plot, format='pdf')
    print(f"\nPlot saved to: {output_plot}")

if __name__ == "__main__":
    run_optimization()
