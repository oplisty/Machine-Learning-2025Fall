"""
script/visualize_w_alpha_w_pred.py

用于可视化 StrategyLongOnly_ConvictionFilter 策略中 w_alpha / w_pred 比值与总收益的关系。
"""

import os
import math
import pandas as pd
import numpy as np
import backtrader as bt
import matplotlib.pyplot as plt

# =========================
# 0. 路径 & 全局参数
# =========================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(os.path.abspath(__file__)))))
XGB_OUT_DIR = os.path.join(BASE_DIR, "ml_model", "output", "xgboost","horizon_3")
FACTOR_IC_PATH = os.path.join(BASE_DIR, "alpha_factor", "alpha_factor_ic_ranking.csv")
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

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
        w_pred=1,
        w_alpha=0,
        q_exit=0.05,
        q_half=0.11,
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
    # w_alpha + w_pred = 1.0
    # w_alpha in [0.0, 1.0]
    # ratio = w_alpha / w_pred
    
    # We iterate w_alpha from 0.0 to 1.0 with step 0.02
    w_alpha_values = np.linspace(0.0, 1.0, 51) # 0, 0.02, ..., 1.0
    results = []

    print(f"Starting optimization with {len(w_alpha_values)} iterations...")

    for w_alpha in w_alpha_values:
        w_pred = 1.0 - w_alpha
        
        # Avoid w_pred being exactly 0 for ratio calculation
        if w_pred < 1e-9:
            ratio = float('inf')
        else:
            ratio = w_alpha / w_pred

        cerebro = bt.Cerebro(stdstats=False)
        cerebro.broker.setcash(INITIAL_CASH)
        cerebro.broker.setcommission(commission=FEE_RATE)
        
        data = MLFactorData(dataname=df_bt)
        cerebro.adddata(data)
        
        cerebro.addstrategy(
            StrategyLongOnly_ConvictionFilter,
            w_pred=w_pred,
            w_alpha=w_alpha
        )
        
        strats = cerebro.run()
        strat = strats[0]
        total_ret = strat.total_return
        
        results.append({
            "ratio": ratio,
            "w_alpha": w_alpha,
            "w_pred": w_pred,
            "total_return": total_ret
        })

    # 3. 结果分析
    df_res = pd.DataFrame(results)

    # Save results to CSV
    output_csv = os.path.join(OUTPUT_DIR, "w_alpha_w_pred_optimization.csv")
    df_res.to_csv(output_csv, index=False)
    print(f"Data saved to: {output_csv}")

    best_row = df_res.loc[df_res["total_return"].idxmax()]
    print("\n=== Optimization Result ===")
    print(f"Max Return: {best_row['total_return']:.4f}")
    print(f"Best Ratio (w_alpha/w_pred): {best_row['ratio']}")
    print(f"Best w_alpha: {best_row['w_alpha']:.4f}")
    print(f"Best w_pred: {best_row['w_pred']:.4f}")

    # 4. 可视化 - 通用美化设置
    # 使用简洁风格，去掉网格
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.grid'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False

    # Filter out infinity for plotting
    df_plot = df_res[df_res["ratio"] != float('inf')].copy()
    
    # --- Plot 1: Original Ratio (Linear Scale) ---
    plt.figure(figsize=(10, 6))
    plt.plot(df_plot["ratio"], df_plot["total_return"], color='#1f77b4', linewidth=2)
    plt.title("Total Return vs Ratio (w_alpha / w_pred)", fontsize=14, fontweight='bold')
    plt.xlabel("Ratio (w_alpha / w_pred)", fontsize=12)
    plt.ylabel("Total Return", fontsize=12)
    
    # Mark max
    if best_row["ratio"] != float('inf'):
        plt.plot(best_row["ratio"], best_row["total_return"], color='#d62728', marker='*', markersize=15, label='Max Return')
        plt.annotate(f"Max: {best_row['total_return']:.2%}\nRatio: {best_row['ratio']:.2f}",
                     xy=(best_row["ratio"], best_row["total_return"]), 
                     xytext=(best_row["ratio"], best_row["total_return"] + 0.05),
                     arrowprops=dict(facecolor='black', arrowstyle='->'),
                     horizontalalignment='center', fontsize=10)
    else:
        print("Best result is at ratio=inf (w_alpha=1.0, w_pred=0.0)")

    plt.legend(frameon=False)
    plt.tight_layout()
    
    output_plot = os.path.join(OUTPUT_DIR, "w_alpha_w_pred_optimization.pdf")
    plt.savefig(output_plot, format='pdf')
    print(f"\nPlot saved to: {output_plot}")

    # --- Plot 2: 局部放大图 (Zoomed-in Plot) ---
    if best_row["ratio"] != float('inf'):
        df_plot_reset = df_plot.reset_index(drop=True)
        best_mask = (df_plot_reset["ratio"] - best_row["ratio"]).abs() < 1e-9
        if best_mask.any():
            best_idx_loc = df_plot_reset.index[best_mask][0]
            start_pos = max(0, best_idx_loc - 10)
            end_pos = min(len(df_plot_reset), best_idx_loc + 11)
            df_zoom = df_plot_reset.iloc[start_pos:end_pos]

            plt.figure(figsize=(10, 6))
            plt.plot(df_zoom["ratio"], df_zoom["total_return"], color='#1f77b4', linewidth=2, marker='o', markersize=6)
            plt.title("Total Return vs Ratio (Zoomed In)", fontsize=14, fontweight='bold')
            plt.xlabel("Ratio (w_alpha / w_pred)", fontsize=12)
            plt.ylabel("Total Return", fontsize=12)
            
            plt.plot(best_row["ratio"], best_row["total_return"], color='#d62728', marker='*', markersize=15, label='Max Return')
            
            y_min, y_max = df_zoom["total_return"].min(), df_zoom["total_return"].max()
            y_range = y_max - y_min if y_max != y_min else 0.01
            offset = y_range * 0.1

            plt.annotate(f"Max: {best_row['total_return']:.2%}\nRatio: {best_row['ratio']:.2f}",
                         xy=(best_row["ratio"], best_row["total_return"]), 
                         xytext=(best_row["ratio"], best_row["total_return"] + offset),
                         arrowprops=dict(facecolor='black', arrowstyle='->'),
                         horizontalalignment='center', fontsize=10)
            plt.legend(frameon=False)
            plt.tight_layout()

            output_zoom = os.path.join(OUTPUT_DIR, "w_alpha_w_pred_optimization_zoom.pdf")
            plt.savefig(output_zoom, format='pdf')
            print(f"Zoomed plot saved to: {output_zoom}")

    # --- Plot 3: Log Scale (lg(Ratio)) ---
    # Filter out ratio=0 for log scale
    df_log = df_plot[df_plot["ratio"] > 0].copy()
    if not df_log.empty:
        df_log["log_ratio"] = np.log10(df_log["ratio"])
        
        plt.figure(figsize=(10, 6))
        plt.plot(df_log["log_ratio"], df_log["total_return"], color='#2ca02c', linewidth=2)
        plt.title("Total Return vs Log10(Ratio)", fontsize=14, fontweight='bold')
        plt.xlabel("Log10(Ratio)", fontsize=12)
        plt.ylabel("Total Return", fontsize=12)
        
        # Mark max
        if best_row["ratio"] > 0 and best_row["ratio"] != float('inf'):
            log_best = np.log10(best_row["ratio"])
            plt.plot(log_best, best_row["total_return"], color='#d62728', marker='*', markersize=15, label='Max Return')
            plt.annotate(f"Max: {best_row['total_return']:.2%}\nlg(Ratio): {log_best:.2f}",
                         xy=(log_best, best_row["total_return"]), 
                         xytext=(log_best, best_row["total_return"] + 0.05),
                         arrowprops=dict(facecolor='black', arrowstyle='->'),
                         horizontalalignment='center', fontsize=10)
        
        plt.legend(frameon=False)
        plt.tight_layout()
        
        output_log = os.path.join(OUTPUT_DIR, "w_alpha_w_pred_optimization_log.pdf")
        plt.savefig(output_log, format='pdf')
        print(f"Log scale plot saved to: {output_log}")

    # 6. Log Ratio Plot
    df_log = df_res[(df_res["ratio"] > 0) & (df_res["ratio"] != float('inf'))].copy()
    if not df_log.empty:
        df_log["log_ratio"] = np.log10(df_log["ratio"])
        
        plt.figure(figsize=(10, 6))
        plt.plot(df_log["log_ratio"], df_log["total_return"], color='#2ca02c', linewidth=2, marker='o', markersize=4)
        plt.xlabel("Log10(Ratio)", fontsize=12)
        plt.ylabel("Total Return", fontsize=12)
        # No grid

        if best_row["ratio"] > 0 and best_row["ratio"] != float('inf'):
             best_log_ratio = np.log10(best_row["ratio"])
             plt.plot(best_log_ratio, best_row["total_return"], color='#d62728', marker='*', markersize=15, label='Max Return')
             plt.annotate(f"Max: {best_row['total_return']:.2%}\nLogRatio: {best_log_ratio:.2f}",
                         xy=(best_log_ratio, best_row["total_return"]), 
                         xytext=(best_log_ratio, best_row["total_return"] + 0.05),
                         arrowprops=dict(facecolor='black', arrowstyle='->'),
                         horizontalalignment='center', fontsize=10)

        plt.legend(frameon=True)
        plt.tight_layout()
        
        output_log = os.path.join(BASE_DIR, "script", "w_alpha_w_pred_optimization_log.pdf")
        plt.savefig(output_log, format='pdf')
        print(f"Log plot saved to: {output_log}")

if __name__ == "__main__":
    run_optimization()
