"""
script/bt.py

Backtrader 回测 + 自动完成报告要求的评估与可视化输出（PDF 矢量图 + CSV）

策略：
- Benchmark_BuyAndHold              ：买入并持有（基准）
- ConvictionFilterStrategy          ：Long-only 强过滤择时（0/0.5/1 仓位）

输出（项目根目录）：
- backtest_results_longonly_longshort.csv          # 策略绩效汇总
- daily_Benchmark_BuyAndHold.csv                   # 基准每日：equity/nav/drawdown/position/score
- daily_Conviction_Filter_Strategy.csv             # 策略每日：equity/nav/drawdown/position/score
- model_eval_summary.csv                           # 模型 vs 真实数据评估（train/valid/test）
- score_quantile_return.csv                        # score 分位收益
- report_nav_curve.pdf                             # 净值曲线（矢量）
- report_drawdown_curve.pdf                        # 回撤曲线（矢量）
- report_position_curve.pdf                        # 仓位曲线（矢量）
- report_score_thresholds.pdf                      # score + 阈值（矢量）
- report_quantile_return.pdf                       # 分位收益图（矢量）

运行：
    python script/bt.py
"""

import os
import math
import pandas as pd
import numpy as np
import backtrader as bt
import matplotlib as mpl
import matplotlib.pyplot as plt


# =========================
# 0. 路径 & 全局参数
# =========================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
XGB_OUT_DIR = os.path.join(BASE_DIR, "ml_model", "output", "xgboost")
FACTOR_IC_PATH = os.path.join(BASE_DIR, "alpha_factor", "alpha_factor_ic_ranking.csv")

TRAIN_START = pd.Timestamp("2018-01-01")
TRAIN_END = pd.Timestamp("2023-12-31")
BACKTEST_START = pd.Timestamp("2024-01-01")
BACKTEST_END = pd.Timestamp("2025-04-24")

INITIAL_CASH = 100000.0
FEE_RATE = 0.0  # 不考虑手续费

# 让 PDF 导出字体更稳定（矢量、可编辑）
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42


# =========================
# 1. 数据准备：合并预测结果 + 构造因子 & alpha_score
# =========================

def load_xgb_results_and_features():
    """
    读取 train/valid/test 预测结果，并合并成一个 DataFrame，
    构造：
        - r_true：真实日收益
        - r_pred：预测日收益（分母用 true_close[t-1]，避免未来函数）
        - 因子：price_vwap_diff, ma_5, ma_20, ma_60, volatility_10, volatility_20
        - z_r_pred：预测收益在训练期的标准化
        - alpha_score：多因子综合打分（IC 加权）
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

    # backtrader OHLCV
    df_all["open"] = df_all["true_open"]
    df_all["high"] = df_all["true_high"]
    df_all["low"] = df_all["true_low"]
    df_all["close"] = df_all["true_close"]
    df_all["volume"] = df_all["true_volume"]

    # VWAP & 因子（只派生，不新增）
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

    # alpha_score
    df_all["alpha_score"] = build_alpha_score(df_all, FACTOR_IC_PATH)

    return df_all


def build_alpha_score(df: pd.DataFrame, ic_path: str = FACTOR_IC_PATH) -> pd.Series:
    """
    从 alpha_factor_ic_ranking.csv 读取因子及 IC，
    对因子序列 z-score，再按 |IC| 加权求和，并按 IC 符号决定方向。
    """
    if not os.path.exists(ic_path):
        print("⚠️ 未找到因子 IC 文件，alpha_score 全部置为 0。")
        return pd.Series(0.0, index=df.index)

    ic_df = pd.read_csv(ic_path)

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
# 3. 策略基类（记录绩效 + 记录每日序列）
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

        self.ret_list = []
        self.last_value = None
        self.equity_peak = None
        self.max_drawdown = 0.0
        self.perf_stats = None

        # 每日序列（满足“可视化展示”）
        self.date_list = []
        self.equity_list = []
        self.pos_list = []
        self.score_list = []       # 记录 score（若子类提供 last_score）
        self.qexit_list = []       # 记录动态阈值（可选）
        self.qhalf_list = []       # 记录动态阈值（可选）

    def next(self):
        v = self.broker.getvalue()

        # 日收益（按资金曲线算）
        if self.last_value is not None:
            self.ret_list.append(v / self.last_value - 1.0)
        self.last_value = v

        # 最大回撤（按资金曲线算）
        if self.equity_peak is None:
            self.equity_peak = v
        else:
            self.equity_peak = max(self.equity_peak, v)
        self.max_drawdown = min(self.max_drawdown, v / self.equity_peak - 1.0)

        # 计算目标仓位
        target = self._get_target_percent()
        if target is None or (isinstance(target, float) and np.isnan(target)):
            target = 0.0
        target = float(max(-1.0, min(1.0, target)))

        # 记录每日数据
        dt = pd.Timestamp(self.datas[0].datetime.date(0))
        self.date_list.append(dt)
        self.equity_list.append(float(v))
        self.pos_list.append(float(target))
        self.score_list.append(float(getattr(self, "last_score", np.nan)))
        self.qexit_list.append(float(getattr(self, "last_q_exit", np.nan)))
        self.qhalf_list.append(float(getattr(self, "last_q_half", np.nan)))

        # 调仓
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
# 4. 策略：基准（买入并持有）
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
# 5. 策略：不做空（Long-only 强过滤）
# =========================

class ConvictionFilterStrategy(BaseMLStrategy):
    """
    仓位 ∈ {0.0, 0.5, 1.0}
    核心：尽量保持满仓，只有在极弱信号时退出，避免上涨行情跑输基准。
    """
    params = dict(
        name="Conviction_Filter_Strategy",
        lookback=120,
        w_pred=0.78,
        w_alpha=0.22,
        q_exit=0.07,   # bottom 7% 才空仓
        q_half=0.09,   # bottom 7~9% 半仓
    )

    def __init__(self):
        super().__init__()
        self.score_hist = []

    def _get_target_percent(self):
        z = float(self.z_r_pred[0]) if not np.isnan(self.z_r_pred[0]) else 0.0
        a = float(self.alpha_score[0]) if not np.isnan(self.alpha_score[0]) else 0.0
        score = self.p.w_pred * z + self.p.w_alpha * a

        # 记录 score 供 BaseMLStrategy 写入 daily
        self.last_score = float(score)

        self.score_hist.append(score)

        # 前期阈值不稳：为了不丢趋势，先满仓
        if len(self.score_hist) < max(30, self.p.lookback // 2):
            self.last_q_exit = np.nan
            self.last_q_half = np.nan
            return 1.0

        hist = np.array(self.score_hist[-self.p.lookback:])
        q_exit_v = float(np.quantile(hist, self.p.q_exit))
        q_half_v = float(np.quantile(hist, self.p.q_half))

        # 记录阈值（用于画 score + 阈值）
        self.last_q_exit = q_exit_v
        self.last_q_half = q_half_v

        if score < q_exit_v:
            return 0.0
        elif score < q_half_v:
            return 0.5
        else:
            return 1.0


# =========================
# 6. 评估与绘图（满足作业/报告第(5)点）
# =========================

def _safe_dropna(df, cols):
    return df.dropna(subset=[c for c in cols if c in df.columns]).copy()


def eval_model_vs_real(df_all: pd.DataFrame):
    """
    模型预测 vs 真实数据评估（train/valid/test 分开）：
    - MAE / RMSE: pred_close vs true_close
    - Spearman IC: r_pred vs r_true
    - Direction Acc: sign(r_pred) == sign(r_true)
    """
    def _one(split_name, d):
        d = _safe_dropna(d, ["true_close", "pred_close", "r_true", "r_pred"])
        if d.empty:
            return dict(split=split_name, mae=np.nan, rmse=np.nan, spearman_ic=np.nan, direction_acc=np.nan)

        err = d["pred_close"] - d["true_close"]
        mae = float(err.abs().mean())
        rmse = float(np.sqrt((err * err).mean()))
        ic = float(d["r_pred"].corr(d["r_true"], method="spearman"))
        direction_acc = float((np.sign(d["r_pred"]) == np.sign(d["r_true"])).mean())
        return dict(split=split_name, mae=mae, rmse=rmse, spearman_ic=ic, direction_acc=direction_acc)

    rows = []
    for sp in ["train", "valid", "test"]:
        rows.append(_one(sp, df_all[df_all["split"] == sp]))

    df_eval = pd.DataFrame(rows)
    out_path = os.path.join(BASE_DIR, "model_eval_summary.csv")
    df_eval.to_csv(out_path, index=False)
    print(f"\n[OK] Saved model eval summary: {out_path}")
    print(df_eval.to_string(index=False))
    return df_eval


def quantile_test_score_vs_return(df_bt: pd.DataFrame, w_pred: float, w_alpha: float):
    """
    分位数检验：score 分位（Q1..Q10） -> 平均真实收益 r_true
    用来证明“投资结果与实际数据”的对应关系（信号有效性）
    """
    d = df_bt.copy()
    d["score"] = w_pred * d["z_r_pred"] + w_alpha * d["alpha_score"]
    d = _safe_dropna(d, ["score", "r_true"])
    if d.empty:
        raise ValueError("分位检验：没有足够数据（score/r_true NaN 太多）。")

    d["q"] = pd.qcut(d["score"], 10, labels=False, duplicates="drop") + 1
    qret = d.groupby("q")["r_true"].mean()

    out_csv = os.path.join(BASE_DIR, "score_quantile_return.csv")
    qret.to_csv(out_csv)
    print(f"\n[OK] Saved quantile returns: {out_csv}")
    return qret


def save_report_plots(daily_bench: pd.DataFrame, daily_strat: pd.DataFrame, qret: pd.Series):
    """
    输出报告所需矢量图（PDF）：
    - 净值曲线
    - 回撤曲线
    - 仓位曲线
    - score + 阈值
    - 分位收益图
    """
    # 1) 净值曲线
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)
    ax.plot(daily_bench.index, daily_bench["nav"], label="Benchmark")
    ax.plot(daily_strat.index, daily_strat["nav"], label="Strategy")
    ax.set_title("Net Asset Value (NAV)")
    ax.set_xlabel("Date")
    ax.set_ylabel("NAV")
    ax.legend()
    fig.tight_layout()
    out1 = os.path.join(BASE_DIR, "report_nav_curve.pdf")
    fig.savefig(out1, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {out1}")

    # 2) 回撤曲线
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)
    ax.plot(daily_bench.index, daily_bench["drawdown"], label="Benchmark")
    ax.plot(daily_strat.index, daily_strat["drawdown"], label="Strategy")
    ax.set_title("Drawdown Curve")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.legend()
    fig.tight_layout()
    out2 = os.path.join(BASE_DIR, "report_drawdown_curve.pdf")
    fig.savefig(out2, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {out2}")

    # 3) 仓位曲线（策略）
    fig = plt.figure(figsize=(10, 3.5))
    ax = fig.add_subplot(111)
    ax.step(daily_strat.index, daily_strat["position"], where="post")
    ax.set_title("Strategy Position (0/0.5/1)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Position")
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    out3 = os.path.join(BASE_DIR, "report_position_curve.pdf")
    fig.savefig(out3, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {out3}")

    # 4) score + 动态阈值（策略）
    if ("score" in daily_strat.columns) and ("q_exit" in daily_strat.columns) and ("q_half" in daily_strat.columns):
        fig = plt.figure(figsize=(10, 4))
        ax = fig.add_subplot(111)
        ax.plot(daily_strat.index, daily_strat["score"], label="score")
        ax.plot(daily_strat.index, daily_strat["q_exit"], label="q_exit(threshold)", linestyle="--")
        ax.plot(daily_strat.index, daily_strat["q_half"], label="q_half(threshold)", linestyle="--")
        ax.set_title("Score & Rolling Quantile Thresholds")
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        ax.legend()
        fig.tight_layout()
        out4 = os.path.join(BASE_DIR, "report_score_thresholds.pdf")
        fig.savefig(out4, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] Saved: {out4}")

    # 5) 分位收益图
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111)
    xs = qret.index.astype(int)
    ax.bar(xs, qret.values)
    ax.set_title("Mean True Return by Score Quantile (Q1..Q10)")
    ax.set_xlabel("Score Quantile")
    ax.set_ylabel("Mean r_true")
    fig.tight_layout()
    out5 = os.path.join(BASE_DIR, "report_quantile_return.pdf")
    fig.savefig(out5, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {out5}")


# =========================
# 7. 运行回测并输出对比结果 + 评估/可视化
# =========================

def run_backtest():
    # 1) 准备数据
    df_all = load_xgb_results_and_features()

    # (5) 模型 vs 真实数据评估（train/valid/test）
    eval_model_vs_real(df_all)

    # 2) 回测区间数据
    mask_bt = (df_all.index >= BACKTEST_START) & (df_all.index <= BACKTEST_END)
    df_bt = df_all.loc[mask_bt].copy()

    df_bt = df_bt.dropna(subset=[
        "open", "high", "low", "close", "volume",
        "r_true", "z_r_pred", "alpha_score"
    ])

    if df_bt.empty:
        raise ValueError("回测区间内没有有效数据。")

    print("\nBacktest date range:", df_bt.index.min(), " ~ ", df_bt.index.max())
    print("Backtest days:", len(df_bt))

    strategies = [
        BuyAndHoldStrategy,
        ConvictionFilterStrategy,
    ]

    summary = []
    daily_map = {}  # name -> daily df

    for strat_cls in strategies:
        cerebro = bt.Cerebro(stdstats=False)
        cerebro.broker.setcash(INITIAL_CASH)
        cerebro.broker.setcommission(commission=FEE_RATE)

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

        # 保存绩效
        if getattr(strat, "perf_stats", None) is not None:
            summary.append(strat.perf_stats)

        # 导出每日序列（用于画图/复核）
        daily_df = pd.DataFrame({
            "equity": strat.equity_list,
            "position": strat.pos_list,
            "score": strat.score_list,
            "q_exit": strat.qexit_list,
            "q_half": strat.qhalf_list,
        }, index=pd.to_datetime(strat.date_list))

        daily_df.index.name = "date"
        daily_df["nav"] = daily_df["equity"] / INITIAL_CASH
        daily_df["drawdown"] = daily_df["nav"] / daily_df["nav"].cummax() - 1.0

        out_daily = os.path.join(BASE_DIR, f"daily_{strat.p.name}.csv")
        daily_df.to_csv(out_daily)
        print(f"[OK] Saved daily series: {out_daily}")

        daily_map[strat.p.name] = daily_df

    # 3) 保存策略绩效汇总
    if summary:
        df_summary = pd.DataFrame(summary)
        print("\n===== Strategy Performance Summary =====")
        print(df_summary.to_string(index=False))

        out_path = os.path.join(BASE_DIR, "backtest_results_longonly_longshort.csv")
        df_summary.to_csv(out_path, index=False)
        print(f"\n[OK] Saved performance summary: {out_path}")

    # 4) 分位检验（用 Conviction 参数保持一致）
    #    注意：这是“信号有效性”证据，满足（5）中的“与实际数据评估”
    #    这里用 ConvictionFilterStrategy 的默认权重
    w_pred = ConvictionFilterStrategy.params.w_pred
    w_alpha = ConvictionFilterStrategy.params.w_alpha
    qret = quantile_test_score_vs_return(df_bt, w_pred=w_pred, w_alpha=w_alpha)

    # 5) 输出报告用矢量图（净值/回撤/仓位/阈值/分位收益）
    bench_name = BuyAndHoldStrategy.params.name
    strat_name = ConvictionFilterStrategy.params.name

    if bench_name not in daily_map or strat_name not in daily_map:
        print("⚠️ 缺少 daily 序列，无法画报告图。")
        return

    daily_bench = daily_map[bench_name].copy()
    daily_strat = daily_map[strat_name].copy()

    # 对齐日期（防止少量差异）
    idx = daily_bench.index.intersection(daily_strat.index)
    daily_bench = daily_bench.loc[idx]
    daily_strat = daily_strat.loc[idx]

    save_report_plots(daily_bench, daily_strat, qret)

    print("\n✅ Done. 已生成评估表与报告图（PDF 矢量）。")


if __name__ == "__main__":
    run_backtest()
