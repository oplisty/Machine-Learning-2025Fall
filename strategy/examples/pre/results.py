import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# 1) 原始数据
# =========================================================
data = [
    {
        "name": "Benchmark_BuyAndHold",
        "final_value": 163127.2,
        "total_return": 0.631272,
        "annual_return": 0.4701597404917721,
        "annual_vol": 0.34320482926588725,
        "sharpe": 1.3699100373891606,
        "max_drawdown": -0.23494640643608877,
    },
    {
        "name": "StrategyLongOnly_ConvictionFilter",
        "final_value": 213918.9,
        "total_return": 1.139189,
        "annual_return": 0.8199992579684132,
        "annual_vol": 0.2684250131346214,
        "sharpe": 3.054854122544699,
        "max_drawdown": -0.12929997476787425,
    },
]
df = pd.DataFrame(data).set_index("name")

bench = "Benchmark_BuyAndHold"
strat = "StrategyLongOnly_ConvictionFilter"

# =========================================================
# 2) 指标配置
# =========================================================
features = [
    "final_value",
    "total_return",
    "annual_return",
    "annual_vol",
    "sharpe",
    "max_drawdown",
]

labels = [
    "Final Value",
    "Total Return",
    "Annual Return",
    "Annual Vol (lower better)",
    "Sharpe",
    "Max Drawdown (lower better)",
]

# 哪些是“越小越好”
lower_is_better = {"annual_vol", "max_drawdown"}

# =========================================================
# 3) 投研级关键点：固定参考区间（非常重要）
#    👉 这些区间是“合理、可解释”的金融区间
# =========================================================
ranges = {
    "final_value": (100_000, 250_000),
    "total_return": (0.0, 1.5),
    "annual_return": (0.0, 1.0),
    "annual_vol": (0.0, 0.5),
    "sharpe": (0.0, 4.0),
    "max_drawdown": (0.0, 0.4),
}

# max_drawdown 用“回撤幅度（正数）”
df2 = df.copy()
df2["max_drawdown"] = df2["max_drawdown"].abs()

# =========================================================
# 4) 归一化（基于固定区间，而不是两条曲线自己）
# =========================================================
scores = pd.DataFrame(index=df2.index, columns=features, dtype=float)

for col in features:
    lo, hi = ranges[col]

    if col in lower_is_better:
        # 越小越好
        scores[col] = (hi - df2[col]) / (hi - lo)
    else:
        # 越大越好
        scores[col] = (df2[col] - lo) / (hi - lo)

scores = scores.clip(0, 1)

# =========================================================
# 5) 雷达图准备
# =========================================================
N = len(features)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

def close(vals):
    vals = vals.tolist()
    return vals + vals[:1]

bench_vals = close(scores.loc[bench, features].values)
strat_vals = close(scores.loc[strat, features].values)

# =========================================================
# 6) 绘图（投研/PPT 风格）
# =========================================================
plt.figure(figsize=(9, 9))
ax = plt.subplot(111, polar=True)

ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=11)

ax.set_ylim(0, 1)
ax.set_yticks([0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=9)

# Strategy：实线，稍突出
ax.plot(angles, strat_vals, linewidth=3, label="Strategy")
ax.fill(angles, strat_vals, alpha=0.18)

# Benchmark：虚线，浅色
ax.plot(angles, bench_vals, linewidth=2, linestyle="--", label="Benchmark")
ax.fill(angles, bench_vals, alpha=0.10)

ax.set_title(
    "Strategic Investment Results\nStrategy vs Benchmark",
    fontsize=14,
    pad=18,
)

ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.10))
plt.tight_layout()
plt.savefig(
    "strategy_vs_benchmark_radar.pdf",
    format="pdf",
    bbox_inches="tight"
)

plt.show()

# =========================================================
# 7) 数值输出（方便你在研报里写解释）
# =========================================================
print("\n=== Normalized scores (fixed ranges) ===")
print(scores)

print("\n=== Strategy - Benchmark ===")
print((scores.loc[strat] - scores.loc[bench]).sort_values(ascending=False))
