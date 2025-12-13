import os
import pandas as pd
import matplotlib.pyplot as plt

# ===== 0) 路径设置 =====
csv_path = "alpha_factor_ic_ranking.csv"
out_dir = "output"
os.makedirs(out_dir, exist_ok=True)

# ===== 1) 读入数据 =====
df = pd.read_csv(csv_path)

# 保险起见：确保需要的列存在
need_cols = {"factor", "pearson_ic", "spearman_ic", "abs_rank_ic"}
missing = need_cols - set(df.columns)
if missing:
    raise ValueError(f"CSV缺少必要列: {missing}. 现有列: {list(df.columns)}")

df_sorted = df.sort_values("abs_rank_ic", ascending=True)

top_n = 15
df_top = df_sorted.tail(top_n).reset_index(drop=True)

# ===== 2) 图1：|Rank IC| 排序条形图 =====
fig, ax = plt.subplots()
ax.barh(df_top["factor"], df_top["abs_rank_ic"])
ax.set_xlabel("|Spearman Rank IC|")
ax.set_ylabel("Factor")

title1 = f"Top {top_n} Factors by |Rank IC|"
ax.set_title(title1)

fig.tight_layout()
png1 = os.path.join(out_dir, "top_factors_by_abs_rank_ic.png")
pdf1 = os.path.join(out_dir, "top_factors_by_abs_rank_ic.pdf")

# PNG：带标题
fig.savefig(png1, dpi=300, bbox_inches="tight")

# PDF：去掉标题（不重画，用临时清空title后保存，再恢复）
ax.set_title("")
fig.tight_layout()
fig.savefig(pdf1, bbox_inches="tight")

plt.close(fig)

# ===== 3) 图2：Pearson vs Spearman 对比条形图 =====
fig, ax = plt.subplots()
y = range(len(df_top))

ax.barh([i - 0.2 for i in y], df_top["pearson_ic"], height=0.4, label="Pearson IC")
ax.barh([i + 0.2 for i in y], df_top["spearman_ic"], height=0.4, label="Spearman Rank IC")

ax.set_yticks(list(y))
ax.set_yticklabels(df_top["factor"])
ax.axvline(0, linewidth=1)
ax.set_xlabel("IC")

title2 = f"Pearson vs Spearman IC (Top {top_n} by |Rank IC|)"
ax.set_title(title2)
ax.legend()

fig.tight_layout()
png2 = os.path.join(out_dir, "pearson_vs_spearman_ic.png")
pdf2 = os.path.join(out_dir, "pearson_vs_spearman_ic.pdf")

# PNG：带标题
fig.savefig(png2, dpi=300, bbox_inches="tight")

# PDF：去掉标题
ax.set_title("")
fig.tight_layout()
fig.savefig(pdf2, bbox_inches="tight")

plt.close(fig)

print("Saved files:")
print(" -", png1)
print(" -", pdf1)
print(" -", png2)
print(" -", pdf2)
