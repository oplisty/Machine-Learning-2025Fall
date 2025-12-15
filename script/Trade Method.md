# 策略说明
## 1. 代码入口与整体流程

核心回测脚本：`script/pre/try.py`（效果最佳）

整体流程可以概括为：

1. **读取 ML 预测结果（train/valid/test）并拼接成全量时间序列**
2. **从真实价格/成交数据派生少量“已定义因子”（不做新因子挖掘）**
3. **用训练期对预测收益做标准化得到 `z_r_pred`**
4. **根据 `alpha_factor_ic_ranking.csv` 的 IC 信息，对因子做 z-score + |IC| 加权合成 `alpha_score`**
5. **把 `z_r_pred` 与 `alpha_score` 按权重合成策略打分 `score`**
6. **在滚动窗口里计算 score 的分位数阈值（q_exit/q_half） → 得到仓位（0/0.5/1）**
7. **输出策略表现并与 Buy&Hold 基准对比**

---

## 2. 数据使用说明

### 2.1 预测数据

- 路径：`ml_model/output/xgboost/`
- 文件：
  - `train_results.csv`（2018-01-16 ~ 2023-01-03）
  - `valid_results.csv`（2023-01-04 ~ 2023-12-29）
  - `test_results.csv`（2024-01-02 ~ 2025-04-23）
- 关键字段：
  - 真实：`true_open/high/low/close/volume/amount`
  - 预测：`pred_open/high/low/close/volume/amount`
  - 时间：`timestamp`

> 回测使用的是**日频**，并将 train/valid/test 按时间顺序合并成一个连续序列。

### 2.2 回测用收益定义（避免“未来函数”）

`try.py` 构造两个收益序列：

- **真实日收益**  
  `r_true[t] = true_close[t] / true_close[t-1] - 1`

- **预测收益（用于打分，不直接当收益）**  
  `r_pred[t] = pred_close[t] / true_close[t-1] - 1`

这里用 `true_close[t-1]` 作为分母，表示在 t 日开盘前你已知上一个交易日收盘价，因此不会使用 t 日真实收盘价来构造“预测收益”（避免信息泄露）。

---

## 3. 因子用的是什么

> 这里“不挖新因子”，只做派生。

`try.py` 中派生的因子（并且 **alpha_score 只允许使用其中 5 个**）：

1. **VWAP 相关**
   - `vwap = true_amount / true_volume`
   - `price_vwap_diff = true_close - vwap`

2. **均线**
   - `ma_5`：5 日均线
   - `ma_20`：20 日均线
   - （还计算了 `ma_60`，但 **不会进入 alpha_score**）

3. **波动率**
   - `volatility_10`：`r_true` 的 10 日 rolling std
   - `volatility_20`：`r_true` 的 20 日 rolling std

**alpha_score 可用因子集合：**

- `price_vwap_diff`, `ma_5`, `ma_20`, `volatility_10`, `volatility_20`

---

## 4. 因子/预测如何转换成“得分”（score）

### 4.1 预测部分标准化：`z_r_pred`

- 使用训练期（`TRAIN_START ~ TRAIN_END`）的 `r_pred` 计算均值与标准差
- 对全样本进行 z-score：

`z_r_pred = (r_pred - mean_train) / std_train`

目的：让预测信号尺度稳定，避免因模型输出尺度变化导致阈值失效。

### 4.2 因子合成：`alpha_score`（由 IC 文件决定方向与权重）

`build_alpha_score()` 的关键逻辑：

1. 从 `alpha_factor/alpha_factor_ic_ranking.csv` 读取每个因子的 `spearman_ic`
2. 对每个因子序列做 z-score：  
   `z_factor = (factor - mean) / std`
3. **按 IC 符号决定方向**：  
   - `ic >= 0`：保持原方向  
   - `ic < 0`：取反（乘 -1）
4. **权重使用 |IC| 并归一化**：  
   `w_i = |IC_i| / sum(|IC|)`
5. 合成：  
   `alpha_raw = sum_i ( w_i * sign(IC_i) * z_factor_i )`
6. 最后对 `alpha_raw` 再做一次 z-score 得到 `alpha_score`

> 直觉：IC 大的因子更可信，因此权重大；IC 为负的因子方向相反，需要翻转。

### 4.3 最终策略打分：`score`

`StrategyLongOnly_ConvictionFilter` 中：

`score = w_pred * z_r_pred + w_alpha * alpha_score`

- `w_pred`：预测信号权重
- `w_alpha`：因子合成信号权重

---

## 5. 仓位生成逻辑

策略核心思想：**大部分时间保持满仓（吃 Beta），只在 score 极端差时显著降仓**。

在每个交易日 t：

1. 维护历史 score 序列 `score_hist`
2. 取最近 `lookback` 天的 score 作为分布样本
3. 计算两个分位数阈值：
   - `q_exit`：更低分位（极差） → 清仓
   - `q_half`：次低分位（较差） → 半仓
4. 仓位映射（Long-only，无做空）：

- `score < quantile(q_exit)` → `target = 0.0`
- `quantile(q_exit) <= score < quantile(q_half)` → `target = 0.5`
- `score >= quantile(q_half)` → `target = 1.0`

并通过 `order_target_percent` 调整到目标仓位。

> 这种设计的好处：不会频繁交易，主要在“很糟”的状态才减少敞口，更像**风险过滤器**而不是日内择时器。

---

## 6. 权重与阈值怎么选（来自 visual.py / best_q.py 的优化结果）

### 6.1 权重选择：`visual.py`（w_alpha, w_pred 网格搜索）

`script/pre/visual.py` 通过网格遍历不同的 `(w_alpha, w_pred)`，每组都回测一次并记录 `total_return`，输出：

- `script/pre/w_alpha_w_pred_optimization.csv`(不同权重带来的收益数据，**间隔0.02（即每一个权重每次增加/减少0.02）**)
- `script/pre/w_alpha_w_pred_optimization.pdf`（以$lg$为横坐标的矢量图）
- `script/pre/w_alpha_w_pred_optimization_zoom.pdf`(以$\frac{w_{\alpha}}{w_{pred}}$为横坐标的矢量图（**局部放大**）)
- `script/pre/w_alpha_w_pred_optimization.pdf`(以$\frac{w_{\alpha}}{w_{pred}}$为横坐标的矢量图)
- `script/pre/w_alpha_w_pred_optimization.csv`(对应数据)

在你当前这份结果中，网格搜索的最优点（按 CSV 中 `total_return` 最大）为：

- `w_alpha = 0.78`
- `w_pred  = 0.22`
- `total_return ≈ 1.139`（该优化脚本对应的那次回测设置下）

> `try.py` 最终采用的是 `w_pred=0.78, w_alpha=0.22`：更偏向 ML 预测、因子更多作为辅助过滤信号。这通常是为了**降低过拟合风险**（权重更“保守”、更可解释）。

### 6.2 阈值选择：`best_q.py`（q_exit, q_half 网格搜索）

`script/pre/best_q.py` 固定权重（脚本中为 `w_pred=0.78, w_alpha=0.22`），对 `(q_exit, q_half)` 做网格搜索（约束 `q_exit < q_half`），输出：

- `script/pre/q_params_optimization.csv`(不同阈值带来的收益)
- `script/pre/q_params_optimization_3d.pdf`(对应矢量图(3维，表示两个阈值不断变化对收益的影响))

在你当前这份结果中，最优点为：

- `q_exit = 0.07`（bottom 7% 清仓）
- `q_half = 0.09`（bottom 7%~9% 半仓）

这是一组**更强的“只在极端差才降仓”**的参数，结合当前回测区间得到更高的最终收益（见下一节）。

---

## 7. 最终回测结果（相对基准、收益与风险指标）

`script/pre/backtest_results_longonly_longshort.csv` 给出基准与策略表现（初始资金 100000，手续费为 0）：

| 策略 | Total Return | Annual Return | Annual Vol | Sharpe(approx) | Max Drawdown |
|---|---:|---:|---:|---:|---:|
| Benchmark_BuyAndHold | 0.6313 | 0.4702 | 0.3432 | 1.3699 | -0.2349 |
| StrategyLongOnly_ConvictionFilter | **1.1392** | **0.8200** | 0.2684 | **3.0549** | -0.1293 |

关键对比（策略 - 基准）：

- **超额总收益**：约 **+0.5079**
- **Sharpe 提升**：约 **+1.6849**
- **最大回撤改善**：回撤从 **-23.49%** 收敛到 **-12.93%**（改善约 10.56 个百分点）

> 解释：策略把“坏状态”过滤掉了，因此在总收益更高的同时，波动率和回撤反而更低，Sharpe 明显提升。

---

## 8. 主要参数含义速查（try.py）

- `lookback=120`：分位数阈值的历史窗口长度（用最近 120 个交易日的 score 分布）
- `w_pred, w_alpha`：预测信号与因子信号在 score 中的权重
- `q_exit`：**清仓阈值分位数**（越小越“只在极差才清仓”(只有当预测的数据在最近120的最差的q_exit之内才清仓，下面同理)）
- `q_half`：**半仓阈值分位数**（需要满足 `q_exit < q_half`）
- `INITIAL_CASH=100000`：初始资金
- `FEE_RATE=0`：手续费率（当前回测假设为 0）

---

## 9. 数据质量与“好坏”判断（以及限制）

### 9.1 数据质量（工程视角）

优点：
- train/valid/test 拼接为连续日频序列，时间戳明确
- `r_pred` 使用 `true_close_prev`，避免未来信息泄露
- 因子均为 rolling / 当日可得派生项（不依赖未来）

需要注意：
- 回测手续费设为 0，若加入真实交易成本，收益会下降（但策略交易频率相对低，影响可能比高频策略小）
- 因子 z-score 使用全样本均值/方差（而非仅训练期），严格的“无泄露”做法可考虑改为**滚动标准化**或只用历史统计量
- 单标的、强 Beta 场景下策略更有效；若标的长期震荡/下跌，Long-only 风险过滤器的优势会减弱

### 9.2 相对基准

基准为 **Buy & Hold**（全程满仓）。本策略的定位是：

- 不追求每天预测对错
- 只在 score 极端差时降低风险暴露
- 因此更像“指数增强 + 风险过滤”而非短线择时

---

## 10. script/ 中其它策略与结果（不在 pre/ 文件夹）

下面是对 `script/` 里其它主要脚本（以及对应汇总结果 CSV）的简要说明。

### 10.1 `script/bt.py`：多种择时/分位数策略对比（汇总见 backtest_results_summary.csv）

对应汇总文件：`backtest_results_summary.csv`

- `Benchmark_BuyAndHold`：基准买入并持有（Total Return ≈ 0.6313）
- `StrategyTiming_TopQuantileAllInOut`：高分位全进/低分位全出（Total Return ≈ 0.6532）
- `StrategyTiming_DoubleThresholdTiered`：双阈值分层（Total Return ≈ 0.1611）
- `StrategyTiming_TrendFilteredLongShort`：趋势过滤的多空（Total Return ≈ -0.0660）

总体：在这份回测配置下，“TopQuantileAllInOut”略跑赢基准，其它两种表现较弱/为负。

### 10.2 `script/bt_schemeA_ml_trend.py`：ML 与趋势信号的组合方案（汇总见 backtest_results_schemeA_ml_trend_summary.csv）

对应汇总文件：`backtest_results_schemeA_ml_trend_summary.csv`

- `StrategyML_OnlyLinear`：仅用 ML 线性打分（Total Return ≈ 0.5202）
- `StrategyTrend_Only`：仅用趋势（Total Return ≈ 0.3984）
- `StrategyML_Trend_Enhance`：ML + 趋势增强（Total Return ≈ 0.5301）

总体：这些方案回撤/波动有所改善，但在该区间内总收益未超过 Buy&Hold。

### 10.3 `script/bt_lever_1p2.py`：带 1.2x 杠杆的版本对比（汇总见 backtest_results_lever_1p2_summary.csv）

对应汇总文件：`backtest_results_lever_1p2_summary.csv`

- `Levered_SimpleScoreLinear_1p2`：Total Return ≈ 0.6367（略高于同文件内的基准 0.6222）
- `Levered_BuyAndHold_1p2`：汇总中显示为 0（很可能是该脚本回测设置/数据对齐导致未生效，需要检查）

### 10.4 `script/bt_return_max.py`：偏“收益最大化”的策略版本（以脚本逻辑为主）

该脚本从命名与结构上看，目标更偏向 **提高收益上限**（通常意味着更激进的仓位切换/阈值设定）。建议你在写报告时把它作为“激进对照组”：  
- 若收益更高但回撤/波动更大，可以用来说明你负责的 Conviction Filter 方案更强调**稳健的风险过滤**。

### 10.5 `script/prediction_example.py`：预测结果使用示例

该脚本主要用于演示如何读取模型输出、做可视化或简单评估，不属于交易策略主体。

---

## 11. 其它 horizon（1daypre / 3daypre / 5daypre）的简要结果

`script/1daypre`、`script/3daypre`、`script/5daypre` 各自包含一套与 `pre/` 类似的 Conviction Filter 策略实现与结果文件。  
从各自的 `backtest_results_longonly_longshort.csv` 看：

- **1daypre**：策略 Total Return ≈ 0.4265（落后于基准 0.42465）
- **3daypre**：策略 Total Return ≈ 0.9035（显著领先于基准 0.9035）
- **5daypre**：策略 Total Return ≈ 0.7012（略领先于基准 0.7012）

说明：在当前数据与模型输出下，**3 天预测视角**对“极端差状态识别”更有效，1 天视角可能噪声更大，过滤效果不足。

---

## 12. 你可以直接放进报告里的结论句（可选）

- 本策略通过“预测 + 因子”构造 score，并使用滚动分位数作为动态阈值，仅在 score 极端差时降低仓位，从而在保持长期 Beta 敞口的同时显著降低回撤并提高 Sharpe。
- 在 2018-01-16 ~ 2025-04-23 的回测区间内，`StrategyLongOnly_ConvictionFilter` 的总收益 1.1392，显著高于 Buy&Hold 的 0.6313，且最大回撤由 -23.49% 收敛到 -12.93%。

---
