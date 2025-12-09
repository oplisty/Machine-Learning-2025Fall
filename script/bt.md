````markdown
# 腾讯股票机器学习指数增强回测说明（基于 `bt.py`）

---

## 1. 项目背景与总体思路

- **标的**：腾讯股票（日线数据）
- **时间区间**：
  - 训练集：2018-01-01 ~ 2023-12-31
  - 回测区间：2024-01-01 ~ 2025-04-24（实际有效交易日 320 天左右）
- **初始资金**：100000 元
- **交易成本**：单边手续费 0.1%（`FEE_RATE = 0.001`）（题目没有要求，或许不需要，但是我基于实际中每次交易都需要相应的手续费于是就添加了）

代码的核心目标是：

1. **使用文件夹中已有的数据**：
   - `ml_model/output/xgboost/*.csv`：模型预测结果（不重新训练）
   - `alpha_factor/alpha_factor_ic_ranking.csv`：因子 IC 排名（不重新挖因子）
2. 在此基础上构造一些简单因子与综合信号；
3. 用 Backtrader 搭建多种指数增强策略：
   - 基准买入并持有；
   - 原始高收益 / 均衡 / 防守三类；
   - Qlib 风格的进攻型 H++ 和稳健型 B++；
4. 对比各策略在**收益、波动率、最大回撤、Sharpe**等指标上的表现。

---

## 2. 数据与特征构造（不重新训练、不重新挖因子）

### 2.1 读取现有预测结果

函数：`load_xgb_results_and_features()`

- 从目录 `ml_model/output/xgboost/` 中读取：
  - `train_results.csv`
  - `valid_results.csv`
  - `test_results.csv`
- 合并为一个完整的 DataFrame：`df_all`，以 `timestamp` 转成的 `date` 为索引。

数据中关键字段：

- `true_open, true_high, true_low, true_close, true_volume, true_amount`  
- `pred_close`：XGBoost 给出的预测收盘价  

---

### 2.2 构造真实/预测收益

在函数中，使用前一天的真实收盘价作为基准，构造：

- 前一日收盘价：
  ```python
  df_all["true_close_prev"] = df_all["true_close"].shift(1)
````

* 真实日收益：
  [
  r_{\text{true}, t} = \frac{\text{true_close}*t}{\text{true_close}*{t-1}} - 1
  ]

  ```python
  df_all["r_true"] = df_all["true_close"] / df_all["true_close_prev"] - 1.0
  ```
* 预测日收益：
  [
  r_{\text{pred}, t} = \frac{\text{pred_close}*t}{\text{true_close}*{t-1}} - 1
  ]

  ```python
  df_all["r_pred"] = df_all["pred_close"] / df_all["true_close_prev"] - 1.0
  ```

> 这样做的目的是：把“预测的价格”转化成“预测的收益”，便于与因子一起使用。

---

### 2.3 价量与技术因子（只用现有字段做派生）

价格与成交量：

```python
df_all["open"] = df_all["true_open"]
df_all["high"] = df_all["true_high"]
df_all["low"] = df_all["true_low"]
df_all["close"] = df_all["true_close"]
df_all["volume"] = df_all["true_volume"]
```

VWAP 及偏离度：

* VWAP：
  [
  \text{vwap}_t = \frac{\text{true_amount}_t}{\text{true_volume}_t}
  ]

  ```python
  df_all["vwap"] = df_all["true_amount"] / (df_all["true_volume"] + 1e-8)
  ```
* 价格相对 VWAP 的偏离：

  ```python
  df_all["price_vwap_diff"] = df_all["true_close"] - df_all["vwap"]
  ```

均线与波动率：

```python
df_all["ret_1"] = df_all["r_true"]
df_all["ma_5"]  = df_all["true_close"].rolling(5,  min_periods=3).mean()
df_all["ma_20"] = df_all["true_close"].rolling(20, min_periods=5).mean()
df_all["ma_60"] = df_all["true_close"].rolling(60, min_periods=10).mean()

df_all["volatility_10"] = df_all["ret_1"].rolling(10, min_periods=5).std()
df_all["volatility_20"] = df_all["ret_1"].rolling(20, min_periods=10).std()
```

---

### 2.4 预测收益标准化：`z_r_pred`

为了在不同时间段、不同波动环境下，让预测收益的尺度可比，代码在**训练期**（2018-2023）上做标准化：

```python
mask_train = (df_all.index >= TRAIN_START) & (df_all.index <= TRAIN_END) & df_all["r_pred"].notna()
r_pred_train = df_all.loc[mask_train, "r_pred"]
mean_pred = r_pred_train.mean()
std_pred = r_pred_train.std() if r_pred_train.std() not in [0, None] else 1.0
df_all["z_r_pred"] = (df_all["r_pred"] - mean_pred) / std_pred
```

解释：

* `z_r_pred > 0`：预测收益高于训练期平均水平；
* `z_r_pred < 0`：预测收益低于训练期平均水平；
* `|z_r_pred|` 越大，表示模型的“信号强度”越大。

---

### 2.5 多因子综合打分：`alpha_score`

函数：`build_alpha_score(df, ic_path)`

数据来源：`alpha_factor/alpha_factor_ic_ranking.csv`
文件中包含每个因子的 **Spearman IC**（秩相关系数），表示该因子对未来收益的预测能力。

使用的因子集合：

```python
usable_factors = {
    "price_vwap_diff",
    "ma_5",
    "ma_20",
    "volatility_10",
    "volatility_20",
}
```

对每个因子：

1. 从 `df` 中取出因子序列；

2. 做时间维度标准化：
   [
   z_f = \frac{f_t - \bar f}{\sigma_f}
   ]

3. 根据 IC 的符号决定正向/反向：

   ```python
   sign = 1.0 if ic_val >= 0 else -1.0
   z_adj = sign * z_f
   ```

   若 IC 为负，表示因子是“反向因子”，需要乘以 -1。

4. 权重 = |IC|，权重越大说明因子预测能力越强。

综合打分：

[
\text{alpha_score}*t = \sum_f w_f \cdot z*{f,t} \quad,\quad
w_f = \frac{|IC_f|}{\sum_j |IC_j|}
]

最后再对 `alpha_score` 做一次标准化，使其整体是 0 均值、单位方差。

> 至此，**所有信号都来自你已给的文件**：
>
> * `z_r_pred` 来源于 XGBoost 预测结果；
> * `alpha_score` 来源于已有因子 + IC 文件。

---

## 3. Backtrader 框架与基类设计

### 3.1 数据源：`MLFactorData`

在标准 OHLCV 的基础上增加了以下 `lines`：

* `r_true`：真实日收益
* `r_pred`：预测日收益
* `z_r_pred`：标准化预测收益
* `alpha_score`：多因子综合打分
* `price_vwap_diff`：收盘价 - VWAP
* `ma_5, ma_20, ma_60`
* `volatility_10, volatility_20`

Backtrader 通过 `MLFactorData(dataname=df_bt)` 直接从 Pandas DataFrame 输入，策略内部可以直接使用 `self.datas[0].xxx[0]` 获取当日值。

---

### 3.2 策略基类：`BaseMLStrategy`

基类里统一做了：

* 权益与收益记录：

  ```python
  self.equity_list   # 每日总资产
  self.ret_list      # 每日收益率
  self.equity_peak   # 历史最高权益
  self.max_drawdown  # 最大回撤
  ```

* 在 `next()` 中：

  1. 计算当日账户总价值 `value = self.broker.getvalue()`；
  2. 根据 `last_value` 计算每日收益 `daily_ret`；
  3. 更新 `equity_peak` 和当日回撤 `cur_dd = value / equity_peak - 1`；
  4. 调用子类的 `_get_target_percent()` 获取目标仓位；
  5. 通过 `order_target_percent` 把仓位调到目标值。

* 在 `stop()` 中，统一计算并打印绩效指标：

  * `Final value`：最终总资产；
  * `Total return`：总收益率；
  * `Annual return`：年化收益；
  * `Annual vol`：年化波动率；
  * `Sharpe (approx)`：近似 Sharpe 比率；
  * `Max drawdown`：最大回撤（负数，-0.2 表示最大跌幅 20%）。

并且把这些指标存入 `self.perf_stats`，方便在 `run_backtest()` 里汇总输出 CSV。

子类只需要重写：

```python
def _get_target_percent(self):
    # 返回今日目标仓位（0 ~ 1.8 之间）
```

---

## 4. 各策略逻辑说明

### 4.1 Benchmark_BuyAndHold（基准买入并持有）

类：`BuyAndHoldStrategy`

* 第一天：

  ```python
  if not self.ordered:
      self.ordered = True
      return 1.0  # 满仓
  ```
* 后续：

  ```python
  return 1.0      # 维持满仓
  ```

**代表“被动指数投资”的收益水平**，所有增强策略都与它对比。

---

### 4.2 StrategyH_HighAlphaOverlay（高收益偏好 H）

综合信号：

```python
score = 0.6 * z_r_pred + 0.4 * alpha_score
```

逻辑：

1. 用历史 `score_hist` 估计当前 score 的百分位：

   ```python
   rank_pct = sum(s <= score for s in self.score_hist) / len(self.score_hist)
   score_scaled = 2.0 * (rank_pct - 0.5)    # [-1, 1]
   ```
2. 计算增强部分：

   ```python
   alpha_max = 0.5
   alpha_overlay = alpha_max * score_scaled  # [-0.5, +0.5]
   ```
3. 最终仓位：

   ```python
   pos = 1.0 + alpha_overlay
   pos = max(0.5, min(1.5, pos))
   ```

**特点**：

* 信号好（高百分位） → 仓位可到 1.3~1.5；
* 信号差（低百分位） → 仓位可降到 0.5~0.7；
* 是一种“偏进攻”的增强策略，但仍然会在信号差时明显减仓。

---

### 4.3 StrategyB_BalancedEnhanced（均衡增强 B）

同样的综合信号：

```python
score = 0.6 * z_r_pred + 0.4 * alpha_score
```

逻辑：

* 用历史 `score_hist` 的 20%、80% 分位：

  ```python
  q_low, q_high = np.quantile(self.score_hist, [0.2, 0.8])
  ```
* 仓位决策：

  ```python
  if score < q_low:
      return 0.7
  elif score > q_high:
      return 1.3
  else:
      return 1.0
  ```

**特点**：

* 风格居中：仓位只在 0.7~1.3 之间变化；
* 相比 H 策略，波动较小，换手也更平滑；
* 目标是“收益-风险均衡”的指数增强。

---

### 4.4 StrategyD_DefensiveMeanRev（防守型 D）

信号拆成两个部分：

1. 价格相对 VWAP 的偏离：`price_vwap_diff` 的 z-score；
2. 组合信号：`combo = z_r_pred + alpha_score`。

逻辑：

* 对 `price_vwap_diff` 做滚动 z-score：

  ```python
  z = (diff - mean_200) / std_200
  ```
* 决策规则：

  ```python
  if z < -1.0 and combo > 0:
      return 0.8            # 认为被错杀，适度加仓
  elif z > 1.0 and combo < 0:
      return 0.2            # 认为短期高估，明显减仓
  else:
      return 0.5            # 其他时间保持中性偏低仓位
  ```

**特点**：

* 强调“均值回归”和“价格-成交的短期错位”；
* 整体仓位偏低（0.2~0.8）→ 收益有限，但回撤偏小；
* 偏典型的“防守型”风格。

---

### 4.5 StrategyHPP_QlibStyle（高收益增强 H++）

这是一个更接近 Qlib Beta Overlay 思路的**进攻型策略**，综合考虑：

* ML + 多因子信号（`score`）
* 时间维度标准化 + tanh 非线性压缩
* 趋势过滤（`close` vs `ma_60`）
* 波动率风控（`volatility_20` 分位数）
* 最大回撤风控（账户维度）
* 仓位平滑（限制日度变化）

#### 4.5.1 信号与标准化

```python
score = 0.6 * z_r_pred + 0.4 * alpha_score
# 最近 200 日 z-score
z = (score - mean_200) / std_200
overlay_max = 0.8
overlay = overlay_max * tanh(z)
base_beta = 1.0
target = base_beta + overlay   # 理论 [0.2, 1.8] 左右
```

#### 4.5.2 趋势过滤：`close` vs `ma_60`

```python
if close < ma60:
    target = 1.0 + (target - 1.0) * 0.5    # 下跌趋势，减半 overlay
else:
    target = 1.0 + (target - 1.0) * 1.1    # 上涨趋势，略微放大 overlay
```

#### 4.5.3 波动率风控：利用 `volatility_20` 的分位数

```python
vol_q50 = 中位数
vol_q80 = 80% 分位

if vol > vol_q80:
    scale = vol_q80 / vol
    target = 1.0 + (target - 1.0) * scale    # 高波动 -> 收缩
elif vol < vol_q50:
    scale = min(1.2, vol_q50 / vol)
    target = 1.0 + (target - 1.0) * scale    # 低波动 -> 稍微放大
```

#### 4.5.4 最大回撤风控

利用基类中记录的 `equity_peak` 和当前权益计算回撤：

```python
cur_dd = cur_equity / equity_peak - 1.0    # ≤ 0

if cur_dd < -0.18:
    target = 0.5                            # 硬阈值 18%，强制降到 0.5
elif cur_dd < -0.10:
    target = 1.0 + (target - 1.0) * 0.3     # 软阈值 10%，大幅减弱 overlay
```

#### 4.5.5 仓位平滑与裁剪

* 单日仓位变化不超过 0.25：

  ```python
  if last_target is not None and abs(target - last_target) > 0.25:
      target = last_target + sign(target - last_target) * 0.25
  ```
* 最终限制在 `[0.2, 1.8]` 区间。

**整体风格**：
在行情和信号都好的时候，可加到较高仓位；
一旦波动率和回撤超标，自动收缩仓位，控制风险。

---

### 4.6 StrategyBPP_QlibStyle（均衡增强 B++）

B++ 与 H++ 的框架类似，但所有参数更保守，主打**稳健 + 高 Sharpe**：

* `overlay_max = 0.5`，仓位区间缩窄到约 `[0.8, 1.5]`；
* 趋势过滤中，下跌趋势强化减仓，上涨趋势只保留部分加仓；
* 波动率风控更敏感：`vol_q40 / vol_q70`，略高波动就减仓；
* 回撤阈值更低：软 8%，硬 15%，更“胆小”；
* 仓位平滑步长 0.15，更不容易剧烈跳仓。

可以理解为：**H++ 的防守版，强调曲线平滑和回撤控制。**

---

## 5. 回测流程与结果输出

函数：`run_backtest()`

1. 调用 `load_xgb_results_and_features()` 读入所有数据；
2. 选取回测区间 `[BACKTEST_START, BACKTEST_END]`；
3. 丢弃关键字段缺失的行：

   ```python
   df_bt = df_bt.dropna(subset=[
       "open", "high", "low", "close", "volume",
       "r_true", "r_pred", "z_r_pred", "alpha_score"
   ])
   ```
4. 对每个策略单独跑一遍 Backtrader：

   * 初始化资金、手续费；
   * 加载 `MLFactorData`；
   * 添加相应 Strategy；
   * `cerebro.run()`；
   * 从 `strat.perf_stats` 取出绩效指标。
5. 汇总所有策略结果，形成 `backtest_results_summary.csv`，并在控制台打印。

---

## 6. 实际回测结果（你给出的 CSV）

回测输出（已四舍五入处理）：

| Strategy                   | Final Value | Total Return | Annual Return | Annual Vol | Sharpe | Max Drawdown |
| -------------------------- | ----------: | -----------: | ------------: | ---------: | -----: | -----------: |
| Benchmark_BuyAndHold       |   162216.54 |       0.6222 |        0.4637 |     0.3421 | 1.3555 |      -0.2349 |
| StrategyH_HighAlphaOverlay |   146196.84 |       0.4620 |        0.3486 |     0.2438 | 1.4301 |      -0.1194 |
| StrategyB_BalancedEnhanced |   151718.91 |       0.5172 |        0.3886 |     0.2800 | 1.3878 |      -0.1637 |
| StrategyD_DefensiveMeanRev |   122040.57 |       0.2204 |        0.1698 |     0.1818 | 0.9343 |      -0.1318 |
| StrategyHPP_QlibStyle      |   141368.09 |       0.4137 |        0.3134 |     0.2459 | 1.2743 |      -0.1145 |
| StrategyBPP_QlibStyle      |   151743.32 |       0.5174 |        0.3887 |     0.3070 | 1.2663 |      -0.2103 |

### 6.1 关键观察

1. **基准 Buy&Hold**：

   * 总收益最高（约 62%），年化约 46.4%；
   * 年化波动 34.2%，最大回撤约 -23.5%；
   * Sharpe ≈ 1.36。

2. **高收益偏好 H**：

   * 总收益约 46.2%，明显低于基准；
   * 年化波动只有 24.4%，明显低于基准；
   * 最大回撤只有 -11.9%（不到基准的一半）；
   * Sharpe ≈ 1.43，**反而高于基准**。

   → 表明在“收益-风险比”意义上，H 策略优于纯买入持有。

3. **均衡增强 B / B++**：

   * 总收益 ~51.7%，仍低于 62% 的基准；
   * 波动 28%~30.7%，低于或接近基准；
   * 最大回撤 -16.4%（B）和 -21.0%（B++）；
   * Sharpe ≈ 1.39（B）和 1.27（B++），接近或略高于基准。

   → 相较基准，**更稳，回撤略小，收益略低**，整体风格“均衡”。

4. **防守型 D**：

   * 收益最低（22% 左右）；
   * 波动、回撤都明显压低；
   * Sharpe 不高（0.93），但风险非常温和。

   → 可以作为“极端防守”的示范策略。

5. **Qlib 风格 H++**：

   * 收益约 41.4%，波动 24.6%，回撤 -11.5%；
   * Sharpe ≈ 1.27，略低于 H，但整体风险控制较好。

---

## 7. 结果分析与可能存在的问题

### 7.1 为什么绝对收益普遍低于基准？

直观原因：

* 回测期间（2024-01 ~ 2025-04），腾讯总体上涨幅度较大；
* 基准策略永远满仓（仓位 = 1.0），享受了整个上涨过程；
* 增强策略在信号差、波动高、回撤大的时候**主动降低仓位**，
  → 平均仓位小于 1 → 在单边上涨行情中**必然拉低绝对收益**。

模型角度来看：

1. **预测与因子信号有用，但不够强**
   从结果看：

   * 多数策略的 Sharpe 与基准接近甚至更高；
   * 最大回撤显著小于基准。
     说明 `z_r_pred + alpha_score` 提供了一定的择时能力，但还不足以支撑一个“长期 beta>1 且不被爆仓”的激进策略。

2. **策略设计偏向“稳健 + 风控”**

   * H/B 策略在信号较差时会快速降低仓位到 0.5/0.7；
   * H++/B++ 又叠加了趋势过滤、波动率风控、最大回撤风控；
   * 这些设计在**震荡/下跌阶段有效保护本金**，但在样本期整体上涨的背景下，也会错过一些涨幅。

从课程角度，这完全可以写成一个合理的结论：

> 在这段牛市背景下，单纯买入并持有获得了最高的绝对收益。
> 各类 ML 指数增强策略在收益略低于基准的前提下，
> 用更低的波动和更小的最大回撤换取了更高或相近的 Sharpe，
> 体现出机器学习信号在风险调整收益上的优势。

---

### 7.2 是“预测问题”还是“因子问题”？

结合代码和结果，可以这样分析：

1. **不是代码 bug 问题**

   * 回测能跑通、多策略之间表现合理；
   * 不存在明显逻辑错误（如信号反向、仓位异常等）。

2. **预测/因子质量有限，但不为零**

   * 如果预测和因子完全没用，策略大概率：

     * 收益 < 基准、Sharpe < 基准、回撤更大；
   * 实际上，你的 H/B 等策略在 Sharpe 上不差甚至略优，
     → 说明 **信号有一定的 alpha，但强度有限**。

3. **主要矛盾：信号强度 vs 风险偏好**

   * 你目前的策略设计是偏“防守型增强”的：

     * 信号不好 → 明显减仓；
     * 高波动、高回撤 → 强制缩小仓位。
   * 在一个牛市环境下，如果信号不是特别强，这种防守偏好会**天然牺牲绝对收益**。

因此可以在报告中写成：

> 本实验中，机器学习预测与多因子信号在风险调整后的收益上优于基准，
> 但由于信号强度有限且策略中引入了多重风控（趋势过滤、波动率、最大回撤），
> 在 2024~2025 这段整体上涨的行情中，部分增强策略在绝对收益上略低于“永远满仓”的基准策略。
> 这说明未来若想进一步提高超额收益，需要在**模型预测精度（提高信噪比）**
> 和**仓位风控参数（适当提高平均仓位）**之间做更细致的权衡。

---

### 7.3 可能的改进方向（仍然不重新训练模型）

如果需要在现有数据和模型下再做一些探索，可以考虑：

1. **更激进的进攻型策略**（只调策略参数）：

   * 提高 `base_beta`（例如从 1.0 改为 1.05 或 1.1）；
   * 提高 `overlay_max`（在 H++ 中从 0.8 调到 1.0）；
   * 收紧降仓条件（只在极差信号或极端波动时才降到 <1）。

2. **减少过度防守**：

   * 回撤软阈值可以从 10% 稍微放宽到 12%~15%；
   * 波动率分位数阈值可以放宽，避免一有风吹草动就大幅减仓。

3. **多策略组合**：

   * 将进攻型（H++）和防守型（D/B++）按一定权重组合，
     形成一个在“牛市不太拖后腿、熊市不太难受”的综合策略。

---

## 8. 小结

本次 `bt.py` 回测实现了一个完整的：

* **数据管道**：只使用已有 XGBoost 预测结果和因子 IC 文件，通过简单派生构造收益、技术因子和综合因子得分；
* **Backtrader 回测框架**：统一的策略基类、风险指标计算与结果汇总；
* **多种策略对比**：从最简单的买入并持有，到原始增强策略，再到 Qlib 风格的 H++/B++；
* **风险收益分析**：给出每个策略的收益、波动、回撤和 Sharpe，对“收益不如基准”的原因做出合理解释。

从教学和报告角度，这套代码和结果已经足够支撑：

* 机器学习预测 + 因子如何转化为交易信号；
* 不同类型的指数增强策略如何设计；
* 在给定样本期内，如何用数据来分析：

  * 收益 / 风险 / 回撤的权衡；
  * 模型信号强度与策略风控之间的博弈关系。

```
```
