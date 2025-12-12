import os
import pickle
import yaml
from typing import Tuple, Dict
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import get_original_cwd
plt.rcParams['font.sans-serif'] = ['STHeiti']
plt.rcParams['axes.unicode_minus'] = False

def load_config(config_path: str) -> Dict:
    """加载 YAML 配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def load_data(csv_path: str) -> pd.DataFrame:
    """加载并排序原始 K 线 CSV 数据"""
    df = pd.read_csv(csv_path)
    df["timestamps"] = pd.to_datetime(df["timestamps"])
    df = df.sort_values("timestamps").reset_index(drop=True)
    return df


def filter_by_time_range(
    df: pd.DataFrame,
    start_time: str,
    end_time: str
) -> pd.DataFrame:
    """根据时间范围过滤数据"""
    start = pd.to_datetime(start_time)
    end = pd.to_datetime(end_time)
    mask = (df["timestamps"] >= start) & (df["timestamps"] <= end)
    return df[mask].reset_index(drop=True)


def build_supervised(
    df: pd.DataFrame,
    feature_cols: list,
    target_cols: list,
    lookback: int = 5,
    horizon: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    构建监督学习数据集：
    X: 过去 `lookback` 天的特征
    y: 未来 horizon 天后的目标值（默认为 1 天）
    """
    values = df[feature_cols].values.astype(float)
    targets = df[target_cols].values.astype(float)
    timestamps = df["timestamps"].values

    X_list, y_list, ts_list = [], [], []
    # i 对应当前窗口的结尾索引，标签取 i + horizon - 1
    for i in range(lookback, len(df) - horizon + 1):
        X_window = values[i - lookback : i].reshape(-1)
        y_next = targets[i + horizon - 1]  # 预测 horizon 天后的目标
        X_list.append(X_window)
        y_list.append(y_next)
        ts_list.append(timestamps[i + horizon - 1])  # 标签对应的时间戳

    X = np.array(X_list)
    y = np.array(y_list)
    ts_arr = np.array(ts_list)
    return X, y, ts_arr


def split_by_time(
    X: np.ndarray,
    y: np.ndarray,
    timestamps: np.ndarray,
    train_range: list,
    valid_range: list,
    test_range: list,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """根据时间范围划分训练集、验证集和测试集"""
    train_start = pd.to_datetime(train_range[0])
    train_end = pd.to_datetime(train_range[1])
    valid_start = pd.to_datetime(valid_range[0])
    valid_end = pd.to_datetime(valid_range[1])
    test_start = pd.to_datetime(test_range[0])
    test_end = pd.to_datetime(test_range[1])

    train_mask = (timestamps >= train_start) & (timestamps <= train_end)
    valid_mask = (timestamps >= valid_start) & (timestamps <= valid_end)
    test_mask = (timestamps >= test_start) & (timestamps <= test_end)

    X_train = X[train_mask]
    y_train = y[train_mask]
    ts_train = timestamps[train_mask]

    X_valid = X[valid_mask]
    y_valid = y[valid_mask]
    ts_valid = timestamps[valid_mask]

    X_test = X[test_mask]
    y_test = y[test_mask]
    ts_test = timestamps[test_mask]

    return X_train, y_train, ts_train, X_valid, y_valid, ts_valid, X_test, y_test, ts_test


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """计算全局回归指标"""
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def compute_column_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_cols: list,
) -> Dict[str, Dict[str, float]]:
    """计算每个目标列的指标"""
    per_col: Dict[str, Dict[str, float]] = {}
    for i, col in enumerate(target_cols):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        per_col[col] = {
            "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
            "mae": float(mean_absolute_error(yt, yp)),
            "r2": float(r2_score(yt, yp)),
        }
    return per_col


def train_xgboost_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    model_kwargs: Dict,
    target_cols: list,
) -> Tuple[
    list,
    Dict[str, float],
    Dict[str, float],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """训练 XGBoost 模型（单目标逐列训练，兼容多输出）"""
    print("\n=== 训练 XGBoost 模型 ===")
    
    # 准备参数（处理 Hydra 的 DictConfig）
    if not isinstance(model_kwargs, dict):
        params = OmegaConf.to_container(model_kwargs, resolve=True)
    else:
        params = dict(model_kwargs)
    eval_metric = params.pop('eval_metric', 'rmse')
    nthread = params.pop('nthread', -1)
    
    n_targets = y_train.shape[1]
    models: list = []
    y_train_pred = np.zeros_like(y_train, dtype=float)
    y_valid_pred = np.zeros_like(y_valid, dtype=float)
    train_hist_list, valid_hist_list = [], []
    
    print("开始训练（逐列）...")
    for i in range(n_targets):
        est = xgb.XGBRegressor(**params, n_jobs=nthread, eval_metric=eval_metric)
        est.fit(
            X_train,
            y_train[:, i],
            eval_set=[(X_train, y_train[:, i]), (X_valid, y_valid[:, i])],
            verbose=False,
        )
        models.append(est)
        y_train_pred[:, i] = est.predict(X_train)
        y_valid_pred[:, i] = est.predict(X_valid)
        res = est.evals_result()
        train_hist_list.append(res["validation_0"][eval_metric])
        valid_hist_list.append(res["validation_1"][eval_metric])
    
    train_history = np.mean(np.vstack(train_hist_list), axis=0)
    valid_history = np.mean(np.vstack(valid_hist_list), axis=0)
    
    # 计算指标
    train_metrics = compute_metrics(y_train, y_train_pred)
    valid_metrics = compute_metrics(y_valid, y_valid_pred)
    
    print(f"\n训练集指标: RMSE={train_metrics['rmse']:.4f}, "
          f"MAE={train_metrics['mae']:.4f}, R2={train_metrics['r2']:.4f}")
    print(f"验证集指标: RMSE={valid_metrics['rmse']:.4f}, "
          f"MAE={valid_metrics['mae']:.4f}, R2={valid_metrics['r2']:.4f}")
    
    return (
        models,
        train_metrics,
        valid_metrics,
        y_train_pred,
        y_valid_pred,
        train_history,
        valid_history,
    )


def save_results(
    timestamps: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_cols: list,
    output_dir: str,
    prefix: str = "",
) -> None:
    """保存预测结果 CSV 和可视化图表"""
    os.makedirs(output_dir, exist_ok=True)

    result_df = pd.DataFrame(
        y_true, columns=[f"true_{c}" for c in target_cols]
    )
    for i, c in enumerate(target_cols):
        result_df[f"pred_{c}"] = y_pred[:, i]
    result_df["timestamp"] = timestamps[-len(result_df):]

    csv_path = os.path.join(output_dir, f"{prefix}results.csv")
    result_df.to_csv(csv_path, index=False)
    print(f"保存结果到: {csv_path}")

    # 绘制 close 和 volume 的图表
    for col in ["close", "volume"]:
        if col in target_cols:
            plt.figure(figsize=(12, 5))
            plt.plot(
                result_df["timestamp"],
                result_df[f"true_{col}"],
                label="真实值",
                color="blue",
                linewidth=1.2,
            )
            plt.plot(
                result_df["timestamp"],
                result_df[f"pred_{col}"],
                label="预测值",
                color="red",
                linewidth=1.2,
                alpha=0.7,
            )
            plt.title(f"XGBoost {col} 预测结果 ({prefix})")
            plt.xlabel("时间")
            plt.ylabel(col)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            fig_path = os.path.join(output_dir, f"{prefix}forecast_{col}.png")
            plt.savefig(fig_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"保存图表到: {fig_path}")


def plot_loss_curve(
    train_history: np.ndarray,
    valid_history: np.ndarray,
    output_dir: str,
    metric_name: str = "rmse",
) -> None:
    """绘制训练/验证损失曲线"""
    if train_history.size == 0 or valid_history.size == 0:
        return
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(train_history, label="train")
    plt.plot(valid_history, label="valid")
    plt.xlabel("Boosting round")
    plt.ylabel(metric_name)
    plt.title(f"Training vs Validation {metric_name}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    fig_path = os.path.join(output_dir, f"loss_curve_{metric_name}.png")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"保存训练/验证曲线到: {fig_path}")


def save_metrics_log(
    train_metrics: Dict[str, float],
    valid_metrics: Dict[str, float],
    test_metrics: Dict[str, float],
    test_metrics_per_col: Dict[str, Dict[str, float]],
    target_cols: list,
    output_dir: str,
) -> None:
    """保存指标日志"""
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "metrics_log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("XGBoost 模型评估指标 (RMSE/MAE 越小越好, R2 越大越好):\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("训练集指标:\n")
        f.write(f"  RMSE={train_metrics['rmse']:.6f}, "
                f"MAE={train_metrics['mae']:.6f}, "
                f"R2={train_metrics['r2']:.6f}\n\n")
        
        f.write("验证集指标:\n")
        f.write(f"  RMSE={valid_metrics['rmse']:.6f}, "
                f"MAE={valid_metrics['mae']:.6f}, "
                f"R2={valid_metrics['r2']:.6f}\n\n")
        
        f.write("测试集指标:\n")
        f.write(f"  RMSE={test_metrics['rmse']:.6f}, "
                f"MAE={test_metrics['mae']:.6f}, "
                f"R2={test_metrics['r2']:.6f}\n\n")
        
        f.write("测试集各列指标:\n")
        for col in target_cols:
            m = test_metrics_per_col[col]
            f.write(f"  {col}: RMSE={m['rmse']:.6f}, "
                    f"MAE={m['mae']:.6f}, R2={m['r2']:.6f}\n")
    print(f"保存指标日志到: {log_path}")


def save_model(model: list, output_dir: str) -> None:
    """保存模型列表"""
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "xgboost_model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"保存模型到: {model_path}")


@hydra.main(version_base=None, config_path="../config", config_name="xgboost")
def main(cfg: DictConfig):
    # Hydra 运行时工作目录会变化，使用 original_cwd 还原到项目根
    base_dir = Path(get_original_cwd())

    # 配置与可覆写参数
    data_config = cfg.data
    model_config = cfg.model
    output_dir_cfg = cfg.output_dir
    checkpoints_output_dir_cfg = cfg.checkpoints_output_dir
    horizon = int(cfg.get("horizon", 1))
    lookback = int(cfg.get("lookback", 10))

    # 构建路径；不同预测天数分目录保存
    data_path = base_dir / data_config.data_path
    output_dir = base_dir / output_dir_cfg / f"horizon_{horizon}"
    
    print(f"\n数据路径: {data_path}")
    print(f"输出目录: {output_dir}")
    
    # 加载数据
    print("\n加载数据...")
    df = load_data(str(data_path))
    print(f"数据时间范围: {df['timestamps'].min()} -> {df['timestamps'].max()}, "
          f"总行数: {len(df)}")
    
    # 获取测试集结束时间，确保 fit_end_time 包含测试集数据
    test_range = data_config["test"]
    test_end = pd.to_datetime(test_range[1])
    
    # 根据 fit 时间范围过滤数据（用于特征拟合）
    # 如果 fit_end_time 早于测试集结束时间，则扩展到测试集结束时间
    fit_start = data_config["fit_start_time"]
    fit_end = data_config["fit_end_time"]
    fit_end_dt = pd.to_datetime(fit_end)
    
    # 确保 fit_end_time 至少包含到测试集结束时间
    if fit_end_dt < test_end:
        print(f"警告: fit_end_time ({fit_end}) 早于测试集结束时间 ({test_range[1]})")
        print(f"自动扩展 fit_end_time 到测试集结束时间以包含测试集数据")
        fit_end = test_range[1]
    
    df_fit = filter_by_time_range(df, fit_start, fit_end)
    print(f"特征拟合数据范围: {fit_start} -> {fit_end}, 行数: {len(df_fit)}")
    
    # 定义特征和目标列
    feature_cols = ["open", "high", "low", "close", "volume", "amount"]
    target_cols = ["open", "high", "low", "close", "volume", "amount"]
    
    # 构建监督学习数据集
    print(f"\n构建监督学习数据集 (lookback={lookback}, horizon={horizon})...")
    X, y, ts = build_supervised(
        df_fit,
        feature_cols,
        target_cols,
        lookback=lookback,
        horizon=horizon,
    )
    print(f"数据集形状: X={X.shape}, y={y.shape}")
    
    # 根据时间范围划分数据集（test_range 已在上面获取）
    train_range = data_config["train"]
    valid_range = data_config["valid"]
    
    print(f"\n划分数据集:")
    print(f"  训练集: {train_range[0]} -> {train_range[1]}")
    print(f"  验证集: {valid_range[0]} -> {valid_range[1]}")
    print(f"  测试集: {test_range[0]} -> {test_range[1]}")
    
    X_train, y_train, ts_train, X_valid, y_valid, ts_valid, X_test, y_test, ts_test = split_by_time(
        X, y, ts, train_range, valid_range, test_range
    )
    
    print(f"\n数据集大小:")
    print(f"  训练集: {X_train.shape[0]} 样本")
    print(f"  验证集: {X_valid.shape[0]} 样本")
    print(f"  测试集: {X_test.shape[0]} 样本")
    
    # 检查测试集是否为空
    if X_test.shape[0] == 0:
        print("\n警告: 测试集为空！请检查 fit_end_time 是否包含测试集的时间范围。")
        print(f"当前 fit_end_time: {fit_end}")
        print(f"测试集时间范围: {test_range[0]} -> {test_range[1]}")
        print("将跳过测试集评估。")
    
    # 训练模型
    model_kwargs = model_config["kwargs"].copy()
    (
        models,
        train_metrics,
        valid_metrics,
        y_train_pred,
        y_valid_pred,
        train_history,
        valid_history,
    ) = train_xgboost_model(
        X_train, y_train, X_valid, y_valid, model_kwargs, target_cols
    )
    
    # 测试集预测（如果测试集不为空）
    if X_test.shape[0] > 0:
        print("\n=== 测试集评估 ===")
        y_test_pred = np.zeros_like(y_test, dtype=float)
        for i, est in enumerate(models):
            y_test_pred[:, i] = est.predict(X_test)
        test_metrics = compute_metrics(y_test, y_test_pred)
        test_metrics_per_col = compute_column_metrics(y_test, y_test_pred, target_cols)
        
        print(f"测试集指标: RMSE={test_metrics['rmse']:.4f}, "
              f"MAE={test_metrics['mae']:.4f}, R2={test_metrics['r2']:.4f}")
        
        print("\n测试集各列指标:")
        for col in target_cols:
            m = test_metrics_per_col[col]
            print(f"  {col}: RMSE={m['rmse']:.4f}, "
                  f"MAE={m['mae']:.4f}, R2={m['r2']:.4f}")
    else:
        # 如果测试集为空，创建空的指标字典
        test_metrics = {"rmse": 0.0, "mae": 0.0, "r2": 0.0}
        test_metrics_per_col = {col: {"rmse": 0.0, "mae": 0.0, "r2": 0.0} for col in target_cols}
        y_test_pred = np.array([]).reshape(0, len(target_cols))
    
    # 保存结果
    print("\n=== 保存结果 ===")
    save_model(models, str(checkpoints_output_dir_cfg))
    save_results(ts_train, y_train, y_train_pred, target_cols, str(output_dir), prefix="train_")
    save_results(ts_valid, y_valid, y_valid_pred, target_cols, str(output_dir), prefix="valid_")
    if X_test.shape[0] > 0:
        save_results(ts_test, y_test, y_test_pred, target_cols, str(output_dir), prefix="test_")
    save_metrics_log(train_metrics, valid_metrics, test_metrics, test_metrics_per_col, target_cols, str(output_dir))
    plot_loss_curve(train_history, valid_history, str(output_dir), metric_name="rmse")
    
    print("\n训练完成！")


if __name__ == "__main__":
    main()

