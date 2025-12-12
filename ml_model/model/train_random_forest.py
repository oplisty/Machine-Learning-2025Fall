import os
import pickle
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import get_original_cwd
plt.rcParams['font.sans-serif'] = ['STHeiti']
plt.rcParams['axes.unicode_minus'] = False

def load_data(csv_path: str) -> pd.DataFrame:
    """加载并排序原始 K 线 CSV 数据"""
    df = pd.read_csv(csv_path)
    df["timestamps"] = pd.to_datetime(df["timestamps"])
    df = df.sort_values("timestamps").reset_index(drop=True)
    return df


def filter_by_time_range(df: pd.DataFrame, start_time: str, end_time: str) -> pd.DataFrame:
    """根据时间范围过滤数据"""
    start = pd.to_datetime(start_time)
    end = pd.to_datetime(end_time)
    mask = (df["timestamps"] >= start) & (df["timestamps"] <= end)
    return df[mask].reset_index(drop=True)


def build_supervised(
    df: pd.DataFrame,
    feature_cols: list,
    target_cols: list,
    lookback: int = 10,
    horizon: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    构建监督学习数据集：
    X: 过去 lookback 天的特征
    y: 未来 horizon 天后的目标
    """
    values = df[feature_cols].values.astype(float)
    targets = df[target_cols].values.astype(float)
    timestamps = df["timestamps"].values

    X_list, y_list, ts_list = [], [], []
    for i in range(lookback, len(df) - horizon + 1):
        X_window = values[i - lookback : i].reshape(-1)
        y_next = targets[i + horizon - 1]
        X_list.append(X_window)
        y_list.append(y_next)
        ts_list.append(timestamps[i + horizon - 1])

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
) -> Tuple[np.ndarray, ...]:
    """根据时间范围划分训练/验证/测试集"""
    train_start = pd.to_datetime(train_range[0])
    train_end = pd.to_datetime(train_range[1])
    valid_start = pd.to_datetime(valid_range[0])
    valid_end = pd.to_datetime(valid_range[1])
    test_start = pd.to_datetime(test_range[0])
    test_end = pd.to_datetime(test_range[1])

    train_mask = (timestamps >= train_start) & (timestamps <= train_end)
    valid_mask = (timestamps >= valid_start) & (timestamps <= valid_end)
    test_mask = (timestamps >= test_start) & (timestamps <= test_end)

    X_train, y_train, ts_train = X[train_mask], y[train_mask], timestamps[train_mask]
    X_valid, y_valid, ts_valid = X[valid_mask], y[valid_mask], timestamps[valid_mask]
    X_test, y_test, ts_test = X[test_mask], y[test_mask], timestamps[test_mask]

    return X_train, y_train, ts_train, X_valid, y_valid, ts_valid, X_test, y_test, ts_test


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def compute_column_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, target_cols: list
) -> Dict[str, Dict[str, float]]:
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


def train_random_forest_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    model_kwargs: Dict,
) -> Tuple[
    RandomForestRegressor,
    Dict[str, float],
    Dict[str, float],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    训练 RandomForest，并记录随树数量增加的训练/验证损失（rmse）
    """
    print("\n=== 训练 RandomForest 模型 ===")

    params = model_kwargs.copy()
    total_estimators = int(params.pop("n_estimators", 300))
    step = max(10, total_estimators // 20)  # 至少 10，每 5% 记录一次

    rf = RandomForestRegressor(
        n_estimators=0,
        warm_start=True,
        **params,
    )

    train_history, valid_history = [], []

    print("开始训练（逐步增加树数量）...")
    n = 0
    while n < total_estimators:
        n = min(total_estimators, n + step)
        rf.set_params(n_estimators=n)
        rf.fit(X_train, y_train)

        y_train_pred_stage = rf.predict(X_train)
        y_valid_pred_stage = rf.predict(X_valid)

        # 只记录 rmse 曲线
        train_rmse = float(np.sqrt(mean_squared_error(y_train, y_train_pred_stage)))
        valid_rmse = float(np.sqrt(mean_squared_error(y_valid, y_valid_pred_stage)))
        train_history.append(train_rmse)
        valid_history.append(valid_rmse)

        print(f"  树数={n}: train_rmse={train_rmse:.4f}, valid_rmse={valid_rmse:.4f}")

    # 最终预测
    y_train_pred = rf.predict(X_train)
    y_valid_pred = rf.predict(X_valid)

    train_metrics = compute_metrics(y_train, y_train_pred)
    valid_metrics = compute_metrics(y_valid, y_valid_pred)

    print(
        f"\n训练集指标: RMSE={train_metrics['rmse']:.4f}, "
        f"MAE={train_metrics['mae']:.4f}, R2={train_metrics['r2']:.4f}"
    )
    print(
        f"验证集指标: RMSE={valid_metrics['rmse']:.4f}, "
        f"MAE={valid_metrics['mae']:.4f}, R2={valid_metrics['r2']:.4f}"
    )

    return (
        rf,
        train_metrics,
        valid_metrics,
        y_train_pred,
        y_valid_pred,
        np.array(train_history),
        np.array(valid_history),
    )


def save_results(
    timestamps: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_cols: list,
    output_dir: str,
    prefix: str = "",
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    result_df = pd.DataFrame(y_true, columns=[f"true_{c}" for c in target_cols])
    for i, c in enumerate(target_cols):
        result_df[f"pred_{c}"] = y_pred[:, i]
    result_df["timestamp"] = timestamps[-len(result_df) :]

    csv_path = os.path.join(output_dir, f"{prefix}results.csv")
    result_df.to_csv(csv_path, index=False)
    print(f"保存结果到: {csv_path}")

    for col in ["close", "volume"]:
        if col in target_cols:
            plt.figure(figsize=(10, 4))
            plt.plot(result_df["timestamp"], result_df[f"true_{col}"], label="真实值")
            plt.plot(result_df["timestamp"], result_df[f"pred_{col}"], label="预测值")
            plt.title(f"RandomForest {col} 预测 ({prefix})")
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
    plt.xlabel("Step (num_trees)")
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
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "metrics_log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("RandomForest 评估指标 (RMSE/MAE 越小越好, R2 越大越好):\n")
        f.write("=" * 60 + "\n\n")
        f.write(
            f"训练集: RMSE={train_metrics['rmse']:.6f}, "
            f"MAE={train_metrics['mae']:.6f}, R2={train_metrics['r2']:.6f}\n"
        )
        f.write(
            f"验证集: RMSE={valid_metrics['rmse']:.6f}, "
            f"MAE={valid_metrics['mae']:.6f}, R2={valid_metrics['r2']:.6f}\n"
        )
        f.write(
            f"测试集: RMSE={test_metrics['rmse']:.6f}, "
            f"MAE={test_metrics['mae']:.6f}, R2={test_metrics['r2']:.6f}\n\n"
        )
        f.write("测试集各列指标:\n")
        for col in target_cols:
            m = test_metrics_per_col[col]
            f.write(
                f"  {col}: RMSE={m['rmse']:.6f}, "
                f"MAE={m['mae']:.6f}, R2={m['r2']:.6f}\n"
            )
    print(f"保存指标日志到: {log_path}")


def save_model(model: RandomForestRegressor, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "random_forest_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"保存模型到: {model_path}")


@hydra.main(version_base=None, config_path="../config", config_name="random_forest")
def main(cfg: DictConfig):
    base_dir = Path(get_original_cwd())

    data_cfg = cfg.data
    model_cfg = cfg.model
    model_kwargs = OmegaConf.to_container(model_cfg.kwargs, resolve=True)
    output_dir_cfg = cfg.output_dir
    checkpoints_output_dir_cfg = cfg.get("checkpoints_output_dir", "ml_model/checkpoints/random_forest")
    horizon = int(cfg.get("horizon", 1))
    lookback = int(cfg.get("lookback", 10))

    data_path = base_dir / data_cfg.data_path
    output_dir = base_dir / output_dir_cfg / f"horizon_{horizon}"
    checkpoints_output_dir = base_dir / checkpoints_output_dir_cfg / f"horizon_{horizon}"

    print(f"\n数据路径: {data_path}")
    print(f"输出目录: {output_dir}")

    # 加载数据
    df = load_data(str(data_path))
    print(
        f"数据时间范围: {df['timestamps'].min()} -> {df['timestamps'].max()}, "
        f"总行数: {len(df)}"
    )

    # 处理 fit 时间范围，确保覆盖测试集
    test_range = data_cfg.test
    test_end = pd.to_datetime(test_range[1])
    fit_start = data_cfg.fit_start_time
    fit_end = data_cfg.fit_end_time
    fit_end_dt = pd.to_datetime(fit_end)
    if fit_end_dt < test_end:
        print(f"警告: fit_end_time ({fit_end}) 早于测试集结束时间 ({test_range[1]})，自动扩展到测试结束。")
        fit_end = test_range[1]

    df_fit = filter_by_time_range(df, fit_start, fit_end)
    print(f"特征拟合数据范围: {fit_start} -> {fit_end}, 行数: {len(df_fit)}")

    feature_cols = ["open", "high", "low", "close", "volume", "amount"]
    target_cols = ["open", "high", "low", "close", "volume", "amount"]

    print(f"\n构建监督学习数据集 (lookback={lookback}, horizon={horizon})...")
    X, y, ts = build_supervised(df_fit, feature_cols, target_cols, lookback=lookback, horizon=horizon)
    print(f"数据集形状: X={X.shape}, y={y.shape}")

    train_range = data_cfg.train
    valid_range = data_cfg.valid

    print("\n划分数据集:")
    print(f"  训练集: {train_range[0]} -> {train_range[1]}")
    print(f"  验证集: {valid_range[0]} -> {valid_range[1]}")
    print(f"  测试集: {test_range[0]} -> {test_range[1]}")

    (
        X_train,
        y_train,
        ts_train,
        X_valid,
        y_valid,
        ts_valid,
        X_test,
        y_test,
        ts_test,
    ) = split_by_time(X, y, ts, train_range, valid_range, test_range)

    print("\n数据集大小:")
    print(f"  训练集: {X_train.shape[0]} 样本")
    print(f"  验证集: {X_valid.shape[0]} 样本")
    print(f"  测试集: {X_test.shape[0]} 样本")

    (
        rf_model,
        train_metrics,
        valid_metrics,
        y_train_pred,
        y_valid_pred,
        train_history,
        valid_history,
    ) = train_random_forest_model(
        X_train, y_train, X_valid, y_valid, model_kwargs
    )

    if X_test.shape[0] > 0:
        print("\n=== 测试集评估 ===")
        y_test_pred = rf_model.predict(X_test)
        test_metrics = compute_metrics(y_test, y_test_pred)
        test_metrics_per_col = compute_column_metrics(y_test, y_test_pred, target_cols)

        print(
            f"测试集指标: RMSE={test_metrics['rmse']:.4f}, "
            f"MAE={test_metrics['mae']:.4f}, R2={test_metrics['r2']:.4f}"
        )
        print("\n测试集各列指标:")
        for col in target_cols:
            m = test_metrics_per_col[col]
            print(
                f"  {col}: RMSE={m['rmse']:.4f}, "
                f"MAE={m['mae']:.4f}, R2={m['r2']:.4f}"
            )
    else:
        test_metrics = {"rmse": 0.0, "mae": 0.0, "r2": 0.0}
        test_metrics_per_col = {col: {"rmse": 0.0, "mae": 0.0, "r2": 0.0} for col in target_cols}
        y_test_pred = np.array([]).reshape(0, len(target_cols))
        print("\n警告: 测试集为空，已跳过测试评估。")

    print("\n=== 保存结果 ===")
    save_model(rf_model, str(checkpoints_output_dir))
    save_results(ts_train, y_train, y_train_pred, target_cols, str(output_dir), prefix="train_")
    save_results(ts_valid, y_valid, y_valid_pred, target_cols, str(output_dir), prefix="valid_")
    if X_test.shape[0] > 0:
        save_results(ts_test, y_test, y_test_pred, target_cols, str(output_dir), prefix="test_")
    save_metrics_log(train_metrics, valid_metrics, test_metrics, test_metrics_per_col, target_cols, str(output_dir))
    plot_loss_curve(train_history, valid_history, str(output_dir), metric_name="rmse")

    print("\n训练完成！")


if __name__ == "__main__":
    main()

