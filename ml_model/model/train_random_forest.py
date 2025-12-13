import os
import pickle
import yaml
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import ParameterGrid, ParameterSampler
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


def save_best_parameters(best_params: Dict, train_rmse: float, valid_rmse: float, search_config: Dict, output_dir: str) -> None:
    """保存最佳参数到文件"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存为 YAML 格式
    params_file = os.path.join(output_dir, "best_parameters.yaml")
    with open(params_file, "w", encoding="utf-8") as f:
        yaml.dump({
            "best_parameters": best_params,
            "train_rmse": float(train_rmse),
            "valid_rmse": float(valid_rmse),
            "method": search_config.get("method", "randomized"),
            "n_iter": search_config.get("n_iter", 50) if search_config.get("method", "randomized") == "randomized" else None,
        }, f, default_flow_style=False, allow_unicode=True, indent=2)
    print(f"保存最佳参数到: {params_file}")
    
    # 保存为文本格式（便于阅读）
    txt_file = os.path.join(output_dir, "best_parameters.txt")
    with open(txt_file, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("随机森林最佳参数搜索结果\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"搜索方法: {search_config.get('method', 'randomized')}\n")
        if search_config.get("method", "randomized") == "randomized":
            f.write(f"迭代次数: {search_config.get('n_iter', 50)}\n")
        f.write(f"训练集 RMSE: {train_rmse:.6f}\n")
        f.write(f"验证集 RMSE: {valid_rmse:.6f}\n\n")
        f.write("最佳参数:\n")
        f.write("-" * 60 + "\n")
        for key, value in best_params.items():
            f.write(f"  {key}: {value}\n")
    print(f"保存最佳参数（文本格式）到: {txt_file}")


def search_best_parameters(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    search_config: Dict,
    base_params: Dict,
    output_dir: str = None,
) -> Dict:
    """
    使用启发式搜索找到最佳参数
    基于训练集和验证集的 RMSE 选择最佳参数（验证集 RMSE 最小）
    不对 n_estimators 进行调参
    """
    print("\n=== 开始参数搜索 ===")
    print("基于训练集和验证集损失选择最佳参数（验证集 RMSE 最小）")
    
    # search_config 已经是普通 dict，直接使用
    param_distributions = search_config.get("param_distributions", {})
    method = search_config.get("method", "randomized").lower()
    n_iter = int(search_config.get("n_iter", 50))
    
    # 确保不搜索 n_estimators
    if "n_estimators" in param_distributions:
        print("警告: 从搜索参数中移除 n_estimators（不对此参数进行调参）")
        param_distributions = {k: v for k, v in param_distributions.items() if k != "n_estimators"}
    
    # 合并基础参数（用于固定参数，如 n_jobs, random_state, n_estimators）
    fixed_params = {k: v for k, v in base_params.items() if k not in param_distributions}
    n_estimators = fixed_params.get("n_estimators", base_params.get("n_estimators", 100))
    
    # 生成参数组合
    if method == "randomized":
        print(f"使用随机搜索 (n_iter={n_iter})...")
        param_list = list(ParameterSampler(
            param_distributions,
            n_iter=n_iter,
            random_state=base_params.get("random_state", 42)
        ))
    else:  # grid search
        print("使用网格搜索...")
        param_list = list(ParameterGrid(param_distributions))
        print(f"总共 {len(param_list)} 个参数组合")
    
    best_params = None
    best_valid_rmse = float('inf')
    best_train_rmse = float('inf')
    best_idx = -1
    
    print(f"\n开始评估 {len(param_list)} 个参数组合...")
    for idx, params in enumerate(param_list):
        # 合并固定参数和搜索参数
        current_params = {**fixed_params, **params}
        
        # 创建并训练模型
        model = RandomForestRegressor(**current_params)
        model.fit(X_train, y_train)
        
        # 计算训练集和验证集 RMSE
        y_train_pred = model.predict(X_train)
        y_valid_pred = model.predict(X_valid)
        
        # 使用按列标准化的 RMSE（每个列的 RMSE 取平均），避免被大数值列主导
        # 或者使用加权平均，这里使用简单的列平均
        train_rmse_per_col = []
        valid_rmse_per_col = []
        for col_idx in range(y_train.shape[1]):
            train_rmse_col = float(np.sqrt(mean_squared_error(y_train[:, col_idx], y_train_pred[:, col_idx])))
            valid_rmse_col = float(np.sqrt(mean_squared_error(y_valid[:, col_idx], y_valid_pred[:, col_idx])))
            train_rmse_per_col.append(train_rmse_col)
            valid_rmse_per_col.append(valid_rmse_col)
        
        # 使用各列 RMSE 的平均值作为总体 RMSE
        train_rmse = float(np.mean(train_rmse_per_col))
        valid_rmse = float(np.mean(valid_rmse_per_col))
        
        # 选择验证集 RMSE 最小的参数
        if valid_rmse < best_valid_rmse:
            best_valid_rmse = valid_rmse
            best_train_rmse = train_rmse
            best_params = current_params.copy()
            best_idx = idx
        
        # 每 10 个组合打印一次进度
        if (idx + 1) % 10 == 0 or (idx + 1) == len(param_list):
            print(f"  进度: {idx + 1}/{len(param_list)} | "
                  f"当前: train_rmse={train_rmse:.4f}, valid_rmse={valid_rmse:.4f} | "
                  f"最佳: valid_rmse={best_valid_rmse:.4f}")
    
    print(f"\n最佳参数组合 (第 {best_idx + 1} 个):")
    print(f"  训练集 RMSE: {best_train_rmse:.6f}")
    print(f"  验证集 RMSE: {best_valid_rmse:.6f}")
    print(f"  参数: {best_params}")
    
    # 保存最佳参数到输出目录
    if output_dir:
        save_best_parameters(best_params, best_train_rmse, best_valid_rmse, search_config, output_dir)
    
    # 返回最佳参数（不包含 n_estimators，因为它已经在 base_params 中）
    return {k: v for k, v in best_params.items() if k in param_distributions}


def train_random_forest_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    model_kwargs: Dict,
    search_config: Dict = None,
    output_dir: str = None,
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
    如果 search_config 启用，会先进行参数搜索
    """
    print("\n=== 训练 RandomForest 模型 ===")

    params = model_kwargs.copy()
    
    # 如果启用参数搜索，先搜索最佳参数
    if search_config and search_config.get("enabled", False):
        print("\n" + "="*60)
        print("参数搜索已启用，开始搜索最佳参数...")
        print("="*60)
        best_params = search_best_parameters(X_train, y_train, X_valid, y_valid, search_config, params, output_dir)
        # 更新参数
        params.update(best_params)
        print(f"\n使用搜索到的最佳参数: {params}")
    else:
        if search_config:
            print(f"\n参数搜索未启用 (enabled={search_config.get('enabled', False)})，使用配置文件中的默认参数")
        else:
            print("\n未配置参数搜索，使用配置文件中的默认参数")

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

    # 使用 OmegaConf 统一处理配置
    data_config = cfg.data
    model_config = cfg.model
    model_kwargs = OmegaConf.to_container(model_config.kwargs, resolve=True)
    
    # 读取参数搜索配置
    if "hyperparameter_search" in model_config:
        search_config = OmegaConf.to_container(model_config.hyperparameter_search, resolve=True)
        print(f"\n参数搜索配置: enabled={search_config.get('enabled', False)}")
    else:
        search_config = None
        print("\n未找到参数搜索配置，将使用默认参数")
    
    output_dir_cfg = cfg.output_dir
    checkpoints_output_dir_cfg = cfg.get("checkpoints_output_dir", "ml_model/checkpoints/random_forest")
    horizon = int(cfg.get("horizon", 1))
    lookback = int(cfg.get("lookback", 10))

    data_path = base_dir / data_config["data_path"]
    output_dir = base_dir / output_dir_cfg / f"horizon_{horizon}"
    checkpoints_output_dir = base_dir / checkpoints_output_dir_cfg / f"horizon_{horizon}"

    print(f"\n数据路径: {data_path}")
    print(f"输出目录: {output_dir}")

    # 加载数据
    print("\n加载数据...")
    df = load_data(str(data_path))
    print(
        f"数据时间范围: {df['timestamps'].min()} -> {df['timestamps'].max()}, "
        f"总行数: {len(df)}"
    )

    # 处理 fit 时间范围，确保覆盖测试集
    test_range = data_config["test"]
    test_end = pd.to_datetime(test_range[1])
    fit_start = data_config["fit_start_time"]
    fit_end = data_config["fit_end_time"]
    fit_end_dt = pd.to_datetime(fit_end)
    
    if fit_end_dt < test_end:
        print(f"警告: fit_end_time ({fit_end}) 早于测试集结束时间 ({test_range[1]})")
        print("自动扩展 fit_end_time 到测试集结束时间以包含测试集数据")
        fit_end = test_range[1]

    df_fit = filter_by_time_range(df, fit_start, fit_end)
    print(f"特征拟合数据范围: {fit_start} -> {fit_end}, 行数: {len(df_fit)}")

    feature_cols = ["open", "high", "low", "close", "volume", "amount"]
    target_cols = ["open", "high", "low", "close", "volume", "amount"]

    print(f"\n构建监督学习数据集 (lookback={lookback}, horizon={horizon})...")
    X, y, ts = build_supervised(df_fit, feature_cols, target_cols, lookback=lookback, horizon=horizon)
    print(f"数据集形状: X={X.shape}, y={y.shape}")

    train_range = data_config["train"]
    valid_range = data_config["valid"]

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

    # 检查测试集是否为空
    if X_test.shape[0] == 0:
        print("\n警告: 测试集为空！请检查 fit_end_time 是否包含测试集的时间范围。")
        print(f"当前 fit_end_time: {fit_end}")
        print(f"测试集时间范围: {test_range[0]} -> {test_range[1]}")
        print("将跳过测试集评估。")

    (
        rf_model,
        train_metrics,
        valid_metrics,
        y_train_pred,
        y_valid_pred,
        train_history,
        valid_history,
    ) = train_random_forest_model(
        X_train, y_train, X_valid, y_valid, model_kwargs, search_config, str(output_dir)
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

