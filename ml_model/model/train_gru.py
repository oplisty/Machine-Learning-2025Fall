import os
import pickle
from typing import Dict, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
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


def build_sequences(
    df: pd.DataFrame,
    feature_cols: list,
    lookback: int = 20,
    horizon: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    构建监督学习序列：
    X: shape (N, lookback, num_features) - 过去 `lookback` 步
    y: shape (N, num_features) - 未来 horizon 步后的目标值
    """
    values = df[feature_cols].values.astype(np.float32)
    timestamps = df["timestamps"].values

    X_list, y_list, ts_list = [], [], []
    for i in range(lookback, len(df) - horizon + 1):
        X_window = values[i - lookback : i]
        y_next = values[i + horizon - 1]
        X_list.append(X_window)
        y_list.append(y_next)
        ts_list.append(timestamps[i + horizon - 1])

    X = np.stack(X_list, axis=0)
    y = np.stack(y_list, axis=0)
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

    X_train, y_train, ts_train = X[train_mask], y[train_mask], timestamps[train_mask]
    X_valid, y_valid, ts_valid = X[valid_mask], y[valid_mask], timestamps[valid_mask]
    X_test, y_test, ts_test = X[test_mask], y[test_mask], timestamps[test_mask]

    return X_train, y_train, ts_train, X_valid, y_valid, ts_valid, X_test, y_test, ts_test


class GRURegressor(nn.Module):
    """GRU 回归模型用于多元时间序列预测"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        output_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        out, _ = self.gru(x)  # out: (B, T, H)
        last = out[:, -1, :]  # 使用最后一个隐藏状态
        return self.fc(last)  # (B, output_dim)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """计算全局回归指标"""
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def compute_column_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, target_cols: list
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


def train_gru_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    model_kwargs: Dict,
    target_cols: list,
    device: torch.device,
) -> Tuple[nn.Module, Dict[str, float], Dict[str, float], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """训练 GRU 模型，记录训练/验证损失曲线"""
    print("\n=== 训练 GRU 模型 ===")

    input_dim = X_train.shape[-1]
    output_dim = y_train.shape[-1]

    model = GRURegressor(
        input_dim=input_dim,
        hidden_dim=model_kwargs.get("hidden_dim", 64),
        num_layers=model_kwargs.get("num_layers", 2),
        output_dim=output_dim,
        dropout=model_kwargs.get("dropout", 0.1),
    ).to(device)

    criterion = nn.MSELoss()
    lr = model_kwargs.get("learning_rate", 1e-3)
    optimizer_name = model_kwargs.get("optimizer", "adam").lower()
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    train_loader = DataLoader(train_ds, batch_size=model_kwargs.get("batch_size", 32), shuffle=True, drop_last=False)
    
    valid_ds = TensorDataset(torch.from_numpy(X_valid), torch.from_numpy(y_valid))
    valid_loader = DataLoader(valid_ds, batch_size=model_kwargs.get("batch_size", 32), shuffle=False, drop_last=False)

    epochs = model_kwargs.get("epochs", 50)
    train_losses, valid_losses = [], []

    model.train()
    for epoch in range(1, epochs + 1):
        epoch_train_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * xb.size(0)

        epoch_train_loss /= len(train_ds)
        train_losses.append(epoch_train_loss)

        # 验证集损失
        model.eval()
        epoch_valid_loss = 0.0
        with torch.no_grad():
            for xb, yb in valid_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                epoch_valid_loss += loss.item() * xb.size(0)
        epoch_valid_loss /= len(valid_ds)
        valid_losses.append(epoch_valid_loss)
        model.train()

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{epochs} - Train Loss: {epoch_train_loss:.6f}, Valid Loss: {epoch_valid_loss:.6f}")

    # 最终预测
    model.eval()
    with torch.no_grad():
        X_train_t = torch.from_numpy(X_train).to(device)
        X_valid_t = torch.from_numpy(X_valid).to(device)
        y_train_pred = model(X_train_t).cpu().numpy()
        y_valid_pred = model(X_valid_t).cpu().numpy()

    train_metrics = compute_metrics(y_train, y_train_pred)
    valid_metrics = compute_metrics(y_valid, y_valid_pred)

    print(f"\n训练集指标: RMSE={train_metrics['rmse']:.4f}, "
          f"MAE={train_metrics['mae']:.4f}, R2={train_metrics['r2']:.4f}")
    print(f"验证集指标: RMSE={valid_metrics['rmse']:.4f}, "
          f"MAE={valid_metrics['mae']:.4f}, R2={valid_metrics['r2']:.4f}")

    return model, train_metrics, valid_metrics, y_train_pred, y_valid_pred, np.array(train_losses), np.array(valid_losses)


def plot_loss_curve(
    train_history: np.ndarray,
    valid_history: np.ndarray,
    output_dir: str,
    metric_name: str = "mse",
) -> None:
    """绘制训练/验证损失曲线"""
    if train_history.size == 0 or valid_history.size == 0:
        return
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(train_history, label="训练损失", linewidth=1.5)
    plt.plot(valid_history, label="验证损失", linewidth=1.5)
    plt.xlabel("Epoch")
    plt.ylabel(metric_name.upper())
    plt.title(f"GRU 训练 vs 验证 {metric_name.upper()}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    fig_path = os.path.join(output_dir, f"loss_curve_{metric_name}.png")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"保存训练/验证曲线到: {fig_path}")


def save_results(
    timestamps: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_cols: list,
    output_dir: str,
    prefix: str = "",
) -> None:
    """保存预测结果 CSV 和图表"""
    os.makedirs(output_dir, exist_ok=True)

    result_df = pd.DataFrame(y_true, columns=[f"true_{c}" for c in target_cols])
    for i, c in enumerate(target_cols):
        result_df[f"pred_{c}"] = y_pred[:, i]
    result_df["timestamp"] = timestamps[-len(result_df):]

    csv_path = os.path.join(output_dir, f"{prefix}results.csv")
    result_df.to_csv(csv_path, index=False)
    print(f"保存结果到: {csv_path}")

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
            plt.title(f"GRU {col} 预测结果 ({prefix})")
            plt.xlabel("时间")
            plt.ylabel(col)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            fig_path = os.path.join(output_dir, f"{prefix}forecast_{col}.png")
            plt.savefig(fig_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"保存图表到: {fig_path}")


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
        f.write("GRU 模型评估指标 (RMSE/MAE 越小越好, R2 越大越好):\n")
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


def save_model(model: nn.Module, output_dir: str) -> None:
    """保存模型"""
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "gru_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"保存模型到: {model_path}")


@hydra.main(version_base=None, config_path="../config", config_name="gru")
def main(cfg: DictConfig):
    base_dir = Path(get_original_cwd())

    data_config = cfg.data
    model_config = cfg.model
    model_kwargs = OmegaConf.to_container(model_config.kwargs, resolve=True)
    output_dir_cfg = cfg.output_dir
    checkpoints_output_dir_cfg = cfg.get("checkpoints_output_dir", "ml_model/checkpoints/gru")
    horizon = int(cfg.get("horizon", 1))
    lookback = int(cfg.get("lookback", 20))

    data_path = base_dir / data_config["data_path"]
    output_dir = base_dir / output_dir_cfg / f"horizon_{horizon}"
    checkpoints_output_dir = base_dir / checkpoints_output_dir_cfg / f"horizon_{horizon}"

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
    X, y, ts = build_sequences(df_fit, feature_cols, lookback=lookback, horizon=horizon)
    print(f"数据集形状: X={X.shape}, y={y.shape}")

    train_range = data_config["train"]
    valid_range = data_config["valid"]

    print("\n划分数据集:")
    print(f"  训练集: {train_range[0]} -> {train_range[1]}")
    print(f"  验证集: {valid_range[0]} -> {valid_range[1]}")
    print(f"  测试集: {test_range[0]} -> {test_range[1]}")

    X_train, y_train, ts_train, X_valid, y_valid, ts_valid, X_test, y_test, ts_test = split_by_time(
        X, y, ts, train_range, valid_range, test_range
    )

    print("\n数据集大小:")
    print(f"  训练集: {X_train.shape[0]} 样本")
    print(f"  验证集: {X_valid.shape[0]} 样本")
    print(f"  测试集: {X_test.shape[0]} 样本")

    if X_test.shape[0] == 0:
        print("\n警告: 测试集为空！请检查 fit_end_time 是否包含测试集时间范围。")
        print(f"当前 fit_end_time: {fit_end}")
        print(f"测试集时间范围: {test_range[0]} -> {test_range[1]}")
        print("将跳过测试集评估。")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")

    model, train_metrics, valid_metrics, y_train_pred, y_valid_pred, train_losses, valid_losses = train_gru_model(
        X_train, y_train, X_valid, y_valid, model_kwargs, target_cols, device
    )

    if X_test.shape[0] > 0:
        print("\n=== 测试集评估 ===")
        model.eval()
        with torch.no_grad():
            X_test_t = torch.from_numpy(X_test).to(device)
            y_test_pred = model(X_test_t).cpu().numpy()
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
        test_metrics = {"rmse": 0.0, "mae": 0.0, "r2": 0.0}
        test_metrics_per_col = {col: {"rmse": 0.0, "mae": 0.0, "r2": 0.0} for col in target_cols}
        y_test_pred = np.array([]).reshape(0, len(target_cols))

    print("\n=== 保存结果 ===")
    save_model(model, str(checkpoints_output_dir))
    save_results(ts_train, y_train, y_train_pred, target_cols, str(output_dir), prefix="train_")
    save_results(ts_valid, y_valid, y_valid_pred, target_cols, str(output_dir), prefix="valid_")
    if X_test.shape[0] > 0:
        save_results(ts_test, y_test, y_test_pred, target_cols, str(output_dir), prefix="test_")
    save_metrics_log(train_metrics, valid_metrics, test_metrics, test_metrics_per_col, target_cols, str(output_dir))
    plot_loss_curve(train_losses, valid_losses, str(output_dir), metric_name="mse")

    print("\n训练完成！")


if __name__ == "__main__":
    main()

