import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt


def load_data(csv_path: str) -> pd.DataFrame:
    """Load and sort the original k-line csv."""
    df = pd.read_csv(csv_path)
    df["timestamps"] = pd.to_datetime(df["timestamps"])
    df = df.sort_values("timestamps").reset_index(drop=True)
    return df


def build_sequences(
    df: pd.DataFrame,
    feature_cols: list,
    lookback: int = 20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build supervised sequences:
      X: shape (N, lookback, num_features) - past `lookback` steps
      y: shape (N, num_features) - current step targets
    """
    values = df[feature_cols].values.astype(np.float32)
    timestamps = df["timestamps"].values

    X_list, y_list, ts_list = [], [], []
    for i in range(lookback, len(df)):
        X_window = values[i - lookback : i]
        y_curr = values[i]
        X_list.append(X_window)
        y_list.append(y_curr)
        ts_list.append(timestamps[i])

    X = np.stack(X_list, axis=0)
    y = np.stack(y_list, axis=0)
    ts_arr = np.array(ts_list)
    return X, y, ts_arr


class RNNRegressor(nn.Module):
    """Simple LSTM/GRU regressor for multivariate time series."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        output_dim: int,
        rnn_type: str = "LSTM",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.rnn_type = rnn_type.upper()
        if self.rnn_type == "LSTM":
            self.rnn = nn.LSTM(
                input_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
        elif self.rnn_type == "GRU":
            self.rnn = nn.GRU(
                input_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
        else:
            raise ValueError(f"Unsupported rnn_type: {rnn_type}")

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        out, _ = self.rnn(x)  # out: (B, T, H)
        last = out[:, -1, :]  # use last hidden state
        return self.fc(last)  # (B, output_dim)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train_rnn_model(
    model_name: str,
    rnn_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: torch.device,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
) -> Tuple[nn.Module, Dict[str, float], np.ndarray]:
    print(f"\n=== Training {model_name} ({rnn_type}) ===")

    input_dim = X_train.shape[-1]
    output_dim = y_train.shape[-1]

    model = RNNRegressor(
        input_dim=input_dim,
        hidden_dim=64,
        num_layers=2,
        output_dim=output_dim,
        rnn_type=rnn_type,
        dropout=0.1,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_ds = TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train)
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=False
    )

    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)

        epoch_loss /= len(train_ds)
        if epoch % 10 == 0 or epoch == 1:
            print(f"[{model_name}] Epoch {epoch}/{epochs} - Train MSE: {epoch_loss:.6f}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        X_test_t = torch.from_numpy(X_test).to(device)
        preds = model(X_test_t).cpu().numpy()

    metrics = compute_metrics(y_test, preds)
    print(
        f"[{model_name}] Metrics: RMSE={metrics['rmse']:.4f}, "
        f"MAE={metrics['mae']:.4f}, R2={metrics['r2']:.4f}"
    )
    return model, metrics, preds


def save_per_model_results(
    model_name: str,
    timestamps: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    feature_cols: list,
    output_dir: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    result_df = pd.DataFrame(
        y_test, columns=[f"true_{c}" for c in feature_cols]
    )
    for i, c in enumerate(feature_cols):
        result_df[f"pred_{c}"] = y_pred[:, i]
    result_df["timestamp"] = timestamps[-len(result_df) :]

    csv_path = os.path.join(output_dir, f"{model_name}_results.csv")
    result_df.to_csv(csv_path, index=False)
    print(f"Saved results for {model_name} to: {csv_path}")

    for col in ["close", "volume"]:
        if col in feature_cols:
            plt.figure(figsize=(10, 4))
            plt.plot(
                result_df["timestamp"],
                result_df[f"true_{col}"],
                label="True",
                color="blue",
                linewidth=1.2,
            )
            plt.plot(
                result_df["timestamp"],
                result_df[f"pred_{col}"],
                label=f"{model_name} Pred",
                linewidth=1.2,
            )
            plt.title(f"{model_name} {col} prediction")
            plt.xlabel("Time")
            plt.ylabel(col)
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            fig_path = os.path.join(
                output_dir, f"{model_name}_forecast_{col}.png"
            )
            plt.savefig(fig_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"Saved plot for {col} ({model_name}) to: {fig_path}")


def save_rnn_comparison_plots(
    timestamps: np.ndarray,
    y_test: np.ndarray,
    preds_dict: Dict[str, np.ndarray],
    feature_cols: list,
    output_dir: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    ts = timestamps[-len(y_test) :]

    for col in ["close", "volume"]:
        if col not in feature_cols:
            continue
        idx = feature_cols.index(col)

        plt.figure(figsize=(10, 4))
        plt.plot(ts, y_test[:, idx], label="True", color="black", linewidth=1.4)
        for name, pred in preds_dict.items():
            plt.plot(ts, pred[:, idx], label=name, linewidth=1.0)
        plt.title(f"LSTM/GRU comparison for {col}")
        plt.xlabel("Time")
        plt.ylabel(col)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        fig_path = os.path.join(output_dir, f"rnn_comparison_{col}.png")
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved RNN comparison plot for {col} to: {fig_path}")


def save_rnn_metrics_log(
    metrics: Dict[str, Dict[str, float]],
    output_dir: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "rnn_metrics_log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(
            "LSTM/GRU model metrics (lower RMSE/MAE is better, higher R2 is better):\n"
        )
        for name, m in metrics.items():
            line = (
                f"{name}: RMSE={m['rmse']:.6f}, "
                f"MAE={m['mae']:.6f}, R2={m['r2']:.6f}\n"
            )
            f.write(line)
    print(f"Saved RNN metrics log to: {log_path}")


def main():
    base_dir = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
    data_path = os.path.join(base_dir, "data", "data.csv")
    output_dir = os.path.join(base_dir, "output", "rnn_forecast")

    print(f"Loading data from: {data_path}")
    df = load_data(data_path)
    print(
        f"Data time range: {df['timestamps'].min()} -> {df['timestamps'].max()}, "
        f"total rows: {len(df)}"
    )

    feature_cols = ["open", "high", "low", "close", "volume", "amount"]
    lookback = 20

    X, y, ts = build_sequences(df, feature_cols, lookback=lookback)
    print(f"Built sequence dataset: X={X.shape}, y={y.shape}")

    # Train/test split (time-based, no shuffle)
    test_ratio = 0.2
    split_idx = int(len(X) * (1 - test_ratio))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    ts_train, ts_test = ts[:split_idx], ts[split_idx:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    metrics_all: Dict[str, Dict[str, float]] = {}
    preds_all: Dict[str, np.ndarray] = {}

    # LSTM
    lstm_model, lstm_metrics, lstm_preds = train_rnn_model(
        model_name="LSTM",
        rnn_type="LSTM",
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        device=device,
        epochs=50,
        batch_size=32,
        lr=1e-3,
    )
    metrics_all["LSTM"] = lstm_metrics
    preds_all["LSTM"] = lstm_preds

    # GRU
    gru_model, gru_metrics, gru_preds = train_rnn_model(
        model_name="GRU",
        rnn_type="GRU",
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        device=device,
        epochs=50,
        batch_size=32,
        lr=1e-3,
    )
    metrics_all["GRU"] = gru_metrics
    preds_all["GRU"] = gru_preds

    # Save per-model results
    for name, preds in preds_all.items():
        save_per_model_results(
            name, ts_test, y_test, preds, feature_cols, output_dir
        )

    # Comparison plots and metrics log
    save_rnn_comparison_plots(ts_test, y_test, preds_all, feature_cols, output_dir)
    save_rnn_metrics_log(metrics_all, output_dir)


if __name__ == "__main__":
    main()


