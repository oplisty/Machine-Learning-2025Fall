import os
from typing import Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def load_data(csv_path: str) -> pd.DataFrame:
    """Load and sort the original k-line csv."""
    df = pd.read_csv(csv_path)
    df["timestamps"] = pd.to_datetime(df["timestamps"])
    df = df.sort_values("timestamps").reset_index(drop=True)
    return df


def build_supervised(
    df: pd.DataFrame,
    feature_cols: list,
    target_cols: list,
    lookback: int = 5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a simple supervised dataset:
    X: past `lookback` days of features
    y: next day targets.
    """
    values = df[feature_cols].values.astype(float)
    targets = df[target_cols].values.astype(float)
    timestamps = df["timestamps"].values

    X_list, y_list, ts_list = [], [], []
    for i in range(lookback, len(df) - 1):
        X_window = values[i - lookback : i].reshape(-1)
        y_next = targets[i + 0]  # predict same day as last context index
        X_list.append(X_window)
        y_list.append(y_next)
        ts_list.append(timestamps[i])

    X = np.array(X_list)
    y = np.array(y_list)
    ts_arr = np.array(ts_list)
    return X, y, ts_arr


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute basic regression metrics."""
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train_and_evaluate_models(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[Dict[str, object], Dict[str, Dict[str, float]], np.ndarray, Dict[str, np.ndarray]]:
    """
    Train and evaluate multiple regression models on the same train/test split.

    Returns:
        models: dict of trained models
        metrics: dict[model_name] -> metrics dict
        y_test: ground truth targets for test split
        preds: dict[model_name] -> prediction array
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )

    models: Dict[str, object] = {
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            random_state=random_state,
            n_jobs=-1,
        ),
        "GradientBoosting": MultiOutputRegressor(
            GradientBoostingRegressor(random_state=random_state)
        ),
        "LinearRegression": LinearRegression(),
    }

    metrics: Dict[str, Dict[str, float]] = {}
    preds: Dict[str, np.ndarray] = {}

    for name, model in models.items():
        print(f"\nTraining model: {name}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        preds[name] = y_pred
        metrics[name] = compute_metrics(y_test, y_pred)
        print(f"  Metrics for {name}: RMSE={metrics[name]['rmse']:.4f}, "
              f"MAE={metrics[name]['mae']:.4f}, R2={metrics[name]['r2']:.4f}")

    return models, metrics, y_test, preds


def save_results_per_model(
    model_name: str,
    timestamps: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    target_cols: list,
    output_dir: str,
) -> None:
    """Save per-model prediction CSV and simple plots for close & volume."""
    os.makedirs(output_dir, exist_ok=True)

    result_df = pd.DataFrame(
        y_test, columns=[f"true_{c}" for c in target_cols]
    )
    for i, c in enumerate(target_cols):
        result_df[f"pred_{c}"] = y_pred[:, i]
    result_df["timestamp"] = timestamps[-len(result_df) :]

    csv_path = os.path.join(output_dir, f"{model_name}_results.csv")
    result_df.to_csv(csv_path, index=False)
    print(f"Saved results for {model_name} to: {csv_path}")

    # Plot close & volume if present
    for col in ["close", "volume"]:
        if col in target_cols:
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
            plt.title(f"{model_name} forecast for {col}")
            plt.xlabel("Time")
            plt.ylabel(col)
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            fig_path = os.path.join(output_dir, f"{model_name}_forecast_{col}.png")
            plt.savefig(fig_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"Saved plot for {col} ({model_name}) to: {fig_path}")


def save_comparison_plots(
    timestamps: np.ndarray,
    y_test: np.ndarray,
    preds: Dict[str, np.ndarray],
    target_cols: list,
    output_dir: str,
) -> None:
    """Plot comparison of multiple models for key targets."""
    os.makedirs(output_dir, exist_ok=True)

    ts = timestamps[-len(y_test) :]

    for col in ["close", "volume"]:
        if col not in target_cols:
            continue
        idx = target_cols.index(col)

        plt.figure(figsize=(10, 4))
        plt.plot(ts, y_test[:, idx], label="True", color="black", linewidth=1.4)

        for name, pred in preds.items():
            plt.plot(ts, pred[:, idx], label=name, linewidth=1.0)

        plt.title(f"Comparison of models for {col}")
        plt.xlabel("Time")
        plt.ylabel(col)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        fig_path = os.path.join(output_dir, f"comparison_{col}.png")
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved comparison plot for {col} to: {fig_path}")


def save_metrics_log(
    metrics: Dict[str, Dict[str, float]],
    output_dir: str,
) -> None:
    """Save metrics for all models to a text log."""
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "metrics_log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("Model evaluation metrics (lower RMSE/MAE is better, higher R2 is better):\n")
        for name, m in metrics.items():
            line = (f"{name}: RMSE={m['rmse']:.6f}, "
                    f"MAE={m['mae']:.6f}, R2={m['r2']:.6f}\n")
            f.write(line)
    print(f"Saved metrics log to: {log_path}")


def main():
    base_dir = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
    data_path = os.path.join(base_dir, "data", "data.csv")
    output_dir = os.path.join(base_dir, "output", "ml_forecast")

    print(f"Loading data from: {data_path}")
    df = load_data(data_path)
    print(
        f"Data time range: {df['timestamps'].min()} -> {df['timestamps'].max()}, "
        f"total rows: {len(df)}"
    )

    feature_cols = ["open", "high", "low", "close", "volume", "amount"]
    target_cols = ["open", "high", "low", "close", "volume", "amount"]

    lookback = 10
    X, y, ts = build_supervised(df, feature_cols, target_cols, lookback=lookback)
    print(f"Built supervised dataset: X={X.shape}, y={y.shape}")

    models, metrics, y_test, preds = train_and_evaluate_models(X, y)

    print("\nSummary metrics for all models:")
    for name, m in metrics.items():
        print(f"  {name}: RMSE={m['rmse']:.4f}, MAE={m['mae']:.4f}, R2={m['r2']:.4f}")

    # Only last len(y_test) timestamps are used as test timestamps
    test_ts = ts[-len(y_test) :]

    # Save per-model results and plots
    for name, y_pred in preds.items():
        save_results_per_model(name, test_ts, y_test, y_pred, target_cols, output_dir)

    # Save comparison plots and metrics log
    save_comparison_plots(test_ts, y_test, preds, target_cols, output_dir)
    save_metrics_log(metrics, output_dir)


if __name__ == "__main__":
    main()


