import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
sys.path.append("/data3/zepeng/xyx/Machine-Learning-2025Fall")
from Model.model import Kronos, KronosTokenizer, KronosPredictor


def plot_prediction(kline_df, pred_df, save_path=None):
    pred_df.index = kline_df.index[-pred_df.shape[0]:]
    sr_close = kline_df['close']
    sr_pred_close = pred_df['close']
    sr_close.name = 'Ground Truth'
    sr_pred_close.name = "Prediction"

    sr_volume = kline_df['volume']
    sr_pred_volume = pred_df['volume']
    sr_volume.name = 'Ground Truth'
    sr_pred_volume.name = "Prediction"

    close_df = pd.concat([sr_close, sr_pred_close], axis=1)
    volume_df = pd.concat([sr_volume, sr_pred_volume], axis=1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax1.plot(close_df['Ground Truth'], label='Ground Truth', color='blue', linewidth=1.5)
    ax1.plot(close_df['Prediction'], label='Prediction', color='red', linewidth=1.5)
    ax1.set_ylabel('Close Price', fontsize=14)
    ax1.legend(loc='lower left', fontsize=12)
    ax1.grid(True)

    ax2.plot(volume_df['Ground Truth'], label='Ground Truth', color='blue', linewidth=1.5)
    ax2.plot(volume_df['Prediction'], label='Prediction', color='red', linewidth=1.5)
    ax2.set_ylabel('Volume', fontsize=14)
    ax2.legend(loc='upper left', fontsize=12)
    ax2.grid(True)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")


# 1. Load Model and Tokenizer
tokenizer = KronosTokenizer.from_pretrained("/data3/zepeng/xyx/Machine-Learning-2025Fall/checkpoints/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("/data3/zepeng/xyx/Machine-Learning-2025Fall/checkpoints/Kronos-base")

# 2. Instantiate Predictor
predictor = KronosPredictor(model, tokenizer, device="cuda:4", max_context=512)

# 3. Prepare Data
df = pd.read_csv("/data3/zepeng/xyx/Machine-Learning-2025Fall/data/data.csv")
df['timestamps'] = pd.to_datetime(df['timestamps'])

# lookback = 400
# pred_len = 120

# x_df = df.loc[:lookback-1, ['open', 'high', 'low', 'close', 'volume', 'amount']]
# x_timestamp = df.loc[:lookback-1, 'timestamps']
# y_timestamp = df.loc[lookback:lookback+pred_len-1, 'timestamps']

# # 4. Make Prediction
# pred_df = predictor.predict(
#     df=x_df,
#     x_timestamp=x_timestamp,
#     y_timestamp=y_timestamp,
#     pred_len=pred_len,
#     T=1.0,
#     top_p=0.9,
#     sample_count=1,
#     verbose=True
# )

# # 5. Visualize Results
# print("Forecasted Data Head:")
# print(pred_df.head())

# # Combine historical and forecasted data for plotting
# kline_df = df.loc[:lookback+pred_len-1]

# # visualize
# plot_prediction(kline_df, pred_df)
start_date = pd.Timestamp("2020-01-01")
end_date = pd.Timestamp("2025-12-31")
mask = (df['timestamps'] >= start_date) & (df['timestamps'] <= end_date)
df = df.loc[mask].reset_index(drop=True)

if df.empty:
    raise ValueError("No data available between 2020-01-01 and 2025-12-31.")

lookback = 400
pred_len = 100
step = pred_len  # move window forward by prediction horizon
price_cols = ['open', 'high', 'low', 'close', 'volume', 'amount']

if len(df) < lookback + pred_len:
    raise ValueError(f"Not enough data in selected range. Need at least {lookback + pred_len} rows, got {len(df)}.")

# 4. Rolling Predictions across the entire period
all_preds = []
window_id = 0
start_idx = 0
plot_data = None

while start_idx + lookback + pred_len <= len(df):
    end_hist = start_idx + lookback
    end_pred = end_hist + pred_len

    x_df = df.loc[start_idx:end_hist-1, price_cols]
    x_timestamp = df.loc[start_idx:end_hist-1, 'timestamps']
    y_timestamp = df.loc[end_hist:end_pred-1, 'timestamps']

    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=pred_len,
        T=1.0,
        top_p=0.9,
        sample_count=1,
        verbose=False
    )
    pred_df = pred_df.copy()
    pred_df['timestamp'] = y_timestamp.values

    pred_export = pred_df.copy()
    pred_export['window_id'] = window_id
    pred_export['window_start'] = x_timestamp.iloc[0]
    pred_export['window_end'] = x_timestamp.iloc[-1]
    all_preds.append(pred_export)

    kline_df = df.loc[start_idx:end_pred-1].copy()
    plot_data = (kline_df, pred_df.copy())

    window_id += 1
    start_idx += step

if not all_preds:
    raise RuntimeError("Sliding window produced no predictions. Check lookback/pred_len configuration.")

# 5. Save combined predictions
output_dir = "/data3/zepeng/xyx/Machine-Learning-2025Fall/output/"
os.makedirs(output_dir, exist_ok=True)

combined_pred_df = pd.concat(all_preds,ignore_index=True)
combined_pred_df['timestamp'] = pd.to_datetime(combined_pred_df['timestamp'])
combined_csv_path = os.path.join(output_dir, "predictions_2020_2025.csv")
combined_pred_df.to_csv(combined_csv_path)
print(f"Saved aggregated predictions to: {combined_csv_path}")
print("Forecasted Data Head:")
print(combined_pred_df.head())

# visualize & save (using the last window)
if plot_data is not None:
    kline_df_last, pred_df_last = plot_data
    output_path = os.path.join(output_dir, "predicted_image/predicted_image.png")
    plot_prediction(kline_df_last, pred_df_last, save_path=output_path)
    print(f"Saved visualization to: {output_path}")

 #visualize predicted curve for 2021-2025
pred_start = pd.Timestamp("2021-01-01")
pred_end = pd.Timestamp("2025-12-31")
pred_mask = (combined_pred_df['timestamp'] >= pred_start) & (combined_pred_df['timestamp'] <= pred_end)
pred_range_df = combined_pred_df.loc[pred_mask].copy()

if pred_range_df.empty:
    print("No predicted timestamps fall within 2021-01-01 to 2025-12-31.")
else:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax1.plot(pred_range_df['timestamp'], pred_range_df['close'], color='red', label='Predicted Close')
    ax1.set_ylabel('Close Price')
    ax1.legend(loc='upper left')
    ax1.grid(True)

    ax2.plot(pred_range_df['timestamp'], pred_range_df['volume'], color='purple', label='Predicted Volume')
    ax2.set_xlabel('Timestamp')
    ax2.set_ylabel('Volume')
    ax2.legend(loc='upper left')
    ax2.grid(True)

    plt.tight_layout()
    range_plot_path = os.path.join(output_dir, "predicted_curve_2021_2025.png")
    plt.savefig(range_plot_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved 2021-2025 prediction curve to: {range_plot_path}")
