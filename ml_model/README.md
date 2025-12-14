# Time Series Forecasting Model Library

A machine learning model library for financial time series forecasting, supporting multiple algorithms including gradient boosting trees, random forests, and deep learning models.

## 📋 Table of Contents

- [Features](#-features)
- [Supported Models](#-supported-models)
- [Requirements](#-requirements)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
- [Configuration](#-configuration)
- [Output](#-output)
- [Project Structure](#-project-structure)

## ✨ Features

- 🎯 **Multiple Model Support**: XGBoost, LightGBM, Random Forest, LSTM, GRU
- 📊 **Time Series Processing**: Automatic sliding window feature construction with customizable lookback and forecast horizons
- 🔧 **Flexible Configuration**: Hydra-based configuration management with command-line parameter overrides
- 📈 **Comprehensive Evaluation**: Multiple evaluation metrics including RMSE, MAE, R²
- 📉 **Visualization**: Automatic generation of loss curves and prediction result charts
- 🎛️ **Hyperparameter Search**: Random Forest supports automatic hyperparameter optimization

## 🤖 Supported Models

| Model | Config File | Training Script | Description |
|-------|------------|----------------|-------------|
| **XGBoost** | `config/xgboost.yaml` | `model/train_xgboost.py` | Gradient Boosting Decision Tree |
| **LightGBM** | `config/lightGBM.yaml` | `model/train_lightgbm.py` | Lightweight Gradient Boosting |
| **Random Forest** | `config/random_forest.yaml` | `model/train_random_forest.py` | Random Forest (with hyperparameter search) |
| **LSTM** | `config/lstm.yaml` | `model/train_lstm.py` | Long Short-Term Memory Network |
| **GRU** | `config/gru.yaml` | `model/train_gru.py` | Gated Recurrent Unit |
| **CatBoost** | `config/catboost.yaml` | - | Configuration ready |

## 🔧 Requirements

- Python 3.12+
- Linux (recommended) / macOS / Windows

### System Dependencies (Linux)

For XGBoost on Linux, ensure OpenMP is available:

```bash
# Ubuntu/Debian
sudo apt-get install libomp-dev

# CentOS/RHEL
sudo yum install libgomp
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Navigate to project root directory
cd Machine-Learning-2025Fall

# Install Python dependencies
pip3 install -r requirements.txt
```

### 2. Prepare Data

Ensure your data file is located at `data/data.csv` with the following columns:

- `timestamps`: Timestamp (date format)
- `open`: Opening price
- `high`: Highest price
- `low`: Lowest price
- `close`: Closing price
- `volume`: Trading volume
- `amount`: Trading amount

### 3. Train Model

#### Train XGBoost with Default Configuration

```bash
cd ml_model/model
python3 train_xgboost.py
```

#### Train with Custom Parameters

```bash
# Modify forecast horizon and lookback window
python3 train_xgboost.py horizon=3 lookback=20

# Modify model hyperparameters
python3 train_xgboost.py model.kwargs.max_depth=10 model.kwargs.n_estimators=500
```

## 📖 Usage

### Training Different Models

```bash
cd ml_model/model

# XGBoost
python3 train_xgboost.py

# LightGBM
python3 train_lightgbm.py

# Random Forest
python3 train_random_forest.py

# LSTM
python3 train_lstm.py

# GRU
python3 train_gru.py
```

### Command-Line Parameter Overrides

All models support configuration parameter overrides via command line:

```bash
# Modify forecast horizon
python3 train_xgboost.py horizon=5

# Modify lookback window
python3 train_xgboost.py lookback=30

# Modify data path
python3 train_xgboost.py data.data_path=data/your_data.csv

# Modify training set time range
python3 train_xgboost.py data.train=[2018-01-02,2023-12-31]

# Combine multiple parameters
python3 train_xgboost.py horizon=3 lookback=20 model.kwargs.max_depth=10
```

### Hyperparameter Search (Random Forest)

Random Forest model supports automatic hyperparameter search. Enable it in the configuration file:

```yaml
# config/random_forest.yaml
hyperparameter_search:
  enabled: true
  method: randomized
  n_iter: 50
```

The best parameters will be automatically searched during training and saved to `best_parameters.yaml` and `best_parameters.txt` in the output directory.

## ⚙️ Configuration

### Configuration File Structure

All configuration files are located in `ml_model/config/` directory and use YAML format:

```yaml
data:
  data_path: data/data.csv          # Data file path
  fit_start_time: 2018-01-02        # Feature fitting start time
  fit_end_time: 2023-01-01         # Feature fitting end time
  train: [2018-01-02, 2023-01-03]  # Training set time range
  valid: [2023-01-04, 2023-12-29]  # Validation set time range
  test: [2024-01-02, 2025-04-24]   # Test set time range

model:
  model_name: xgboost
  kwargs:
    # Model-specific hyperparameters
    max_depth: 8
    n_estimators: 647
    learning_rate: 0.421
    # ...

output_dir: ml_model/output/xgboost
checkpoints_output_dir: ml_model/checkpoints/xgboost

horizon: 5      # Forecast horizon (days ahead to predict)
lookback: 30    # Lookback window (days of history to use as input)
```

### Key Parameters

- **`horizon`**: Forecast window size, i.e., how many days ahead to predict
- **`lookback`**: Lookback window size, i.e., how many days of historical data to use as input features
- **`data.train/valid/test`**: Time series data split, ordered chronologically to avoid data leakage

## 📊 Output

After training, all results are saved in `ml_model/output/<model_name>/horizon_<horizon>/` directory:

### File Structure

```
output/
└── xgboost/
    └── horizon_5/
        ├── train_results.csv          # Training set predictions
        ├── valid_results.csv          # Validation set predictions
        ├── test_results.csv           # Test set predictions
        ├── metrics_log.txt            # Evaluation metrics log
        ├── loss_curve_rmse.png        # Training/validation loss curve
        ├── train_forecast_close.png   # Training set close price forecast
        ├── train_forecast_volume.png  # Training set volume forecast
        ├── valid_forecast_close.png   # Validation set close price forecast
        ├── valid_forecast_volume.png  # Validation set volume forecast
        ├── test_forecast_close.png    # Test set close price forecast
        └── test_forecast_volume.png   # Test set volume forecast
```

### Evaluation Metrics

- **RMSE** (Root Mean Squared Error): Root mean squared error, lower is better
- **MAE** (Mean Absolute Error): Mean absolute error, lower is better
- **R²** (Coefficient of Determination): Coefficient of determination, higher is better (maximum is 1)

### Model Saving

Trained models are saved in `ml_model/checkpoints/<model_name>/horizon_<horizon>/` directory:

- **XGBoost/LightGBM/Random Forest**: `*.pkl` (pickle format)
- **LSTM/GRU**: `*.pth` (PyTorch format) + `scaler_X.pkl`, `scaler_y.pkl` (scalers)

## 📁 Project Structure

```
ml_model/
├── config/                 # Configuration files directory
│   ├── xgboost.yaml
│   ├── lightGBM.yaml
│   ├── random_forest.yaml
│   ├── lstm.yaml
│   └── gru.yaml
├── model/                  # Training scripts directory
│   ├── train_xgboost.py
│   ├── train_lightgbm.py
│   ├── train_random_forest.py
│   ├── train_lstm.py
│   └── train_gru.py
├── output/                 # Output results directory
│   ├── xgboost/
│   ├── lightgbm/
│   ├── random_forest/
│   ├── lstm/
│   └── gru/
├── checkpoints/            # Model weights directory
│   ├── xgboost/
│   ├── lightgbm/
│   ├── random_forest/
│   ├── lstm/
│   └── gru/
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🔍 FAQ

### 1. XGBoost Error "OpenMP runtime is not installed" on Linux

```bash
# Ubuntu/Debian
sudo apt-get install libomp-dev
pip3 install --upgrade xgboost

# CentOS/RHEL
sudo yum install libgomp
pip3 install --upgrade xgboost
```

### 2. Empty Test Set

Check if `fit_end_time` in the configuration file includes the test set time range. The system will automatically extend `fit_end_time` to the test set end time.

### 3. How to Modify Data Path

```bash
# Method 1: Command-line override
python3 train_xgboost.py data.data_path=data/your_data.csv

# Method 2: Directly edit configuration file
# Edit data.data_path in config/xgboost.yaml
```

### 4. How to View Training Progress

Training scripts output real-time training progress and evaluation metrics. For XGBoost/LightGBM, you can view the training process through loss curve plots.

## 📝 Examples

### Complete Training Workflow

```bash
# 1. Navigate to model directory
cd ml_model/model

# 2. Train XGBoost with default configuration (predict 5 days ahead)
python3 train_xgboost.py

# 3. Train model to predict 3 days ahead
python3 train_xgboost.py horizon=3

# 4. Train Random Forest with hyperparameter search enabled
python3 train_random_forest.py

# 5. Train LSTM model
python3 train_lstm.py horizon=1 lookback=20
```

### Viewing Results

After training, you can view:

1. **Evaluation Metrics**: `output/<model>/horizon_<n>/metrics_log.txt`
2. **Prediction Charts**: `output/<model>/horizon_<n>/*_forecast_*.png`
3. **Loss Curves**: `output/<model>/horizon_<n>/loss_curve_rmse.png`
4. **Prediction Data**: `output/<model>/horizon_<n>/*_results.csv`

## 🤝 Contributing

Contributions are welcome! Please feel free to submit Issues and Pull Requests.

## 📄 License

This project is for learning and research purposes only.

---

**Tip**: For first-time users, we recommend starting with the XGBoost model as it trains quickly and produces stable results. Deep learning models (LSTM/GRU) require more training time and are recommended to run on GPU-enabled environments.
