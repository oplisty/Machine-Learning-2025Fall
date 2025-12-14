# Machine Learning: Quantitative Trading of a single stock with a small sample size

**The project web can be found [here]()**

This project is aimed at building a Machine Learning model to implement quantitative trading on the Tencent Stock based on ¥100000. Specifically , we build a model to satisfy the following requirements:

* Complete preprocessing of the provided data and mine relevant factors.
* Construct a machine learning model, build a training dataset, predict stock prices, and analyze the prediction effectiveness.
* Based on the stock prices predicted by machine learning, identify buying and selling opportunities, develop a trading strategy, and achieve index enhancement to capture excess returns (alpha).
* The strategy must aim to keep the maximum drawdown as low as possible.
* Create visualizations of the final investment results, evaluate the results against actual market data, and conduct a comprehensive assessment of the strategy's and model's overall effectiveness.

## FrameWork

![workflow](workflow.png)



## Quick Start

### Requirements Install



```shell
conda create -n mlhw python=3.12 -y
conda activate mlhw
pip install -r requirements.txt
```

### Data Preprocessing and $\alpha$ Factor Mining



### Model Building and Preprocessing 

#### Prepare Data

Ensure your data file is located at `data/data.csv` with the following columns:

- `timestamps`: Timestamp (date format)
- `open`: Opening price
- `high`: Highest price
- `low`: Lowest price
- `close`: Closing price
- `volume`: Trading volume
- `amount`: Trading amount

#### Model Training

##### Train XGBoost with Default Configuration

```bash
cd ml_model/model
python train_xgboost.py
```

##### Train with Custom Parameters

```bash
# Modify forecast horizon and lookback window
python train_xgboost.py horizon=3 lookback=20

# Modify model hyperparameters
python train_xgboost.py model.kwargs.max_depth=10 model.kwargs.n_estimators=500
```

#### Training Different Models

```bash
cd ml_model/model

# XGBoost
python train_xgboost.py

# LightGBM
python train_lightgbm.py

# Random Forest
python train_random_forest.py

# LSTM
python train_lstm.py

# GRU
python train_gru.py
```

#### Command-Line Parameter Overrides

All models support configuration parameter overrides via command line:

```bash
# Modify forecast horizon
python train_xgboost.py horizon=5

# Modify lookback window
python train_xgboost.py lookback=30

# Modify data path
python train_xgboost.py data.data_path=data/your_data.csv

# Modify training set time range
python train_xgboost.py data.train=[2018-01-02,2023-12-31]

# Combine multiple parameters
python train_xgboost.py horizon=3 lookback=20 model.kwargs.max_depth=10
```

#### Hyperparameter Search (Random Forest)

Random Forest model supports automatic hyperparameter search. Enable it in the configuration file:

```yaml
# config/random_forest.yaml
hyperparameter_search:
  enabled: true
  method: randomized
  n_iter: 50
```

The best parameters will be automatically searched during training and saved to `best_parameters.yaml` and `best_parameters.txt` in the output directory.

#### ⚙Configuration

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

#### Key Parameters

- **`horizon`**: Forecast window size, i.e., how many days ahead to predict
- **`lookback`**: Lookback window size, i.e., how many days of historical data to use as input features
- **`data.train/valid/test`**: Time series data split, ordered chronologically to avoid data leakage

#### 📊 Output

After training, all results are saved in `ml_model/output/<model_name>/horizon_<horizon>/` directory:

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

#### Model Saving

Trained models are saved in `ml_model/checkpoints/<model_name>/horizon_<horizon>/` directory:

- **XGBoost/LightGBM/Random Forest**: `*.pkl` (pickle format)
- **LSTM/GRU**: `*.pth` (PyTorch format) + `scaler_X.pkl`, `scaler_y.pkl` (scalers)

#### 📁 Project Structure

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

#### 🔍 FAQ

##### 1. XGBoost Error "OpenMP runtime is not installed" on Linux

```bash
# Ubuntu/Debian
sudo apt-get install libomp-dev
pip install --upgrade xgboost

# CentOS/RHEL
sudo yum install libgomp
pip install --upgrade xgboost
```

##### 2. Empty Test Set

Check if `fit_end_time` in the configuration file includes the test set time range. The system will automatically extend `fit_end_time` to the test set end time.

##### 3. How to Modify Data Path

```bash
# Method 1: Command-line override
python train_xgboost.py data.data_path=data/your_data.csv

# Method 2: Directly edit configuration file
# Edit data.data_path in config/xgboost.yaml
```

##### 4. How to View Training Progress

Training scripts output real-time training progress and evaluation metrics. For XGBoost/LightGBM, you can view the training process through loss curve plots.

#### Viewing Results

After training, you can view:

1. **Evaluation Metrics**: `output/<model>/horizon_<n>/metrics_log.txt`
2. **Prediction Charts**: `output/<model>/horizon_<n>/*_forecast_*.png`
3. **Loss Curves**: `output/<model>/horizon_<n>/loss_curve_rmse.png`
4. **Prediction Data**: `output/<model>/horizon_<n>/*_results.csv`
