# Share Market Data Analysis and Suspicious Activity Prediction Using LSTM

## Project idea

This mini project combines:

1. Historical share-market analysis
2. LSTM-based next-day closing-price prediction
3. LSTM-based suspicious-market-activity classification
4. A Streamlit dashboard

### Important modeling note

Raw OHLCV stock data does **not** contain ground-truth labels proving that a trading day was fraudulent. Therefore this educational version creates a transparent **suspicious-activity label** from unusual return and volume behavior.

This should be described in your report as a **prototype / anomaly-detection demonstration**, not as a system that proves legal fraud.

To use a real fraud model, replace `Suspicious_Label` with independently verified labeled surveillance data.

## Recommended environment

Python 3.11 or 3.12 is a practical choice for this project.

On Windows, current TensorFlow documentation states that native-Windows GPU support is limited to TensorFlow 2.10 and earlier; newer TensorFlow GPU setups should use WSL2. CPU execution on Windows is also supported. See the official TensorFlow installation guide.

## Folder structure

```text
share_market_lstm_fraud_project/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── README.md
│   └── market_data.csv              # downloaded/prepared locally
│
├── artifacts/
│   ├── price_lstm.keras              # created during training
│   ├── fraud_lstm.keras
│   ├── feature_scaler.joblib
│   ├── close_scaler.joblib
│   ├── metrics.json
│   ├── price_training.png
│   └── fraud_training.png
│
├── src/
    ├── data_pipeline.py
    ├── train.py
    └── predict.py
└── tests/
    └── test_data_pipeline.py
```

## 1. Create environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Windows CMD:

```cmd
.venv\Scripts\activate
```

## 2. Install libraries

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Download the dataset

The `data/` folder starts empty because market data is downloaded for the ticker
you choose; the project does not include fabricated or mislabeled market data.
Install the requirements first, then run this from the project root to create
the prepared CSV without training a model:

```bash
python src/data_pipeline.py --ticker TCS.NS --period 5y
```

The default output is `data/market_data.csv`. You can choose another ticker,
history period, or output path:

```bash
python src/data_pipeline.py --ticker RELIANCE.NS --period 5y --output data/reliance.csv
```

The CSV contains OHLCV prices, engineered technical features, and the
documented `Suspicious_Label` heuristic. It is generated locally and ignored by
Git.

## 4. Train the models

Default ticker: `TCS.NS`

```bash
python src/train.py
```

You can select another Yahoo Finance ticker:

```bash
python src/train.py --ticker RELIANCE.NS --period 5y
```

The training script:

- downloads historical OHLCV data
- creates technical features
- creates the transparent suspicious-activity label
- performs a chronological train/validation/test split
- scales only from the training period
- creates 30-day sequences
- trains the price LSTM
- trains the suspicious-activity LSTM
- evaluates both models
- saves models, scalers, plots, metrics, and the processed CSV

## 5. Run the dashboard

```bash
streamlit run app.py
```

Then open the localhost URL shown by Streamlit.

## 6. What the features mean

- Open, High, Low, Close, Volume: market data
- Return_1D: daily percentage return
- Volatility_20: 20-day rolling volatility
- Volume_Ratio: volume divided by 20-day average volume
- Return_Z: unusualness of today's return compared with its recent history
- Price_Z: distance of price from its 20-day moving average measured in rolling standard deviations
- RSI_14: momentum indicator
- MACD, MACD_Signal: trend/momentum indicators

## 7. Suspicious activity label used in this prototype

A day is marked suspicious when both conditions are met:

```text
abs(Return_Z) > 2.5
AND
Volume_Ratio > 1.8
```

This is a **heuristic**, not a legal or regulatory definition of fraud.

## 8. Model design

### Price LSTM

```text
30 days × 13 features
        ↓
LSTM(64)
        ↓
Dropout
        ↓
LSTM(32)
        ↓
Dense(16)
        ↓
Next-day Close
```

### Suspicious-activity LSTM

```text
30 days × 13 features
        ↓
LSTM(64)
        ↓
Dropout
        ↓
LSTM(32)
        ↓
Dense(16)
        ↓
Sigmoid
        ↓
Suspicious probability
```

## 9. Evaluation

### Price model

- MAE
- RMSE
- MAPE

### Suspicious-activity model

- Precision
- Recall
- F1-score
- ROC-AUC when both classes are present
- Confusion matrix

For a suspicious-event detector, recall is useful because missing rare events can matter more than raw accuracy.

## 10. Suggested report title

**Share Market Data Analysis and Suspicious Activity Prediction Using LSTM Networks**

If your college requires the exact original title, use:

**Share Market Data Analysis and Fraud Prediction Using LSTM Networks**

and clearly state in the limitations section that the fraud label in this prototype is a heuristic proxy.

## 11. Future enhancement

Replace heuristic labels with verified market-surveillance cases and add trader/order-level fields such as order frequency, cancellation rate, trade size, price impact, and account relationships. This would make the fraud-detection problem substantially closer to a real surveillance system.
