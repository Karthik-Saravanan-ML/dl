from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf

from data_pipeline import FEATURE_COLUMNS, prepare_market_data


ROOT = Path(__file__).resolve().parents[1]
ART_DIR = ROOT / "artifacts"


def predict_latest(ticker="TCS.NS", period="2y", lookback=30):
    required_artifacts = [
        ART_DIR / "feature_scaler.joblib",
        ART_DIR / "close_scaler.joblib",
        ART_DIR / "price_lstm.keras",
        ART_DIR / "fraud_lstm.keras",
    ]
    missing_artifacts = [path.name for path in required_artifacts if not path.exists()]
    if missing_artifacts:
        raise FileNotFoundError(
            "Trained model files are missing: "
            f"{', '.join(missing_artifacts)}. Run `python src/train.py` first."
        )

    df = prepare_market_data(ticker, period)

    feature_scaler = joblib.load(ART_DIR / "feature_scaler.joblib")
    close_scaler = joblib.load(ART_DIR / "close_scaler.joblib")
    price_model = tf.keras.models.load_model(ART_DIR / "price_lstm.keras")
    fraud_model = tf.keras.models.load_model(ART_DIR / "fraud_lstm.keras")

    if lookback < 1:
        raise ValueError("lookback must be at least 1.")
    if len(df) < lookback:
        raise ValueError(f"Need at least {lookback} rows for prediction.")

    recent = df[FEATURE_COLUMNS].tail(lookback)
    X = feature_scaler.transform(recent)
    X = np.expand_dims(X.astype(np.float32), axis=0)

    pred_close_scaled = price_model.predict(X, verbose=0)
    pred_close = float(close_scaler.inverse_transform(pred_close_scaled)[0, 0])

    fraud_probability = float(fraud_model.predict(X, verbose=0)[0, 0])
    latest_close = float(df["Close"].iloc[-1])

    return {
        "ticker": ticker,
        "last_date": str(df.index[-1].date()),
        "latest_close": latest_close,
        "predicted_next_close": pred_close,
        "suspicious_probability": fraud_probability,
        "suspicious_prediction": int(fraud_probability >= 0.50),
        "data": df,
    }
