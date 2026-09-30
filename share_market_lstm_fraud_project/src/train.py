import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler

from data_pipeline import FEATURE_COLUMNS, prepare_market_data


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ART_DIR = ROOT / "artifacts"
DATA_DIR.mkdir(exist_ok=True)
ART_DIR.mkdir(exist_ok=True)


def make_sequences(X_scaled, close_scaled, labels, lookback=30):
    X_seq, y_price, y_fraud, target_idx = [], [], [], []

    for i in range(lookback, len(X_scaled)):
        X_seq.append(X_scaled[i - lookback:i])
        y_price.append(close_scaled[i, 0])
        y_fraud.append(labels[i])
        target_idx.append(i)

    return (
        np.asarray(X_seq, dtype=np.float32),
        np.asarray(y_price, dtype=np.float32),
        np.asarray(y_fraud, dtype=np.float32),
        np.asarray(target_idx),
    )


def build_price_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"],
    )
    return model


def build_fraud_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def plot_history(history, metric, filename):
    plt.figure(figsize=(9, 5))
    plt.plot(history.history.get(metric, []), label=f"train_{metric}")
    plt.plot(history.history.get(f"val_{metric}", []), label=f"val_{metric}")
    plt.xlabel("Epoch")
    plt.ylabel(metric)
    plt.title(f"Training history: {metric}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ART_DIR / filename, dpi=160)
    plt.close()


def main(ticker="TCS.NS", period="5y", lookback=30, epochs=40):
    if lookback < 1:
        raise ValueError("lookback must be at least 1.")
    if epochs < 1:
        raise ValueError("epochs must be at least 1.")

    np.random.seed(42)
    tf.random.set_seed(42)

    print(f"Downloading {ticker} ...")
    df = prepare_market_data(ticker, period)
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    if train_end <= lookback or val_end <= train_end or n <= val_end:
        raise ValueError(
            f"Not enough prepared observations ({n}) for lookback={lookback} "
            "and chronological train/validation/test splits. Choose a longer "
            "history period or a smaller lookback."
        )

    df.to_csv(DATA_DIR / "market_data.csv")

    label_counts = df["Suspicious_Label"].value_counts().to_dict()
    print("Suspicious-label distribution:", label_counts)

    scaler = StandardScaler()
    scaler.fit(df[FEATURE_COLUMNS].iloc[:train_end])

    close_scaler = StandardScaler()
    close_scaler.fit(df[["Close"]].iloc[:train_end])

    X_all = scaler.transform(df[FEATURE_COLUMNS])
    close_all = close_scaler.transform(df[["Close"]])
    labels_all = df["Suspicious_Label"].to_numpy(dtype=np.float32)

    X_seq, y_price, y_fraud, target_idx = make_sequences(
        X_all, close_all, labels_all, lookback
    )

    train_mask = target_idx < train_end
    val_mask = (target_idx >= train_end) & (target_idx < val_end)
    test_mask = target_idx >= val_end

    X_train, X_val, X_test = X_seq[train_mask], X_seq[val_mask], X_seq[test_mask]
    yp_train, yp_val, yp_test = y_price[train_mask], y_price[val_mask], y_price[test_mask]
    yf_train, yf_val, yf_test = y_fraud[train_mask], y_fraud[val_mask], y_fraud[test_mask]

    print("Shapes:")
    print("  Train:", X_train.shape)
    print("  Val:  ", X_val.shape)
    print("  Test: ", X_test.shape)

    joblib.dump(scaler, ART_DIR / "feature_scaler.joblib")
    joblib.dump(close_scaler, ART_DIR / "close_scaler.joblib")

    callbacks_price = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=7, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            ART_DIR / "price_lstm.keras",
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    callbacks_fraud = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", mode="min", patience=7, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            ART_DIR / "fraud_lstm.keras",
            monitor="val_loss",
            mode="min",
            save_best_only=True,
        ),
    ]

    print("\nTraining price LSTM...")
    price_model = build_price_model((lookback, len(FEATURE_COLUMNS)))
    price_history = price_model.fit(
        X_train,
        yp_train,
        validation_data=(X_val, yp_val),
        epochs=epochs,
        batch_size=32,
        callbacks=callbacks_price,
        verbose=1,
    )
    plot_history(price_history, "loss", "price_training.png")

    print("\nTraining suspicious-activity LSTM...")
    fraud_model = build_fraud_model((lookback, len(FEATURE_COLUMNS)))

    positives = float(yf_train.sum())
    negatives = float(len(yf_train) - yf_train.sum())
    if positives > 0 and negatives > 0:
        total = positives + negatives
        class_weight = {
            0: total / (2.0 * negatives),
            1: total / (2.0 * positives),
        }
        print("Class weights:", class_weight)
    else:
        class_weight = None
        print("Only one class is present in training data; class_weight disabled.")

    fraud_history = fraud_model.fit(
        X_train,
        yf_train,
        validation_data=(X_val, yf_val),
        epochs=epochs,
        batch_size=32,
        class_weight=class_weight,
        callbacks=callbacks_fraud,
        verbose=1,
    )
    plot_history(fraud_history, "auc", "fraud_training.png")

    # Reload best checkpoints
    price_model = tf.keras.models.load_model(ART_DIR / "price_lstm.keras")
    fraud_model = tf.keras.models.load_model(ART_DIR / "fraud_lstm.keras")

    # Price metrics
    pred_price_scaled = price_model.predict(X_test, verbose=0).reshape(-1, 1)
    pred_price = close_scaler.inverse_transform(pred_price_scaled).ravel()
    true_price = close_scaler.inverse_transform(yp_test.reshape(-1, 1)).ravel()

    mae = float(mean_absolute_error(true_price, pred_price))
    rmse = float(np.sqrt(mean_squared_error(true_price, pred_price)))
    mape = float(np.mean(
        np.abs((true_price - pred_price) / np.maximum(np.abs(true_price), 1e-8))
    ) * 100)

    # Fraud metrics
    fraud_prob = fraud_model.predict(X_test, verbose=0).ravel()
    fraud_pred = (fraud_prob >= 0.50).astype(int)

    precision = float(precision_score(yf_test, fraud_pred, zero_division=0))
    recall = float(recall_score(yf_test, fraud_pred, zero_division=0))
    f1 = float(f1_score(yf_test, fraud_pred, zero_division=0))
    cm = confusion_matrix(yf_test, fraud_pred).tolist()

    if len(np.unique(yf_test)) == 2:
        auc = float(roc_auc_score(yf_test, fraud_prob))
    else:
        auc = None

    metrics = {
        "ticker": ticker,
        "period": period,
        "lookback_days": lookback,
        "price": {
            "MAE": mae,
            "RMSE": rmse,
            "MAPE_percent": mape,
        },
        "suspicious_activity": {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": auc,
            "confusion_matrix": cm,
        },
        "label_definition": "abs(Return_Z) > 2.5 AND Volume_Ratio > 1.8",
    }

    with open(ART_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\n==== FINAL TEST METRICS ====")
    print(json.dumps(metrics, indent=2))
    print("\nTraining complete. Run:")
    print("  streamlit run app.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="TCS.NS")
    parser.add_argument("--period", default="5y")
    parser.add_argument("--lookback", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=40)
    args = parser.parse_args()

    main(
        ticker=args.ticker,
        period=args.period,
        lookback=args.lookback,
        epochs=args.epochs,
    )
