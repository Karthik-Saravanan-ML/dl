import json
from pathlib import Path
import sys

import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "src"))

from predict import predict_latest


ART_DIR = ROOT / "artifacts"

st.set_page_config(
    page_title="Share Market LSTM Analyzer",
    page_icon="📈",
    layout="wide",
)

st.title("Share Market Data Analysis & LSTM Prediction")
st.caption(
    "Educational prototype: price forecasting + suspicious market-activity detection"
)

with st.sidebar:
    st.header("Settings")
    ticker = st.text_input("Yahoo Finance ticker", value="TCS.NS")
    period = st.selectbox("History", ["1y", "2y", "5y", "10y"], index=1)
    run = st.button("Run analysis", type="primary")

if not (ART_DIR / "price_lstm.keras").exists():
    st.warning(
        "Models are not trained yet. Run `python src/train.py` first."
    )
    st.stop()

if not run:
    st.info("Choose a ticker and click **Run analysis** to load the market data.")
    st.stop()

st.cache_data.clear()

@st.cache_data(show_spinner=True)
def get_prediction(ticker_value, period_value):
    result = predict_latest(ticker_value, period_value)
    # DataFrame is okay to cache because it is returned with the result.
    return result


try:
    result = get_prediction(ticker, period)
except Exception as exc:
    st.error(f"Prediction failed: {exc}")
    st.stop()

df = result["data"]

m1, m2, m3, m4 = st.columns(4)
m1.metric("Latest Close", f"{result['latest_close']:.2f}")
m2.metric("Predicted Next Close", f"{result['predicted_next_close']:.2f}")
m3.metric(
    "Suspicious Probability",
    f"{result['suspicious_probability'] * 100:.1f}%"
)
m4.metric("Last Market Date", result["last_date"])

st.subheader("Price trend")

fig = go.Figure()
fig.add_trace(
    go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="OHLC",
    )
)
fig.update_layout(
    height=520,
    xaxis_title="Date",
    yaxis_title="Price",
    xaxis_rangeslider_visible=False,
)
st.plotly_chart(fig, use_container_width=True)

c1, c2 = st.columns(2)

with c1:
    st.subheader("Volume ratio")
    volume_fig = go.Figure()
    volume_fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Volume_Ratio"],
            mode="lines",
            name="Volume Ratio",
        )
    )
    volume_fig.add_hline(y=1.8, line_dash="dash")
    volume_fig.update_layout(height=360)
    st.plotly_chart(volume_fig, use_container_width=True)

with c2:
    st.subheader("RSI")
    rsi_fig = go.Figure()
    rsi_fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["RSI_14"],
            mode="lines",
            name="RSI",
        )
    )
    rsi_fig.add_hline(y=70, line_dash="dash")
    rsi_fig.add_hline(y=30, line_dash="dash")
    rsi_fig.update_layout(height=360)
    st.plotly_chart(rsi_fig, use_container_width=True)

st.subheader("Model interpretation")

if result["suspicious_probability"] >= 0.50:
    st.error(
        "The model predicts elevated suspicious-activity probability for the next step."
    )
else:
    st.success(
        "The model predicts lower suspicious-activity probability for the next step."
    )

st.info(
    "This probability is a machine-learning output from an educational proxy label; "
    "it is not proof that market fraud occurred."
)

if (ART_DIR / "metrics.json").exists():
    with open(ART_DIR / "metrics.json", "r", encoding="utf-8") as f:
        metrics = json.load(f)

    st.subheader("Test-set metrics")
    pc1, pc2, pc3 = st.columns(3)
    pc1.metric("Price MAE", f"{metrics['price']['MAE']:.4f}")
    pc2.metric("Price RMSE", f"{metrics['price']['RMSE']:.4f}")
    pc3.metric("Price MAPE", f"{metrics['price']['MAPE_percent']:.2f}%")

    fc1, fc2, fc3, fc4 = st.columns(4)
    fc1.metric("Precision", f"{metrics['suspicious_activity']['precision']:.3f}")
    fc2.metric("Recall", f"{metrics['suspicious_activity']['recall']:.3f}")
    fc3.metric("F1", f"{metrics['suspicious_activity']['f1']:.3f}")
    auc = metrics["suspicious_activity"]["roc_auc"]
    fc4.metric("ROC-AUC", "N/A" if auc is None else f"{auc:.3f}")

with st.expander("Latest engineered features"):
    st.dataframe(
        df[
            [
                "Close", "Volume", "Return_1D", "Volatility_20",
                "Volume_Ratio", "Return_Z", "Price_Z",
                "RSI_14", "MACD", "MACD_Signal", "Suspicious_Label"
            ]
        ].tail(15),
        use_container_width=True,
    )
