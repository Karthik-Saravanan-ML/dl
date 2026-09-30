import argparse
from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "Open", "High", "Low", "Close", "Volume",
    "Return_1D", "Volatility_20", "Volume_Ratio",
    "Return_Z", "Price_Z", "RSI_14", "MACD", "MACD_Signal"
]


def download_market_data(ticker: str = "TCS.NS", period: str = "5y") -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError(
            "yfinance is required to download market data. "
            "Install project dependencies with `pip install -r requirements.txt`."
        ) from exc

    df = yf.download(
        ticker,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False,
        multi_level_index=False,
    )

    if df is None or df.empty:
        raise ValueError(f"No market data returned for ticker: {ticker}")

    # yfinance can return a slightly different column layout depending on version.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns from downloaded data: {missing}")

    df = df[required].copy()
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    df = df.sort_index()
    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required market columns: {missing}")
    if df.empty:
        raise ValueError("Market data is empty.")

    out = df.copy()
    out[required] = out[required].apply(pd.to_numeric, errors="coerce")

    out["Return_1D"] = out["Close"].pct_change()

    out["SMA_20"] = out["Close"].rolling(20).mean()
    out["STD_20"] = out["Close"].rolling(20).std()
    out["Volatility_20"] = out["Return_1D"].rolling(20).std()

    volume_ma20 = out["Volume"].rolling(20).mean()
    out["Volume_Ratio"] = out["Volume"] / volume_ma20

    return_mean20 = out["Return_1D"].rolling(20).mean()
    return_std20 = out["Return_1D"].rolling(20).std()
    out["Return_Z"] = (out["Return_1D"] - return_mean20) / return_std20
    out.loc[return_std20 == 0, "Return_Z"] = 0

    out["Price_Z"] = (out["Close"] - out["SMA_20"]) / out["STD_20"]

    delta = out["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    out["RSI_14"] = 100 - (100 / (1 + rs))
    out.loc[(loss == 0) & (gain > 0), "RSI_14"] = 100
    out.loc[(loss == 0) & (gain == 0), "RSI_14"] = 50

    ema12 = out["Close"].ewm(span=12, adjust=False).mean()
    ema26 = out["Close"].ewm(span=26, adjust=False).mean()
    out["MACD"] = ema12 - ema26
    out["MACD_Signal"] = out["MACD"].ewm(span=9, adjust=False).mean()

    # Educational proxy for suspicious activity:
    # unusually large recent return + unusually high volume.
    out["Suspicious_Label"] = (
        (out["Return_Z"].abs() > 2.5) &
        (out["Volume_Ratio"] > 1.8)
    ).astype(int)

    zero_price_std = out["STD_20"] == 0
    out.loc[zero_price_std, "Price_Z"] = 0
    out = out.replace([np.inf, -np.inf], np.nan).dropna().copy()
    return out


def prepare_market_data(ticker: str = "TCS.NS", period: str = "5y") -> pd.DataFrame:
    raw = download_market_data(ticker, period)
    return add_indicators(raw)


def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare historical share-market data."
    )
    parser.add_argument("--ticker", default="TCS.NS")
    parser.add_argument("--period", default="5y")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "market_data.csv",
        help="CSV destination (default: data/market_data.csv)",
    )
    args = parser.parse_args()

    df = prepare_market_data(args.ticker, args.period)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index_label="Date")
    print(f"Saved {len(df)} prepared rows to {args.output}")


if __name__ == "__main__":
    main()
