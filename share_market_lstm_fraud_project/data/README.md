# Market data

This folder intentionally contains no bundled dataset. Download the ticker and
history you need from Yahoo Finance after installing `requirements.txt`:

```bash
python src/data_pipeline.py --ticker TCS.NS --period 5y
```

This creates `market_data.csv` with OHLCV data, engineered features, and the
prototype suspicious-activity label. Training creates the same file
automatically. The CSV is excluded from Git by the root `.gitignore`.

The label is only a heuristic based on unusual return and volume behavior; it
is not verified fraud data.
