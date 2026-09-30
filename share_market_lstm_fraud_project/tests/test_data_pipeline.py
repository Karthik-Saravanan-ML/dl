import unittest

import numpy as np
import pandas as pd

from src.data_pipeline import FEATURE_COLUMNS, add_indicators


class AddIndicatorsTests(unittest.TestCase):
    def test_flat_prices_produce_finite_features_and_neutral_rsi(self):
        dates = pd.date_range("2024-01-01", periods=60, freq="B")
        prices = np.full(len(dates), 100.0)
        market_data = pd.DataFrame(
            {
                "Open": prices,
                "High": prices,
                "Low": prices,
                "Close": prices,
                "Volume": np.full(len(dates), 1000.0),
            },
            index=dates,
        )

        prepared = add_indicators(market_data)

        self.assertFalse(prepared.empty)
        self.assertTrue(np.isfinite(prepared[FEATURE_COLUMNS].to_numpy()).all())
        self.assertTrue((prepared["RSI_14"] == 50).all())
        self.assertTrue((prepared["Price_Z"] == 0).all())
        self.assertTrue((prepared["Suspicious_Label"] == 0).all())

    def test_missing_market_columns_are_reported(self):
        market_data = pd.DataFrame({"Close": [1.0]})

        with self.assertRaisesRegex(ValueError, "Missing required market columns"):
            add_indicators(market_data)

    def test_large_return_and_volume_spike_is_labeled(self):
        dates = pd.date_range("2024-01-01", periods=80, freq="B")
        prices = np.full(len(dates), 100.0)
        volumes = np.full(len(dates), 1000.0)
        prices[40:] = 130.0
        volumes[40] = 3000.0
        market_data = pd.DataFrame(
            {
                "Open": prices,
                "High": prices,
                "Low": prices,
                "Close": prices,
                "Volume": volumes,
            },
            index=dates,
        )

        prepared = add_indicators(market_data)

        self.assertEqual(prepared.loc[dates[40], "Suspicious_Label"], 1)


if __name__ == "__main__":
    unittest.main()
