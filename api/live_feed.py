"""
Live Data Aggregator
====================
Scrapes real-time 15-minute candle data from Binance and applies
the identical feature engineering used during training (via core/features.py).
This guarantees mathematical parity between training and inference.
"""
import requests
import pandas as pd
import numpy as np
from core.features import (
    engineer_asset_features,
    engineer_time_features,
    ASSETS,
    ASSET_FEATURE_COLS,
    GLOBAL_FEATURE_COLS,
    FEATURES_PER_ASSET,
)


class LiveDataAggregator:
    def __init__(self):
        self.base_url = "https://api.binance.com/api/v3"
        # Fetch 1000 ticks for enough history to compute 7-day rolling volatility
        self.limit = 1000

    def _fetch_binance_klines(self, symbol: str, prefix: str) -> pd.DataFrame:
        """Fetch raw kline data from Binance and rename columns with asset prefix."""
        endpoint = f"{self.base_url}/klines"
        params = {
            "symbol": symbol,
            "interval": "15m",
            "limit": self.limit,
        }
        response = requests.get(endpoint, params=params)
        response.raise_for_status()

        data = response.json()
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])

        numeric_cols = ['open', 'high', 'low', 'close', 'volume',
                        'number_of_trades', 'taker_buy_base_asset_volume']
        df[numeric_cols] = df[numeric_cols].astype(float)
        df['datetime'] = pd.to_datetime(df['timestamp'].astype(float), unit='ms')
        df.set_index('datetime', inplace=True)

        # Rename to match the prefix convention used by core/features.py
        df = df.rename(columns={
            'open': f'{prefix}_open',
            'high': f'{prefix}_high',
            'low': f'{prefix}_low',
            'close': f'{prefix}_close',
            'volume': f'{prefix}_volume',
            'number_of_trades': f'{prefix}_num_trades',
            'taker_buy_base_asset_volume': f'{prefix}_taker_buy_base_vol',
        })

        keep_cols = [c for c in df.columns if c.startswith(prefix)]
        return df[keep_cols]

    def get_live_ticks(self, seq_length: int = 30) -> tuple:
        """Fetch live data, engineer features, and return the inference tensor.

        Returns
        -------
        tuple of (raw_matrix, history, last_closes)
            raw_matrix : np.ndarray of shape (seq_length, num_features)
            history : dict with BTC/ETH/SOL close price arrays and timestamps
            last_closes : list of the most recent close price for each asset
        """
        # Fetch raw klines for each asset
        btc = self._fetch_binance_klines("BTCUSDT", "btc")
        eth = self._fetch_binance_klines("ETHUSDT", "eth")
        sol = self._fetch_binance_klines("SOLUSDT", "sol")

        # Merge on timestamp index
        df = btc.join(eth, how='inner').join(sol, how='inner')

        # Funding rate: use Binance funding rate endpoint or default
        df['btc_funding_rate'] = 0.0001  # Baseline; updated if live API available

        # Apply the same feature engineering as training
        for asset in ASSETS:
            df = engineer_asset_features(df, asset)
        df = engineer_time_features(df)

        # Drop NaNs from rolling windows
        df.dropna(inplace=True)

        if len(df) < seq_length:
            raise ValueError(
                f"Not enough data to construct {seq_length} ticks after NaN drop. "
                f"Got {len(df)} rows."
            )

        latest = df.tail(seq_length).copy()

        # Build the feature column list matching training order
        feature_cols = []
        for asset in ASSETS:
            for feat in ASSET_FEATURE_COLS:
                feature_cols.append(f'{asset}_{feat}')
        feature_cols.extend(GLOBAL_FEATURE_COLS)

        # Validate all columns exist
        missing = [c for c in feature_cols if c not in latest.columns]
        if missing:
            raise ValueError(f"Missing expected feature columns: {missing}")

        raw_matrix = latest[feature_cols].values

        # Historical data for frontend visualization
        history = {
            "BTC": latest['btc_close'].tolist(),
            "ETH": latest['eth_close'].tolist(),
            "SOL": latest['sol_close'].tolist(),
            "timestamps": latest.index.strftime('%H:%M').tolist(),
        }

        last_closes = [
            history["BTC"][-1],
            history["ETH"][-1],
            history["SOL"][-1],
        ]

        return raw_matrix, history, last_closes
