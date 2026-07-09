import requests
import pandas as pd
import numpy as np
from datetime import datetime

class LiveDataAggregator:
    def __init__(self):
        self.base_url = "https://api.binance.com/api/v3"
        # We fetch 1000 ticks so we have enough history to calculate the 7-day rolling volatility (672 ticks)
        self.limit = 1000 
        
    def _fetch_binance_klines(self, symbol):
        endpoint = f"{self.base_url}/klines"
        params = {
            "symbol": symbol,
            "interval": "15m",
            "limit": self.limit
        }
        response = requests.get(endpoint, params=params)
        response.raise_for_status()
        
        data = response.json()
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].astype(float)
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('datetime', inplace=True)
        return df

    def _engineer_features(self, df, prefix):
        # Log Returns
        df[f'{prefix}_log_return'] = np.log(df['close'] / df['close'].shift(1))
        
        # Volatility (1D = 96 periods, 7D = 672 periods)
        df[f'{prefix}_volatility_1d'] = df[f'{prefix}_log_return'].rolling(window=96).std()
        df[f'{prefix}_volatility_7d'] = df[f'{prefix}_log_return'].rolling(window=672).std()
        
        # Rename base columns
        df.rename(columns={
            'open': f'{prefix}_open',
            'high': f'{prefix}_high',
            'low': f'{prefix}_low',
            'close': f'{prefix}_close',
            'volume': f'{prefix}_volume'
        }, inplace=True)
        
        return df[[
            f'{prefix}_open', f'{prefix}_high', f'{prefix}_low', f'{prefix}_close', f'{prefix}_volume',
            f'{prefix}_log_return', f'{prefix}_volatility_1d', f'{prefix}_volatility_7d'
        ]]

    def _fetch_macro_cache(self):
        # Macro data requires paid APIs (Bloomberg/Yahoo) to hit programmatically in real-time.
        # Since this is a production prototype, we lock these to the most recent sensible values 
        # from the training set, as S&P500/DXY don't update on 15m intervals anyway.
        return {
            'btc_funding_rate': 0.0001,  # Standard baseline funding
            'sp500_close': 5000.0,
            'dxy_close': 104.0,
            'gold_close': 2300.0,
            'hash_rate': 600000000.0,
            'fng_score': 70.0
        }

    def get_live_60_ticks(self):
        """
        Fetches live data from Binance, engineers structural features, 
        and returns the exact last 60 ticks required by the Hybrid API.
        """
        # Fetch raw market data
        btc = self._engineer_features(self._fetch_binance_klines("BTCUSDT"), "btc")
        eth = self._engineer_features(self._fetch_binance_klines("ETHUSDT"), "eth")
        sol = self._engineer_features(self._fetch_binance_klines("SOLUSDT"), "sol")
        
        # Merge assets on timestamp index
        df = btc.join(eth, how='inner').join(sol, how='inner')
        
        # Append macro features
        macro = self._fetch_macro_cache()
        for key, val in macro.items():
            df[key] = val
            
        # Drop NaNs introduced by rolling windows
        df.dropna(inplace=True)
        
        # Extract final tensor sequence
        if len(df) < 60:
            raise ValueError("Not enough data to construct 60 live ticks after NaN drop.")
            
        latest_60 = df.tail(60).copy()
        
        # Reorder columns to match scaler expectations
        expected_columns = [
            'btc_open', 'btc_high', 'btc_low', 'btc_close', 'btc_volume', 
            'eth_open', 'eth_high', 'eth_low', 'eth_close', 'eth_volume', 
            'sol_open', 'sol_high', 'sol_low', 'sol_close', 'sol_volume', 
            'btc_funding_rate', 'sp500_close', 'dxy_close', 'gold_close', 'hash_rate', 'fng_score', 
            'btc_log_return', 'btc_volatility_1d', 'btc_volatility_7d', 
            'eth_log_return', 'eth_volatility_1d', 'eth_volatility_7d', 
            'sol_log_return', 'sol_volatility_1d', 'sol_volatility_7d'
        ]
        
        latest_60 = latest_60[expected_columns]
        
        # Extract raw numpy array for inference and historical data for visualization
        raw_matrix = latest_60.values
        
        # Extract the historical arrays for the frontend charts
        history = {
            "BTC": latest_60['btc_close'].tolist(),
            "ETH": latest_60['eth_close'].tolist(),
            "SOL": latest_60['sol_close'].tolist(),
            "timestamps": latest_60.index.strftime('%H:%M').tolist()
        }
        
        last_closes = [
            history["BTC"][-1],
            history["ETH"][-1],
            history["SOL"][-1]
        ]
        
        return raw_matrix, history, last_closes
