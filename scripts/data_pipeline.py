"""
Training Data Pipeline
======================
Loads raw CSVs, applies unified feature engineering, generates targets,
and exports the final multi-asset structural dataset for model training.
"""
import pandas as pd
import numpy as np
from core.features import (
    engineer_asset_features,
    engineer_time_features,
    generate_direction_targets,
    ASSETS,
)


def load_and_align_data():
    """Load and temporally align all raw data sources."""
    print("Loading BTC 15m OHLCV...")
    btc = pd.read_csv('data/btc_15m_data_2018_to_2025.csv')
    btc['datetime'] = pd.to_datetime(btc['Open time'])
    btc.set_index('datetime', inplace=True)
    btc = btc[['Open', 'High', 'Low', 'Close', 'Volume',
               'Number of trades', 'Taker buy base asset volume']].rename(columns={
        'Open': 'btc_open', 'High': 'btc_high', 'Low': 'btc_low',
        'Close': 'btc_close', 'Volume': 'btc_volume',
        'Number of trades': 'btc_num_trades',
        'Taker buy base asset volume': 'btc_taker_buy_base_vol',
    })

    print("Loading ETH 15m OHLCV...")
    eth = pd.read_csv('data/ETHUSDT_15m.csv')
    eth['datetime'] = pd.to_datetime(eth['datetime'])
    eth.set_index('datetime', inplace=True)
    # ETH CSV lacks microstructure columns
    eth = eth[['open', 'high', 'low', 'close', 'volume']].rename(
        columns=lambda x: f"eth_{x}"
    )

    print("Loading SOL 15m OHLCV...")
    sol = pd.read_csv('data/SOLUSDT_15minutes.csv')
    sol['datetime'] = pd.to_datetime(sol['timestamp'], unit='ms')
    sol.set_index('datetime', inplace=True)
    sol = sol[['open', 'high', 'low', 'close', 'volume',
               'number_of_trades', 'taker_buy_base_asset_volume']].rename(columns={
        'open': 'sol_open', 'high': 'sol_high', 'low': 'sol_low',
        'close': 'sol_close', 'volume': 'sol_volume',
        'number_of_trades': 'sol_num_trades',
        'taker_buy_base_asset_volume': 'sol_taker_buy_base_vol',
    })

    print("Merging Multi-Asset Data...")
    df = btc.join(eth, how='inner').join(sol, how='inner')

    print("Loading Funding Rates...")
    funding = pd.read_csv('data/funding_rate.csv')
    funding['datetime'] = pd.to_datetime(funding['timestamp'], unit='ms')
    funding.set_index('datetime', inplace=True)
    funding = funding[['funding_rate']].rename(columns={'funding_rate': 'btc_funding_rate'})

    df = df.join(funding, how='left')
    df['btc_funding_rate'] = df['btc_funding_rate'].ffill().fillna(0.0)

    # NOTE: Macro data removed per guide's recommendation (Point 2 & 7).
    # Slow daily macro variables (S&P500, DXY, Gold) are unreliable for
    # 15-minute forecasts and were previously hardcoded in production.
    # Replacing with high-frequency microstructure features instead.

    return df


def feature_engineering(df):
    """Apply unified feature engineering to all assets."""
    print("Engineering features per asset (via core/features.py)...")

    for asset in ASSETS:
        df = engineer_asset_features(df, asset)

    # Time encodings
    df = engineer_time_features(df)

    # Generate multi-horizon cost-aware targets
    print("Generating multi-horizon cost-aware targets...")
    df = generate_direction_targets(df)

    # Forward fill only, no backward fill (Point 2: prevents future info leak)
    df = df.ffill()
    df.dropna(inplace=True)

    return df


if __name__ == "__main__":
    print("=== HYBRID ARCHITECTURE DATA FUSION PIPELINE (v2) ===")
    df = load_and_align_data()
    df = feature_engineering(df)

    output_path = 'data/multi_asset_15m_structural_v2.csv'
    print(f"Exporting master tensor to {output_path}...")
    df.reset_index(inplace=True)
    df.to_csv(output_path, index=False)

    print(f"Done! Final Shape: {df.shape}")
    print("Columns generated:")
    print(list(df.columns))
