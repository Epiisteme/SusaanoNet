import pandas as pd
import numpy as np

def load_and_align_data():
    print("Loading BTC 15m OHLCV...")
    # BTC uses 'Open time' as string
    btc = pd.read_csv('data/btc_15m_data_2018_to_2025.csv')
    btc['datetime'] = pd.to_datetime(btc['Open time'])
    btc.set_index('datetime', inplace=True)
    btc = btc[['Open', 'High', 'Low', 'Close', 'Volume']].rename(columns=lambda x: f"btc_{x.lower()}")
    
    print("Loading ETH 15m OHLCV...")
    # ETH uses 'datetime' as string
    eth = pd.read_csv('data/ETHUSDT_15m.csv')
    eth['datetime'] = pd.to_datetime(eth['datetime'])
    eth.set_index('datetime', inplace=True)
    eth = eth[['open', 'high', 'low', 'close', 'volume']].rename(columns=lambda x: f"eth_{x}")
    
    print("Loading SOL 15m OHLCV...")
    # SOL uses 'timestamp' in Unix ms
    sol = pd.read_csv('data/SOLUSDT_15minutes.csv')
    sol['datetime'] = pd.to_datetime(sol['timestamp'], unit='ms')
    sol.set_index('datetime', inplace=True)
    sol = sol[['open', 'high', 'low', 'close', 'volume']].rename(columns=lambda x: f"sol_{x}")
    
    print("Merging Multi-Asset Data...")
    # Inner join guarantees we only train on timestamps where all 3 assets existed (prevents massive NaNs)
    df = btc.join(eth, how='inner').join(sol, how='inner')
    
    print("Loading Funding Rates...")
    funding = pd.read_csv('data/funding_rate.csv')
    funding['datetime'] = pd.to_datetime(funding['timestamp'], unit='ms')
    funding.set_index('datetime', inplace=True)
    funding = funding[['funding_rate']].rename(columns={'funding_rate': 'btc_funding_rate'})
    
    # Left join so we keep our exact 15-minute grid, then forward fill the 8-hour blocks
    df = df.join(funding, how='left')
    df['btc_funding_rate'] = df['btc_funding_rate'].ffill().fillna(0.0) # Assume 0 if missing at start
    
    print("Loading Macro Data...")
    macro = pd.read_csv('data/Bitcoin Market Analysis Dataset (2021-2025).csv')
    # Macro uses 'date' string like '2021-01-01'
    macro['datetime'] = pd.to_datetime(macro['date'])
    macro.set_index('datetime', inplace=True)
    
    # Extract only the critical macro indicators
    macro = macro[['sp500_close', 'dxy_close', 'gold_close', 'hash_rate', 'fng_score']]
    
    df = df.join(macro, how='left')
    # Forward fill daily data across the 15-minute intervals
    df = df.ffill().bfill() # bfill just catches the first few hours if macro started slightly later
    
    return df

def feature_engineering(df):
    print("Engineering features per asset...")
    
    assets = ['btc', 'eth', 'sol']
    
    for asset in assets:
        close_col = f'{asset}_close'
        
        # Log Returns
        df[f'{asset}_log_return'] = np.log(df[close_col] / df[close_col].shift(1))
        
        # Volatility (1D and 7D)
        # 1 day = 96 periods of 15m
        df[f'{asset}_volatility_1d'] = df[f'{asset}_log_return'].rolling(window=96).std() * np.sqrt(96)
        df[f'{asset}_volatility_7d'] = df[f'{asset}_log_return'].rolling(window=96*7).std() * np.sqrt(96*7)
        
    df.dropna(inplace=True)
    return df

if __name__ == "__main__":
    print("=== HYBRID ARCHITECTURE DATA FUSION PIPELINE ===")
    df = load_and_align_data()
    df = feature_engineering(df)
    
    output_path = 'data/multi_asset_15m_structural.csv'
    print(f"Exporting master tensor to {output_path}...")
    # Keep datetime as column for downstream processes
    df.reset_index(inplace=True)
    df.to_csv(output_path, index=False)
    
    print(f"Done! Final Shape: {df.shape}")
    print("Columns generated:")
    print(list(df.columns))
