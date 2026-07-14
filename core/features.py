"""
Unified Feature Engineering Module
===================================
Single source of truth for all feature transformations.
Used by both data_pipeline.py (training) and api/live_feed.py (production).
This guarantees mathematical parity between training and inference.
"""
import numpy as np
import pandas as pd


# Default trading cost threshold for cost-aware 3-class targets.
# Binance spot taker fee is 0.1%, so a round-trip is ~0.2%.
# We use 0.1% as a conservative single-leg threshold.
DEFAULT_COST_THRESHOLD = 0.001

# Feature ordering constants
ASSETS = ['btc', 'eth', 'sol']

# Per-asset features produced by engineer_asset_features()
ASSET_FEATURE_COLS = [
    'open', 'high', 'low', 'close', 'volume',
    'taker_buy_ratio', 'num_trades', 'avg_trade_size',
    'log_return',
    'volatility_1d', 'volatility_7d',
    'rsi', 'macd', 'bb_width',
]

# Global features appended to every asset node
GLOBAL_FEATURE_COLS = [
    'btc_funding_rate',
    'hour_sin', 'hour_cos',
    'dow_sin', 'dow_cos',
]

# Total features per asset node in the tensor
FEATURES_PER_ASSET = len(ASSET_FEATURE_COLS) + len(GLOBAL_FEATURE_COLS)


def engineer_asset_features(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Compute all per-asset features from raw OHLCV + microstructure columns.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: '{prefix}_open', '{prefix}_high', '{prefix}_low',
        '{prefix}_close', '{prefix}_volume'. Optionally contains
        '{prefix}_taker_buy_base_vol' and '{prefix}_num_trades'.
    prefix : str
        Asset prefix, e.g. 'btc', 'eth', 'sol'.

    Returns
    -------
    pd.DataFrame
        DataFrame with all computed feature columns prefixed by `prefix`.
    """
    close = df[f'{prefix}_close']
    volume = df[f'{prefix}_volume']

    # --- Microstructure features ---
    taker_col = f'{prefix}_taker_buy_base_vol'
    trades_col = f'{prefix}_num_trades'

    if taker_col in df.columns:
        df[f'{prefix}_taker_buy_ratio'] = df[taker_col] / volume.replace(0, np.nan)
        df[f'{prefix}_taker_buy_ratio'] = df[f'{prefix}_taker_buy_ratio'].fillna(0.5)
    else:
        df[f'{prefix}_taker_buy_ratio'] = 0.5

    if trades_col in df.columns:
        df[f'{prefix}_num_trades'] = np.log1p(df[trades_col])
        df[f'{prefix}_avg_trade_size'] = np.log1p(volume / df[trades_col].replace(0, 1))
    else:
        df[f'{prefix}_num_trades'] = 0.0
        df[f'{prefix}_avg_trade_size'] = 0.0

    # --- Log Returns ---
    df[f'{prefix}_log_return'] = np.log(close / close.shift(1))

    # --- Volatility (rolling std of log returns, NO sqrt multiplier) ---
    log_ret = df[f'{prefix}_log_return']
    df[f'{prefix}_volatility_1d'] = log_ret.rolling(window=96).std()
    df[f'{prefix}_volatility_7d'] = log_ret.rolling(window=96 * 7).std()

    # --- RSI (14 period) ---
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df[f'{prefix}_rsi'] = 100 - (100 / (1 + rs))

    # --- MACD (12, 26) ---
    exp12 = close.ewm(span=12, adjust=False).mean()
    exp26 = close.ewm(span=26, adjust=False).mean()
    df[f'{prefix}_macd'] = exp12 - exp26

    # --- Bollinger Band Width (20 period, 2 std dev) ---
    sma = close.rolling(window=20).mean()
    std = close.rolling(window=20).std()
    df[f'{prefix}_bb_width'] = ((sma + std * 2) - (sma - std * 2)) / sma

    return df


def engineer_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical time-of-day and day-of-week encodings.

    Parameters
    ----------
    df : pd.DataFrame
        Must have a DatetimeIndex.

    Returns
    -------
    pd.DataFrame
        DataFrame with hour_sin, hour_cos, dow_sin, dow_cos columns added.
    """
    hour = df.index.hour + df.index.minute / 60.0
    dow = df.index.dayofweek

    df['hour_sin'] = np.sin(2 * np.pi * hour / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24.0)
    df['dow_sin'] = np.sin(2 * np.pi * dow / 7.0)
    df['dow_cos'] = np.cos(2 * np.pi * dow / 7.0)

    return df


def generate_direction_targets(
    df: pd.DataFrame,
    horizons: dict = None,
    cost_threshold: float = DEFAULT_COST_THRESHOLD,
) -> pd.DataFrame:
    """Generate cost-aware 3-class directional targets for each asset and horizon.

    Classes:
        0 = Down  (return < -cost_threshold)
        1 = Neutral (abs(return) <= cost_threshold)
        2 = Up    (return > cost_threshold)

    Also generates raw continuous log-return targets for quantile regression.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain '{asset}_log_return' columns.
    horizons : dict
        Mapping of horizon name to number of bars, e.g. {'1h': 4, '4h': 16}.
        Defaults to {'15m': 1, '1h': 4, '4h': 16}.
    cost_threshold : float
        Minimum return magnitude to classify as Up or Down.

    Returns
    -------
    pd.DataFrame
        DataFrame with target columns added.
    """
    if horizons is None:
        horizons = {'15m': 1, '1h': 4, '4h': 16}

    for asset in ASSETS:
        close = df[f'{asset}_close']
        for horizon_name, bars in horizons.items():
            # Continuous target: future log return over `bars` periods
            future_return = np.log(close.shift(-bars) / close)
            df[f'{asset}_target_return_{horizon_name}'] = future_return

            # 3-class direction target
            direction = np.where(
                future_return > cost_threshold, 2,
                np.where(future_return < -cost_threshold, 0, 1)
            )
            df[f'{asset}_target_dir_{horizon_name}'] = direction

    return df


def get_feature_columns() -> list:
    """Return the ordered list of feature columns per asset (excluding targets).

    This defines the exact column ordering used to construct the
    (Time, Assets, Features) tensor during training and inference.
    """
    cols = []
    for asset in ASSETS:
        for feat in ASSET_FEATURE_COLS:
            cols.append(f'{asset}_{feat}')
    # Global features are added once, then broadcast to all asset nodes
    cols.extend(GLOBAL_FEATURE_COLS)
    return cols


def get_target_columns(horizons: dict = None) -> dict:
    """Return target column names grouped by type.

    Returns
    -------
    dict with keys 'regression' and 'classification', each containing
    a list of column names.
    """
    if horizons is None:
        horizons = {'15m': 1, '1h': 4, '4h': 16}

    regression_cols = []
    classification_cols = []

    for asset in ASSETS:
        for horizon_name in horizons.keys():
            regression_cols.append(f'{asset}_target_return_{horizon_name}')
            classification_cols.append(f'{asset}_target_dir_{horizon_name}')

    return {
        'regression': regression_cols,
        'classification': classification_cols,
    }
