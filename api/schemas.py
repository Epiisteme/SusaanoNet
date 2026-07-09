from pydantic import BaseModel, Field
from typing import List, Dict

class MultiAssetTick(BaseModel):
    """Represents a single 15-minute timeframe containing all 3 assets + derivatives + macro data."""
    btc_open: float
    btc_high: float
    btc_low: float
    btc_close: float
    btc_volume: float
    eth_open: float
    eth_high: float
    eth_low: float
    eth_close: float
    eth_volume: float
    sol_open: float
    sol_high: float
    sol_low: float
    sol_close: float
    sol_volume: float
    btc_funding_rate: float
    sp500_close: float
    dxy_close: float
    gold_close: float
    hash_rate: float
    fng_score: float
    btc_log_return: float
    btc_volatility_1d: float
    btc_volatility_7d: float
    eth_log_return: float
    eth_volatility_1d: float
    eth_volatility_7d: float
    sol_log_return: float
    sol_volatility_1d: float
    sol_volatility_7d: float

class MultiAssetForecastRequest(BaseModel):
    """The incoming payload required to make a T+1 Multi-Asset Quantile prediction."""
    sequence: List[MultiAssetTick] = Field(..., min_items=60, max_items=60)
    last_btc_close: float
    last_eth_close: float
    last_sol_close: float

class QuantilePrediction(BaseModel):
    p10_crash_boundary: float
    p50_median_forecast: float
    p90_breakout_boundary: float

class MultiAssetForecastResponse(BaseModel):
    """The outgoing T+1 absolute price risk boundaries for all assets."""
    BTC: QuantilePrediction
    ETH: QuantilePrediction
    SOL: QuantilePrediction