"""
API Schemas
===========
Pydantic models for request/response validation.
Feature ordering matches core/features.py definitions exactly.
"""
from pydantic import BaseModel, Field
from typing import List, Dict


class MultiAssetTick(BaseModel):
    """A single 15-minute timeframe containing all 3 assets + structural features."""
    # BTC OHLCV
    btc_open: float
    btc_high: float
    btc_low: float
    btc_close: float
    btc_volume: float
    btc_taker_buy_ratio: float
    btc_num_trades: float
    btc_avg_trade_size: float
    btc_log_return: float
    btc_volatility_1d: float
    btc_volatility_7d: float
    btc_rsi: float
    btc_macd: float
    btc_bb_width: float
    # ETH OHLCV
    eth_open: float
    eth_high: float
    eth_low: float
    eth_close: float
    eth_volume: float
    eth_taker_buy_ratio: float
    eth_num_trades: float
    eth_avg_trade_size: float
    eth_log_return: float
    eth_volatility_1d: float
    eth_volatility_7d: float
    eth_rsi: float
    eth_macd: float
    eth_bb_width: float
    # SOL OHLCV
    sol_open: float
    sol_high: float
    sol_low: float
    sol_close: float
    sol_volume: float
    sol_taker_buy_ratio: float
    sol_num_trades: float
    sol_avg_trade_size: float
    sol_log_return: float
    sol_volatility_1d: float
    sol_volatility_7d: float
    sol_rsi: float
    sol_macd: float
    sol_bb_width: float
    # Global features
    btc_funding_rate: float
    hour_sin: float
    hour_cos: float
    dow_sin: float
    dow_cos: float


class MultiAssetForecastRequest(BaseModel):
    """Incoming payload for a multi-asset multi-horizon prediction."""
    sequence: List[MultiAssetTick] = Field(..., min_items=60, max_items=60)
    last_btc_close: float
    last_eth_close: float
    last_sol_close: float


class QuantilePrediction(BaseModel):
    """Quantile risk boundaries for a single asset at a single horizon."""
    p10_crash_boundary: float
    p50_median_forecast: float
    p90_breakout_boundary: float


class DirectionPrediction(BaseModel):
    """Directional classification output for a single asset at a single horizon."""
    direction: str  # "up", "neutral", or "down"
    confidence: float


class AssetForecast(BaseModel):
    """Combined quantile + direction forecast for a single asset."""
    quantiles_15m: QuantilePrediction
    quantiles_1h: QuantilePrediction
    quantiles_4h: QuantilePrediction
    direction_15m: DirectionPrediction
    direction_1h: DirectionPrediction
    direction_4h: DirectionPrediction


class MultiAssetForecastResponse(BaseModel):
    """Outgoing multi-horizon, multi-asset forecast response."""
    BTC: AssetForecast
    ETH: AssetForecast
    SOL: AssetForecast