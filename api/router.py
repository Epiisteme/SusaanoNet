from fastapi import APIRouter, HTTPException
import numpy as np
import logging

from api.schemas import MultiAssetForecastRequest, MultiAssetForecastResponse, QuantilePrediction
from core.inference import RealTimeHybridInference
from api.live_feed import LiveDataAggregator

logger = logging.getLogger(__name__)
mamba_router = APIRouter()

# Initialize inference engine
try:
    logger.info("Initializing Hybrid Architecture (VSN -> GNN -> Mamba3-MIMO)...")
    engine = RealTimeHybridInference(artifacts_dir="artifacts")
    logger.info("Hybrid Engine Online and ready for multi-asset ticks.")
except Exception as e:
    logger.error(f"Failed to load Hybrid Engine: {e}")
    engine = None

# Initialize the Live Data Scraper
try:
    live_feed = LiveDataAggregator()
except Exception as e:
    logger.error(f"Failed to load Live Feed Aggregator: {e}")
    live_feed = None

@mamba_router.post("/predict/hybrid", response_model=MultiAssetForecastResponse)
async def get_multi_asset_quantiles(request: MultiAssetForecastRequest):
    """
    Receives 60 minutes of multi-asset market data (30 features total), pushes it through 
    the Hybrid Network (VSN -> GNN -> Mamba3 -> Quantile), and returns the 
    10th, 50th, and 90th percentile risk boundaries for BTC, ETH, and SOL.
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Prediction engine is currently offline.")

    # Extract into optimized Numpy array (60, 30)
    try:
        raw_matrix = np.array([
            [
                t.btc_open, t.btc_high, t.btc_low, t.btc_close, t.btc_volume,
                t.eth_open, t.eth_high, t.eth_low, t.eth_close, t.eth_volume,
                t.sol_open, t.sol_high, t.sol_low, t.sol_close, t.sol_volume,
                t.btc_funding_rate, t.sp500_close, t.dxy_close, t.gold_close, t.hash_rate, t.fng_score,
                t.btc_log_return, t.btc_volatility_1d, t.btc_volatility_7d,
                t.eth_log_return, t.eth_volatility_1d, t.eth_volatility_7d,
                t.sol_log_return, t.sol_volatility_1d, t.sol_volatility_7d
            ]
            for t in request.sequence
        ])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Data parsing error: {str(e)}")

    # Execute forward pass
    try:
        last_closes = [request.last_btc_close, request.last_eth_close, request.last_sol_close]
        
        predictions = engine.predict_quantiles(
            recent_60m_data=raw_matrix,
            last_closes=last_closes
        )
        
        return MultiAssetForecastResponse(
            BTC=QuantilePrediction(**predictions["BTC"]),
            ETH=QuantilePrediction(**predictions["ETH"]),
            SOL=QuantilePrediction(**predictions["SOL"])
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failure: {str(e)}")

@mamba_router.get("/predict/live")
async def get_live_production_quantiles():
    """
    Production Live Endpoint.
    Scrapes Binance for real-time historical data, constructs the tensor,
    executes inference, and returns historical prices and quantile boundaries.
    """
    if engine is None or live_feed is None:
        raise HTTPException(status_code=503, detail="Prediction engine or Live Feed is currently offline.")

    try:
        # Scrape and build live data
        raw_matrix, history, last_closes = live_feed.get_live_60_ticks()
        
        # Fire prediction
        predictions = engine.predict_quantiles(
            recent_60m_data=raw_matrix,
            last_closes=last_closes
        )
        
        return {
            "history": history,
            "forecasts": {
                "BTC": predictions["BTC"],
                "ETH": predictions["ETH"],
                "SOL": predictions["SOL"]
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Live feed failure: {str(e)}")