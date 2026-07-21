"""
API Router
==========
FastAPI endpoints for multi-asset multi-horizon forecasting.
"""
from fastapi import APIRouter, HTTPException
import numpy as np
import logging

from api.schemas import MultiAssetForecastRequest, MultiAssetForecastResponse
from core.inference import RealTimeHybridInference
from api.live_feed import LiveDataAggregator
from core.features import ASSET_FEATURE_COLS, GLOBAL_FEATURE_COLS, ASSETS

logger = logging.getLogger(__name__)
mamba_router = APIRouter()

# Initialize inference engine
try:
    logger.info("Initializing Hybrid Architecture (VSN -> GNN -> Mamba3-MIMO)...")
    engine = RealTimeHybridInference(artifacts_dir="artifacts")
    logger.info("Hybrid Engine Online.")
except Exception as e:
    logger.error(f"Failed to load Hybrid Engine: {e}")
    engine = None

# Initialize live data scraper
try:
    live_feed = LiveDataAggregator()
except Exception as e:
    logger.error(f"Failed to load Live Feed Aggregator: {e}")
    live_feed = None


@mamba_router.post("/predict/hybrid")
async def get_multi_asset_quantiles(request: MultiAssetForecastRequest):
    """
    Receives 60 ticks of multi-asset market data, executes inference through
    the Hybrid Network (VSN -> GNN -> Mamba3 -> Quantile + Direction),
    and returns multi-horizon risk boundaries and directional predictions.
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Prediction engine is currently offline.")

    # Build feature column extraction order from the request
    try:
        raw_matrix = np.array([
            [getattr(t, f'{asset}_{feat}') for asset in ASSETS for feat in ASSET_FEATURE_COLS]
            + [getattr(t, col) for col in GLOBAL_FEATURE_COLS]
            for t in request.sequence
        ])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Data parsing error: {str(e)}")

    try:
        last_closes = [request.last_btc_close, request.last_eth_close, request.last_sol_close]
        predictions = engine.predict(
            recent_data=raw_matrix,
            last_closes=last_closes,
        )
        return predictions
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failure: {str(e)}")


@mamba_router.get("/predict/live")
async def get_live_production_quantiles():
    """
    Production Live Endpoint.
    Scrapes Binance for real-time data, constructs the tensor,
    executes inference, and returns forecasts with historical data.
    """
    if engine is None or live_feed is None:
        raise HTTPException(status_code=503, detail="Prediction engine or Live Feed is offline.")

    try:
        raw_matrix, history, last_closes = live_feed.get_live_ticks()
        predictions = engine.predict(
            recent_data=raw_matrix,
            last_closes=last_closes,
        )
        return {
            "history": history,
            "forecasts": predictions,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Live feed failure: {str(e)}")