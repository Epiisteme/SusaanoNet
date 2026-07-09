import os
import torch
import joblib
import numpy as np
import pandas as pd
from core.model import HybridArchitecture

class RealTimeHybridInference:
    """
    The production execution engine for the Hybrid Architecture.
    Loads the Variable Selection Network, GNN, and Mamba-3 MIMO layers, 
    and returns multi-asset quantile risk boundaries.
    """
    def __init__(self, artifacts_dir: str = "artifacts"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        weights_path = os.path.join(artifacts_dir, "hybrid_mamba3_weights.pth")
        scaler_path = os.path.join(artifacts_dir, "hybrid_scaler.pkl")
        
        if not os.path.exists(weights_path) or not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Missing model artifacts in {artifacts_dir}/")

        self.scaler = joblib.load(scaler_path)
        
        # Build the Hybrid Architecture skeleton
        self.model = HybridArchitecture(
            input_dim=14, 
            d_model=128, 
            d_state=64, 
            num_assets=3, 
            num_quantiles=3, 
            forecast_horizons=1
        )
        
        # Inject the weights
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval() 
        
        # We need the exact column order the scaler was fit on (30 columns total, minus datetime)
        self.raw_columns = [
            'btc_open', 'btc_high', 'btc_low', 'btc_close', 'btc_volume', 
            'eth_open', 'eth_high', 'eth_low', 'eth_close', 'eth_volume', 
            'sol_open', 'sol_high', 'sol_low', 'sol_close', 'sol_volume', 
            'btc_funding_rate', 'sp500_close', 'dxy_close', 'gold_close', 'hash_rate', 'fng_score', 
            'btc_log_return', 'btc_volatility_1d', 'btc_volatility_7d', 
            'eth_log_return', 'eth_volatility_1d', 'eth_volatility_7d', 
            'sol_log_return', 'sol_volatility_1d', 'sol_volatility_7d'
        ]
        
        # How features are grouped per asset node internally
        self.btc_cols = [c for c in self.raw_columns if c.startswith('btc_') and c != 'btc_funding_rate']
        self.eth_cols = [c for c in self.raw_columns if c.startswith('eth_')]
        self.sol_cols = [c for c in self.raw_columns if c.startswith('sol_')]
        self.global_cols = ['btc_funding_rate', 'sp500_close', 'dxy_close', 'gold_close', 'hash_rate', 'fng_score']
        
        # To find indices for fast tensor reconstruction
        self.btc_idx = [self.raw_columns.index(c) for c in self.btc_cols]
        self.eth_idx = [self.raw_columns.index(c) for c in self.eth_cols]
        self.sol_idx = [self.raw_columns.index(c) for c in self.sol_cols]
        self.global_idx = [self.raw_columns.index(c) for c in self.global_cols]
        
    def predict_quantiles(self, recent_60m_data: np.ndarray, last_closes: list) -> dict:
        """
        Accepts a (60, 30) numpy array of the last 60 periods of multi-asset tick data.
        Returns a dict of the 10th, 50th, and 90th percentile predicted prices for BTC, ETH, SOL.
        """
        if recent_60m_data.shape != (60, 30):
            raise ValueError(f"Expected input shape (60, 30), got {recent_60m_data.shape}")

        # 1. Scale the raw 30-feature incoming data
        scaled_input = self.scaler.transform(recent_60m_data) # (60, 30)
        
        # 2. Reconstruct into the Spatial Tensor: (Time, Assets, Features) -> (60, 3, 14)
        tensor_data = np.zeros((60, 3, 14))
        for t in range(60):
            row = scaled_input[t]
            # BTC Node (8 specific + 6 global)
            tensor_data[t, 0, :8] = row[self.btc_idx]
            tensor_data[t, 0, 8:] = row[self.global_idx]
            # ETH Node
            tensor_data[t, 1, :8] = row[self.eth_idx]
            tensor_data[t, 1, 8:] = row[self.global_idx]
            # SOL Node
            tensor_data[t, 2, :8] = row[self.sol_idx]
            tensor_data[t, 2, 8:] = row[self.global_idx]
            
        # 3. Add Batch Dimension and send to device -> (1, 60, 3, 14)
        input_tensor = torch.tensor(tensor_data, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # 4. Forward Pass through VSN -> GNN -> Mamba-3 -> Quantile Head
        with torch.no_grad():
            # Output Shape: (Batch=1, Assets=3, Horizons=1, Quantiles=3)
            quantiles = self.model(input_tensor)
            
        # Remove Batch and Horizon dims -> (3, 3) (Assets, Quantiles)
        quantiles = quantiles[0, :, 0, :].cpu().numpy()
        
        # 5. Inverse Transform the predicted log returns back into absolute prices
        results = {}
        assets = ['BTC', 'ETH', 'SOL']
        log_return_indices = [
            self.raw_columns.index('btc_log_return'),
            self.raw_columns.index('eth_log_return'),
            self.raw_columns.index('sol_log_return')
        ]
        
        for asset_i, asset_name in enumerate(assets):
            asset_quantiles = []
            for q_i in range(3): # 0=10th, 1=50th, 2=90th
                predicted_scaled_return = quantiles[asset_i, q_i]
                
                # Create a dummy row of 30 zeros to inverse transform
                dummy_row = np.zeros((1, 30))
                dummy_row[0, log_return_indices[asset_i]] = predicted_scaled_return
                
                unscaled_log_return = self.scaler.inverse_transform(dummy_row)[0, log_return_indices[asset_i]]
                predicted_price = last_closes[asset_i] * np.exp(unscaled_log_return)
                asset_quantiles.append(float(predicted_price))
                
            results[asset_name] = {
                "p10_crash_boundary": asset_quantiles[0],
                "p50_median_forecast": asset_quantiles[1],
                "p90_breakout_boundary": asset_quantiles[2]
            }
            
        return results