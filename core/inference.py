"""
Real-Time Inference Engine
==========================
Loads the trained Hybrid Architecture and executes multi-horizon
quantile regression + directional classification predictions.
"""
import os
import torch
import joblib
import numpy as np
from core.model import HybridArchitecture
from core.features import (
    ASSETS, ASSET_FEATURE_COLS, GLOBAL_FEATURE_COLS, FEATURES_PER_ASSET,
)


class RealTimeHybridInference:
    """Production inference engine for the Hybrid Architecture.

    Loads pre-trained weights and scaler, accepts raw feature matrices,
    and returns multi-horizon quantile + directional predictions.
    """

    def __init__(self, artifacts_dir: str = "artifacts"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        weights_path = os.path.join(artifacts_dir, "hybrid_mamba3_weights.pth")
        scaler_path = os.path.join(artifacts_dir, "hybrid_scaler.pkl")

        if not os.path.exists(weights_path) or not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Missing model artifacts in {artifacts_dir}/")

        self.scaler = joblib.load(scaler_path)

        # Number of per-asset features + global features
        num_asset_features = len(ASSET_FEATURE_COLS)
        num_global_features = len(GLOBAL_FEATURE_COLS)
        input_dim = num_asset_features + num_global_features

        self.model = HybridArchitecture(
            input_dim=input_dim,
            d_model=128,
            d_state=64,
            num_assets=3,
            num_quantiles=3,
            forecast_horizons=3,  # 15m, 1h, 4h
        )

        self.model.load_state_dict(
            torch.load(weights_path, map_location=self.device)
        )
        self.model.to(self.device)
        self.model.eval()

        # Feature layout constants
        self.num_asset_features = num_asset_features
        self.num_global_features = num_global_features
        self.num_total_features = len(ASSET_FEATURE_COLS) * 3 + num_global_features

    def predict(self, recent_data: np.ndarray, last_closes: list) -> dict:
        """Run inference on a raw feature matrix.

        Parameters
        ----------
        recent_data : np.ndarray
            Shape (seq_length, num_total_features). Feature order must match
            the column ordering defined in core/features.py.
        last_closes : list
            [btc_close, eth_close, sol_close] — most recent close prices.

        Returns
        -------
        dict
            Nested predictions per asset with quantile boundaries and directions.
        """
        seq_length = recent_data.shape[0]
        expected_cols = self.num_total_features

        if recent_data.shape[1] != expected_cols:
            raise ValueError(
                f"Expected {expected_cols} features, got {recent_data.shape[1]}"
            )

        # Scale the input
        scaled_input = self.scaler.transform(recent_data)

        # Reconstruct into spatial tensor: (Time, Assets, Features)
        tensor_data = np.zeros((seq_length, 3, FEATURES_PER_ASSET))

        for t in range(seq_length):
            row = scaled_input[t]
            for asset_i in range(3):
                # Per-asset features
                start = asset_i * self.num_asset_features
                end = start + self.num_asset_features
                tensor_data[t, asset_i, :self.num_asset_features] = row[start:end]

                # Global features (appended after all asset features in the flat row)
                global_start = 3 * self.num_asset_features
                tensor_data[t, asset_i, self.num_asset_features:] = row[
                    global_start:global_start + self.num_global_features
                ]

        # Add batch dimension -> (1, T, A, F)
        input_tensor = torch.tensor(
            tensor_data, dtype=torch.float32
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_tensor)

        # Parse outputs into structured response
        # (This will be fully implemented in Phase 2 after model rewrite)
        quantiles = outputs['quantiles'][0].cpu().numpy()  # (A, H, Q)
        directions = outputs['directions'][0].cpu().numpy()  # (A, H, 3)

        horizon_names = ['15m', '1h', '4h']
        direction_labels = ['down', 'neutral', 'up']
        results = {}

        for asset_i, asset_name in enumerate(('BTC', 'ETH', 'SOL')):
            asset_result = {}
            for h_i, h_name in enumerate(horizon_names):
                # Quantile predictions -> absolute prices
                log_ret_idx = ASSET_FEATURE_COLS.index('log_return')
                q_vals = quantiles[asset_i, h_i]

                asset_prices = {}
                for q_i, q_label in enumerate(['p10_crash_boundary', 'p50_median_forecast', 'p90_breakout_boundary']):
                    predicted_return = q_vals[q_i]
                    asset_prices[q_label] = float(
                        last_closes[asset_i] * np.exp(predicted_return)
                    )

                asset_result[f'quantiles_{h_name}'] = asset_prices

                # Direction predictions
                dir_logits_raw = directions[asset_i, h_i]
                e_x = np.exp(dir_logits_raw - np.max(dir_logits_raw))
                dir_probs = e_x / e_x.sum()
                
                predicted_class = int(np.argmax(dir_probs))
                asset_result[f'direction_{h_name}'] = {
                    'direction': direction_labels[predicted_class],
                    'confidence': float(np.max(dir_probs)),
                }

            results[asset_name] = asset_result

        return results