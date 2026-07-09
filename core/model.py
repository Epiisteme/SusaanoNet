import torch
import torch.nn as nn
import torch.nn.functional as F
from core.mamba3 import Mamba3

class GatedLinearUnit(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim * 2)
        
    def forward(self, x):
        x = self.fc(x)
        val, gate = x.chunk(2, dim=-1)
        return val * torch.sigmoid(gate)

class VariableSelectionNetwork(nn.Module):
    """
    Dynamically weights the importance of different features (e.g., Price vs Macro).
    """
    def __init__(self, input_dim, d_model):
        super().__init__()
        self.feature_weights = nn.Linear(input_dim, input_dim)
        self.feature_processors = nn.ModuleList([
            GatedLinearUnit(1, d_model) for _ in range(input_dim)
        ])
        
    def forward(self, x):
        # x shape: (Batch, Time, Assets, Features)
        weights = torch.softmax(self.feature_weights(x), dim=-1)
        
        processed_features = []
        for i in range(x.size(-1)):
            feat = x[..., i:i+1] # Keep feature dim
            processed = self.feature_processors[i](feat)
            weight = weights[..., i:i+1]
            processed_features.append(processed * weight)
            
        # Sum across features to get dense representation
        out = sum(processed_features)
        return out # Shape: (Batch, Time, Assets, d_model)

class DynamicGraphAttention(nn.Module):
    """
    Maps spatial dependencies (e.g., how BTC drags ETH and SOL).
    """
    def __init__(self, d_model):
        super().__init__()
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.scale = d_model ** -0.5
        
    def forward(self, x):
        # x shape: (Batch, Time, Assets, d_model)
        # We compute attention across the Assets dimension
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        # Attention scores: (Batch, Time, Assets, Assets)
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn_weights = torch.softmax(attn_scores, dim=-1)
        
        # Context vectors
        out = torch.matmul(attn_weights, v)
        return out + x # Residual connection

class HybridArchitecture(nn.Module):
    """
    The full Multi-Asset, Probabilistic, Spatial-Temporal Forecasting Engine.
    """
    def __init__(self, input_dim, d_model=128, d_state=64, num_assets=3, num_quantiles=3, forecast_horizons=1):
        super().__init__()
        self.num_assets = num_assets
        self.num_quantiles = num_quantiles
        self.forecast_horizons = forecast_horizons
        
        # Feature Processing
        self.vsn = VariableSelectionNetwork(input_dim, d_model)
        
        # Spatial Encoding
        self.gnn = DynamicGraphAttention(d_model)
        
        # Sequence Learning (Mamba-3 MIMO with Complex RoPE States)
        self.mamba3 = Mamba3(
            d_model=d_model,
            d_state=d_state,
            expand=2,
            headdim=32,
            ngroups=1,
            is_mimo=True,
            mimo_rank=num_assets  # R parallel streams for the assets
        )
        
        # Multi-Horizon Quantile Projection
        output_dim = num_quantiles * forecast_horizons
        self.regression_head = nn.Linear(d_model, output_dim)
        
    def forward(self, x):
        # x shape: (Batch, Time, Assets, Features)
        B, T, A, F = x.shape
        
        # Pass through VSN
        x = self.vsn(x) # (B, T, A, d_model)
        
        # Pass through GNN
        x = self.gnn(x) # (B, T, A, d_model)
        
        # Mamba-3 expects (Batch, Time, d_model) for MIMO where streams are internal.
        # Alternatively, we can flatten Batch and Assets: (B*A, T, d_model)
        x_flat = x.view(B * A, T, -1)
        mamba_out = self.mamba3(x_flat) # (B*A, T, d_model)
        
        # Extract last timestep for forecasting
        last_step_out = mamba_out[:, -1, :] # (B*A, d_model)
        
        # Project to Quantiles
        quantiles = self.regression_head(last_step_out) # (B*A, Quantiles * Horizons)
        
        # Reshape back to (Batch, Assets, Horizons, Quantiles)
        quantiles = quantiles.view(B, A, self.forecast_horizons, self.num_quantiles)
        
        return quantiles

class PinballLoss(nn.Module):
    """
    Quantile Regression Loss (Pinball Loss)
    Forces the network to output 10th, 50th, and 90th percentile risk boundaries.
    """
    def __init__(self, quantiles=[0.1, 0.5, 0.9]):
        super().__init__()
        self.quantiles = quantiles
        
    def forward(self, predictions, targets):
        # predictions: (Batch, Assets, Horizons, Quantiles)
        # targets: (Batch, Assets, Horizons)
        
        # Expand targets to match predictions shape: (Batch, Assets, Horizons, 1)
        targets = targets.unsqueeze(-1)
        
        loss = 0.0
        for i, q in enumerate(self.quantiles):
            # Error = Target - Prediction
            error = targets - predictions[..., i:i+1]
            
            # Pinball calculation: max(q * error, (q - 1) * error)
            q_loss = torch.max(q * error, (q - 1.0) * error)
            loss += torch.mean(q_loss)
            
        return loss / len(self.quantiles)
