"""
Hybrid Multi-Asset Forecasting Architecture (v2)
=================================================
Implements the guide's corrected architecture:
  Per-Asset Encoder (VSN + Asset Embeddings)
  -> Joint Cross-Asset Attention (GNN1)
  -> Joint Temporal Encoder (Mamba-3, no B*A flattening)
  -> Second Cross-Asset Attention (GNN2)
  -> Separate Per-Asset Heads (Quantile + Direction)

Key fixes over v1:
  - No more B*A flattening before Mamba (Point 6)
  - Learned asset embeddings (Point 6)
  - Multi-horizon outputs: 15m, 1h, 4h (Point 5)
  - Classification head with BCEWithLogitsLoss (Point 3)
  - Monotonic quantile constraint: P10 <= P50 <= P90 (Point 6)
  - Residual temporal blocks with LayerNorm and Dropout (Point 6)
"""
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
    """Dynamically weights feature importance per asset."""

    def __init__(self, input_dim, d_model, dropout_rate=0.2):
        super().__init__()
        self.feature_weights = nn.Linear(input_dim, input_dim)
        self.feature_processors = nn.ModuleList([
            GatedLinearUnit(1, d_model) for _ in range(input_dim)
        ])
        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (B, T, A, F)
        weights = torch.softmax(self.feature_weights(x), dim=-1)

        processed = []
        for i in range(x.size(-1)):
            feat = x[..., i:i+1]
            out = self.feature_processors[i](feat)
            processed.append(out * weights[..., i:i+1])

        out = sum(processed)
        return self.norm(self.dropout(out))  # (B, T, A, d_model)


class CrossAssetAttention(nn.Module):
    """Multi-head attention across the asset dimension.

    Captures spatial dependencies (e.g., BTC leading ETH and SOL).
    """

    def __init__(self, d_model, num_heads=4, dropout_rate=0.2):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (B, T, A, d_model)
        B, T, A, D = x.shape
        residual = x

        # Reshape for multi-head attention: (B*T, A, D)
        x_flat = x.view(B * T, A, D)

        qkv = self.qkv(x_flat).view(B * T, A, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B*T, H, A, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-1, -2)) * scale  # (B*T, H, A, A)
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # (B*T, H, A, head_dim)
        out = out.transpose(1, 2).contiguous().view(B * T, A, D)
        out = self.out_proj(out)
        out = self.dropout(out)

        out = out.view(B, T, A, D)
        return self.norm(out + residual)


class TemporalMambaBlock(nn.Module):
    """Joint temporal encoder using Mamba-3.

    Processes a concatenated cross-asset stream so Mamba sees all assets
    evolving together over time — NOT as independent timelines.

    Architecture:
        (B, T, A, d_model) -> flatten assets into features -> (B, T, A*d_model)
        -> project to d_model -> Mamba-3 -> project back -> (B, T, A, d_model)
    """

    def __init__(self, d_model, d_state, num_assets, dropout_rate=0.2):
        super().__init__()
        self.num_assets = num_assets
        joint_dim = d_model * num_assets

        # Project the concatenated asset features down to d_model for Mamba
        self.pre_proj = nn.Linear(joint_dim, d_model)
        self.pre_norm = nn.LayerNorm(d_model)

        self.mamba = Mamba3(
            d_model=d_model,
            d_state=d_state,
            expand=2,
            headdim=32,
            ngroups=1,
            is_mimo=False,
        )

        # Project back up to the joint dimension, then reshape
        self.post_proj = nn.Linear(d_model, joint_dim)
        self.post_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # x: (B, T, A, d_model)
        B, T, A, D = x.shape
        residual = x

        # Flatten assets into feature dimension: (B, T, A*d_model)
        x_joint = x.view(B, T, A * D)

        # Project to Mamba's d_model: (B, T, d_model)
        x_proj = self.pre_norm(self.pre_proj(x_joint))

        # Temporal sequence processing via Mamba-3: (B, T, d_model)
        mamba_out = self.mamba(x_proj)

        # Project back and reshape: (B, T, A*d_model) -> (B, T, A, d_model)
        out = self.post_proj(self.dropout(mamba_out))
        out = out.view(B, T, A, D)
        out = self.post_norm(out + residual)

        return out


class AssetHead(nn.Module):
    """Per-asset output head producing quantile and direction predictions.

    Quantile outputs use monotonic constraint: P10 <= P50 <= P90.
    Direction outputs are raw logits for BCEWithLogitsLoss.
    """

    def __init__(self, d_model, num_horizons=3, num_quantiles=3,
                 num_direction_classes=3, dropout_rate=0.2):
        super().__init__()
        self.num_horizons = num_horizons
        self.num_quantiles = num_quantiles
        self.num_direction_classes = num_direction_classes

        self.shared_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.LayerNorm(d_model),
        )

        # Quantile head: predicts base + 2 positive deltas per horizon
        # (base, delta_50, delta_90) so P10=base, P50=base+softplus(d1), P90=P50+softplus(d2)
        self.quantile_head = nn.Linear(d_model, num_horizons * num_quantiles)

        # Direction head: 3-class logits (down, neutral, up) per horizon
        self.direction_head = nn.Linear(d_model, num_horizons * num_direction_classes)

    def forward(self, x):
        # x: (B, d_model) — representation for a single asset
        h = self.shared_proj(x)

        # Quantile predictions with monotonic constraint
        raw_q = self.quantile_head(h)  # (B, H*Q)
        raw_q = raw_q.view(-1, self.num_horizons, self.num_quantiles)

        # Enforce P10 <= P50 <= P90
        p10 = raw_q[..., 0]                                    # base
        p50 = p10 + F.softplus(raw_q[..., 1])                  # base + positive delta
        p90 = p50 + F.softplus(raw_q[..., 2])                  # P50 + positive delta
        quantiles = torch.stack([p10, p50, p90], dim=-1)        # (B, H, Q)

        # Direction logits (raw, for BCEWithLogitsLoss)
        dir_logits = self.direction_head(h)  # (B, H*C)
        dir_logits = dir_logits.view(-1, self.num_horizons, self.num_direction_classes)

        return quantiles, dir_logits


class HybridArchitecture(nn.Module):
    """Multi-Asset Spatial-Temporal Forecasting Engine (v2).

    Architecture flow:
        Input (B, T, A, F)
        -> Per-Asset VSN + Asset Embeddings
        -> Cross-Asset Attention (GNN1)
        -> Joint Temporal Mamba-3
        -> Cross-Asset Attention (GNN2)
        -> Per-Asset Output Heads (Quantile + Direction)
    """

    def __init__(self, input_dim, d_model=128, d_state=64, num_assets=3,
                 num_quantiles=3, forecast_horizons=3, num_direction_classes=3,
                 dropout_rate=0.2):
        super().__init__()
        self.num_assets = num_assets
        self.num_quantiles = num_quantiles
        self.forecast_horizons = forecast_horizons
        self.d_model = d_model

        # 1. Per-Asset Feature Encoder
        self.vsn = VariableSelectionNetwork(input_dim, d_model, dropout_rate)

        # 2. Learned Asset Embeddings (Point 6)
        self.asset_embedding = nn.Embedding(num_assets, d_model)

        # 3. Pre-Mamba Cross-Asset Attention (GNN1)
        self.gnn_pre = CrossAssetAttention(d_model, num_heads=4, dropout_rate=dropout_rate)

        # 4. Joint Temporal Encoder (Mamba-3, NO B*A flattening)
        self.temporal = TemporalMambaBlock(d_model, d_state, num_assets, dropout_rate)

        # 5. Post-Mamba Cross-Asset Attention (GNN2)
        self.gnn_post = CrossAssetAttention(d_model, num_heads=4, dropout_rate=dropout_rate)

        # 6. Separate Per-Asset Output Heads
        self.asset_heads = nn.ModuleList([
            AssetHead(d_model, forecast_horizons, num_quantiles,
                      num_direction_classes, dropout_rate)
            for _ in range(num_assets)
        ])

        self.final_dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # x: (B, T, A, F)
        B, T, A, F = x.shape

        # 1. Per-Asset Feature Encoding
        x = self.vsn(x)  # (B, T, A, d_model)

        # 2. Add Learned Asset Embeddings
        asset_ids = torch.arange(A, device=x.device)
        asset_emb = self.asset_embedding(asset_ids)  # (A, d_model)
        x = x + asset_emb.unsqueeze(0).unsqueeze(0)  # broadcast: (B, T, A, d_model)

        # 3. Pre-Mamba Cross-Asset Attention
        x = self.gnn_pre(x)  # (B, T, A, d_model)

        # 4. Joint Temporal Processing (assets stay coupled through Mamba)
        x = self.temporal(x)  # (B, T, A, d_model)

        # 5. Post-Mamba Cross-Asset Attention
        x = self.gnn_post(x)  # (B, T, A, d_model)

        # 6. Extract last timestep for forecasting
        last = self.final_dropout(x[:, -1, :, :])  # (B, A, d_model)

        # 7. Per-Asset Heads
        all_quantiles = []
        all_directions = []
        for asset_i in range(A):
            asset_repr = last[:, asset_i, :]  # (B, d_model)
            q, d = self.asset_heads[asset_i](asset_repr)
            all_quantiles.append(q)
            all_directions.append(d)

        # Stack: (B, A, H, Q) and (B, A, H, C)
        quantiles = torch.stack(all_quantiles, dim=1)
        directions = torch.stack(all_directions, dim=1)

        return {
            'quantiles': quantiles,     # (B, Assets, Horizons, Quantiles)
            'directions': directions,   # (B, Assets, Horizons, Classes)
        }


class CombinedLoss(nn.Module):
    """Combined Pinball + Direction loss (Point 3).

    total_loss = alpha * quantile_pinball_loss + (1 - alpha) * direction_CE_loss
    """

    def __init__(self, quantiles=(0.1, 0.5, 0.9), alpha=0.5):
        super().__init__()
        self.quantiles = quantiles
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, outputs, reg_targets, dir_targets):
        """
        Parameters
        ----------
        outputs : dict
            'quantiles': (B, A, H, Q), 'directions': (B, A, H, C)
        reg_targets : torch.Tensor
            (B, A, H) — continuous log-return targets
        dir_targets : torch.LongTensor
            (B, A, H) — 3-class direction labels {0: down, 1: neutral, 2: up}
        """
        pred_q = outputs['quantiles']
        pred_d = outputs['directions']

        # Pinball loss
        targets_expanded = reg_targets.unsqueeze(-1)  # (B, A, H, 1)
        pinball = 0.0
        for i, q in enumerate(self.quantiles):
            error = targets_expanded - pred_q[..., i:i+1]
            pinball += torch.mean(torch.max(q * error, (q - 1.0) * error))
        pinball = pinball / len(self.quantiles)

        # Direction cross-entropy loss
        # pred_d: (B, A, H, C) -> reshape to (B*A*H, C)
        # dir_targets: (B, A, H) -> reshape to (B*A*H,)
        B, A, H, C = pred_d.shape
        pred_d_flat = pred_d.view(-1, C)
        dir_targets_flat = dir_targets.view(-1)
        direction_loss = self.ce_loss(pred_d_flat, dir_targets_flat)

        total = self.alpha * pinball + (1.0 - self.alpha) * direction_loss

        return total, {
            'pinball': pinball.item(),
            'direction': direction_loss.item(),
            'total': total.item(),
        }
