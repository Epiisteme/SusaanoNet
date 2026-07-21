"""
Hybrid Architecture Training Script (v2)
=========================================
Leak-free, walk-forward validated, multi-horizon training with
combined quantile + directional loss.

Fixes from guide's review:
  - Point 1: Scaler fit on train only. 60/20/20 split. TimeSeriesSplit.
  - Point 3: Combined Pinball + CrossEntropy loss.
  - Point 5: Multi-horizon targets (15m, 1h, 4h).
  - Point 9: Configurable hyperparameters, gradient clipping, cosine decay.
  - Point 10: Full metrics suite (ROC-AUC, Balanced Accuracy, MCC, PR-AUC).
"""
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    roc_auc_score, balanced_accuracy_score, matthews_corrcoef,
    average_precision_score, classification_report,
)
import joblib
import os
import json
from datetime import datetime

from core.model import HybridArchitecture, CombinedLoss
from core.features import (
    ASSETS, ASSET_FEATURE_COLS, GLOBAL_FEATURE_COLS,
    FEATURES_PER_ASSET, get_feature_columns, get_target_columns,
)

# ============================================================================
# Configuration
# ============================================================================

CONFIG = {
    'csv_path': 'data/multi_asset_15m_structural_v2.csv',
    'seq_length': 30,
    'batch_size': 128,
    'epochs': 50,
    'lr': 0.0001,
    'd_model': 128,
    'd_state': 64,
    'dropout': 0.2,
    'weight_decay': 1e-4,
    'grad_clip': 1.0,
    'early_stop_patience': 8,
    'loss_alpha': 0.5,  # 0.5 * pinball + 0.5 * direction_CE
    'warmup_epochs': 3,
    'walk_forward_splits': 5,
    'walk_forward_gap': 60,
    'seed': 42,
    'horizons': {'15m': 1, '1h': 4, '4h': 16},
}

HORIZON_NAMES = list(CONFIG['horizons'].keys())
NUM_HORIZONS = len(HORIZON_NAMES)
NUM_ASSETS = 3
NUM_ASSET_FEATURES = len(ASSET_FEATURE_COLS)
NUM_GLOBAL_FEATURES = len(GLOBAL_FEATURE_COLS)


# ============================================================================
# Data Loading (Leak-Free)
# ============================================================================

def load_data(csv_path):
    """Load CSV and separate features from targets."""
    print("Loading Multi-Asset CSV...")
    df = pd.read_csv(csv_path)

    if 'datetime' in df.columns:
        df = df.drop(columns=['datetime'])

    feature_cols = get_feature_columns()
    target_info = get_target_columns(CONFIG['horizons'])

    # Verify all columns exist
    all_needed = feature_cols + target_info['regression'] + target_info['classification']
    missing = [c for c in all_needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    features_df = df[feature_cols]
    reg_targets_df = df[target_info['regression']]
    dir_targets_df = df[target_info['classification']]

    return features_df, reg_targets_df, dir_targets_df


def build_tensor(features_np, num_timesteps):
    """Reconstruct flat feature array into (Time, Assets, Features) tensor."""
    tensor = np.zeros((num_timesteps, NUM_ASSETS, FEATURES_PER_ASSET))

    for t in range(num_timesteps):
        row = features_np[t]
        for asset_i in range(NUM_ASSETS):
            start = asset_i * NUM_ASSET_FEATURES
            end = start + NUM_ASSET_FEATURES
            tensor[t, asset_i, :NUM_ASSET_FEATURES] = row[start:end]

            global_start = NUM_ASSETS * NUM_ASSET_FEATURES
            tensor[t, asset_i, NUM_ASSET_FEATURES:] = row[
                global_start:global_start + NUM_GLOBAL_FEATURES
            ]

    return tensor


def create_sequences(tensor, reg_targets, dir_targets, seq_length):
    """Create windowed sequences with multi-horizon targets.

    For each window ending at index i, the targets are pre-computed
    in the CSV at row i (the last row of the input window).
    """
    xs, ys_reg, ys_dir = [], [], []
    n = len(tensor)

    for i in range(seq_length, n):
        xs.append(tensor[i - seq_length:i])

        # Regression targets: (Assets, Horizons)
        reg_row = reg_targets[i - 1]  # targets at the last tick of the window
        reg_reshaped = reg_row.reshape(NUM_ASSETS, NUM_HORIZONS)
        ys_reg.append(reg_reshaped)

        # Direction targets: (Assets, Horizons)
        dir_row = dir_targets[i - 1]
        dir_reshaped = dir_row.reshape(NUM_ASSETS, NUM_HORIZONS)
        ys_dir.append(dir_reshaped)

    return np.array(xs), np.array(ys_reg), np.array(ys_dir)


# ============================================================================
# Metrics (Point 10)
# ============================================================================

def compute_metrics(pred_dir_probs, true_dir, pred_quantiles, true_returns):
    """Compute the full metrics suite required by the guide.

    Parameters
    ----------
    pred_dir_probs : np.ndarray (N, A, H, C) — direction softmax probabilities
    true_dir : np.ndarray (N, A, H) — true direction labels {0, 1, 2}
    pred_quantiles : np.ndarray (N, A, H, Q) — predicted quantile values
    true_returns : np.ndarray (N, A, H) — true continuous returns

    Returns
    -------
    dict of metrics
    """
    results = {}
    asset_names = ['BTC', 'ETH', 'SOL']

    for a_i, asset in enumerate(asset_names):
        for h_i, horizon in enumerate(HORIZON_NAMES):
            key = f"{asset}_{horizon}"

            y_true = true_dir[:, a_i, h_i].astype(int)
            y_probs = pred_dir_probs[:, a_i, h_i, :]  # (N, 3)
            y_pred = np.argmax(y_probs, axis=-1)

            # Balanced Accuracy
            results[f'{key}_balanced_acc'] = balanced_accuracy_score(y_true, y_pred)

            # MCC
            results[f'{key}_mcc'] = matthews_corrcoef(y_true, y_pred)

            # ROC-AUC (one-vs-rest, macro)
            try:
                results[f'{key}_roc_auc'] = roc_auc_score(
                    y_true, y_probs, multi_class='ovr', average='macro'
                )
            except ValueError:
                results[f'{key}_roc_auc'] = 0.5

            # PR-AUC (one-vs-rest, macro)
            try:
                from sklearn.preprocessing import label_binarize
                y_bin = label_binarize(y_true, classes=[0, 1, 2])
                pr_aucs = []
                for c in range(3):
                    pr_aucs.append(average_precision_score(y_bin[:, c], y_probs[:, c]))
                results[f'{key}_pr_auc'] = np.mean(pr_aucs)
            except ValueError:
                results[f'{key}_pr_auc'] = 0.0

            # Actionable accuracy (excluding neutral predictions)
            actionable_mask = y_pred != 1  # not neutral
            if actionable_mask.sum() > 0:
                results[f'{key}_actionable_acc'] = (
                    (y_pred[actionable_mask] == y_true[actionable_mask]).mean()
                )
                results[f'{key}_coverage'] = actionable_mask.mean()
            else:
                results[f'{key}_actionable_acc'] = 0.0
                results[f'{key}_coverage'] = 0.0

            # Quantile coverage (what % of true returns fall within P10-P90)
            p10 = pred_quantiles[:, a_i, h_i, 0]
            p90 = pred_quantiles[:, a_i, h_i, 2]
            y_ret = true_returns[:, a_i, h_i]
            coverage_80 = ((y_ret >= p10) & (y_ret <= p90)).mean()
            results[f'{key}_quantile_coverage_80'] = coverage_80

            # Majority-class baseline accuracy
            from collections import Counter
            majority_class = Counter(y_true).most_common(1)[0]
            results[f'{key}_majority_baseline'] = majority_class[1] / len(y_true)

    return results


# ============================================================================
# Training Loop
# ============================================================================

def train_one_epoch(model, loader, criterion, optimizer, device, grad_clip):
    model.train()
    total_loss = 0.0
    loss_details = {'pinball': 0.0, 'direction': 0.0}

    for batch_X, batch_reg, batch_dir in loader:
        batch_X = batch_X.to(device)
        batch_reg = batch_reg.to(device)
        batch_dir = batch_dir.to(device).long()

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss, details = criterion(outputs, batch_reg, batch_dir)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item() * batch_X.size(0)
        loss_details['pinball'] += details['pinball'] * batch_X.size(0)
        loss_details['direction'] += details['direction'] * batch_X.size(0)

    n = len(loader.dataset)
    return total_loss / n, {k: v / n for k, v in loss_details.items()}


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_q, all_d, all_reg_t, all_dir_t = [], [], [], []

    for batch_X, batch_reg, batch_dir in loader:
        batch_X = batch_X.to(device)
        batch_reg = batch_reg.to(device)
        batch_dir = batch_dir.to(device).long()

        outputs = model(batch_X)
        loss, _ = criterion(outputs, batch_reg, batch_dir)
        total_loss += loss.item() * batch_X.size(0)

        # Collect predictions for metrics
        all_q.append(outputs['quantiles'].cpu().numpy())
        dir_probs = torch.softmax(outputs['directions'], dim=-1)
        all_d.append(dir_probs.cpu().numpy())
        all_reg_t.append(batch_reg.cpu().numpy())
        all_dir_t.append(batch_dir.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    pred_q = np.concatenate(all_q)
    pred_d = np.concatenate(all_d)
    true_reg = np.concatenate(all_reg_t)
    true_dir = np.concatenate(all_dir_t)

    metrics = compute_metrics(pred_d, true_dir, pred_q, true_reg)
    return avg_loss, metrics


# ============================================================================
# Walk-Forward Validation (Point 1)
# ============================================================================

def run_walk_forward(features_df, reg_targets_df, dir_targets_df):
    """Walk-forward validation with expanding window and gap."""
    print("\n" + "=" * 70)
    print("WALK-FORWARD VALIDATION")
    print("=" * 70)

    tscv = TimeSeriesSplit(
        n_splits=CONFIG['walk_forward_splits'],
        gap=CONFIG['walk_forward_gap'],
    )

    features_np = features_df.values
    reg_np = reg_targets_df.values
    dir_np = dir_targets_df.values

    fold_results = []

    for fold_i, (train_idx, val_idx) in enumerate(tscv.split(features_np)):
        print(f"\n--- Fold {fold_i + 1}/{CONFIG['walk_forward_splits']} ---")
        print(f"    Train: {len(train_idx)} rows | Val: {len(val_idx)} rows")

        # Fit scaler on TRAINING FOLD ONLY (Point 1)
        scaler = StandardScaler()
        train_features_scaled = scaler.fit_transform(features_np[train_idx])
        val_features_scaled = scaler.transform(features_np[val_idx])

        # Build tensors
        train_tensor = build_tensor(train_features_scaled, len(train_idx))
        val_tensor = build_tensor(val_features_scaled, len(val_idx))

        train_reg = reg_np[train_idx]
        val_reg = reg_np[val_idx]
        train_dir = dir_np[train_idx]
        val_dir = dir_np[val_idx]

        # Create sequences
        X_train, y_reg_train, y_dir_train = create_sequences(
            train_tensor, train_reg, train_dir, CONFIG['seq_length']
        )
        X_val, y_reg_val, y_dir_val = create_sequences(
            val_tensor, val_reg, val_dir, CONFIG['seq_length']
        )

        if len(X_train) == 0 or len(X_val) == 0:
            print("    Skipping fold: not enough data.")
            continue

        # DataLoaders
        train_ds = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_reg_train, dtype=torch.float32),
            torch.tensor(y_dir_train, dtype=torch.float32),
        )
        val_ds = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_reg_val, dtype=torch.float32),
            torch.tensor(y_dir_val, dtype=torch.float32),
        )
        train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False)

        # Train this fold
        device = _get_device()
        model = _build_model().to(device)
        criterion = CombinedLoss(alpha=CONFIG['loss_alpha'])
        optimizer = optim.AdamW(
            model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay']
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=CONFIG['epochs'] - CONFIG['warmup_epochs']
        )

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(CONFIG['epochs']):
            # Warmup: linearly increase LR for the first few epochs
            if epoch < CONFIG['warmup_epochs']:
                warmup_lr = CONFIG['lr'] * (epoch + 1) / CONFIG['warmup_epochs']
                for pg in optimizer.param_groups:
                    pg['lr'] = warmup_lr

            train_loss, _ = train_one_epoch(
                model, train_loader, criterion, optimizer, device, CONFIG['grad_clip']
            )
            val_loss, val_metrics = evaluate(model, val_loader, criterion, device)

            if epoch >= CONFIG['warmup_epochs']:
                scheduler.step()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_metrics = val_metrics
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0 or epoch == 0:
                avg_bacc = np.mean([v for k, v in val_metrics.items() if 'balanced_acc' in k])
                print(f"    Epoch {epoch+1:3d} | Train: {train_loss:.4f} | "
                      f"Val: {val_loss:.4f} | Bal.Acc: {avg_bacc:.4f}")

            if patience_counter >= CONFIG['early_stop_patience']:
                print(f"    Early stopping at epoch {epoch+1}")
                break

        fold_results.append(best_metrics)
        print(f"    Best Val Loss: {best_val_loss:.4f}")

    # Aggregate fold results
    if fold_results:
        print("\n" + "=" * 70)
        print("WALK-FORWARD AGGREGATE RESULTS")
        print("=" * 70)
        all_keys = fold_results[0].keys()
        for key in sorted(all_keys):
            values = [fr[key] for fr in fold_results]
            mean_val = np.mean(values)
            std_val = np.std(values)
            print(f"  {key:40s}: {mean_val:.4f} +/- {std_val:.4f}")

    return fold_results


# ============================================================================
# Final Training (60/20/20 Split)
# ============================================================================

def train_final_model(features_df, reg_targets_df, dir_targets_df):
    """Train the final model with a strict 60/20/20 chronological split."""
    print("\n" + "=" * 70)
    print("FINAL MODEL TRAINING (60/20/20 Split)")
    print("=" * 70)

    os.makedirs('artifacts', exist_ok=True)

    features_np = features_df.values
    reg_np = reg_targets_df.values
    dir_np = dir_targets_df.values
    n = len(features_np)

    train_end = int(n * 0.6)
    val_end = int(n * 0.8)

    print(f"Train: 0-{train_end} | Val: {train_end}-{val_end} | Test: {val_end}-{n}")

    # Fit scaler on TRAINING DATA ONLY (Point 1)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(features_np[:train_end])
    val_scaled = scaler.transform(features_np[train_end:val_end])
    test_scaled = scaler.transform(features_np[val_end:])
    joblib.dump(scaler, 'artifacts/hybrid_scaler.pkl')

    # Build tensors
    train_tensor = build_tensor(train_scaled, train_end)
    val_tensor = build_tensor(val_scaled, val_end - train_end)
    test_tensor = build_tensor(test_scaled, n - val_end)

    # Create sequences
    X_train, yr_train, yd_train = create_sequences(
        train_tensor, reg_np[:train_end], dir_np[:train_end], CONFIG['seq_length']
    )
    X_val, yr_val, yd_val = create_sequences(
        val_tensor, reg_np[train_end:val_end], dir_np[train_end:val_end], CONFIG['seq_length']
    )
    X_test, yr_test, yd_test = create_sequences(
        test_tensor, reg_np[val_end:], dir_np[val_end:], CONFIG['seq_length']
    )

    print(f"Sequences -> Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # DataLoaders
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(yr_train, dtype=torch.float32),
        torch.tensor(yd_train, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(yr_val, dtype=torch.float32),
        torch.tensor(yd_val, dtype=torch.float32),
    )
    test_ds = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(yr_test, dtype=torch.float32),
        torch.tensor(yd_test, dtype=torch.float32),
    )

    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=CONFIG['batch_size'], shuffle=False)

    # Model setup
    device = _get_device()
    model = _build_model().to(device)
    criterion = CombinedLoss(alpha=CONFIG['loss_alpha'])
    optimizer = optim.AdamW(
        model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay']
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG['epochs'] - CONFIG['warmup_epochs']
    )

    best_val_loss = float('inf')
    patience_counter = 0

    print("\nTraining (VSN -> GNN1 -> Mamba3 -> GNN2 -> AssetHeads)...")
    for epoch in range(CONFIG['epochs']):
        if epoch < CONFIG['warmup_epochs']:
            warmup_lr = CONFIG['lr'] * (epoch + 1) / CONFIG['warmup_epochs']
            for pg in optimizer.param_groups:
                pg['lr'] = warmup_lr

        train_loss, train_details = train_one_epoch(
            model, train_loader, criterion, optimizer, device, CONFIG['grad_clip']
        )
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)

        if epoch >= CONFIG['warmup_epochs']:
            scheduler.step()

        # Checkpoint on VALIDATION loss only (Point 1)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'artifacts/hybrid_mamba3_weights.pth')
        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            avg_bacc = np.mean([v for k, v in val_metrics.items() if 'balanced_acc' in k])
            avg_auc = np.mean([v for k, v in val_metrics.items() if 'roc_auc' in k])
            print(f"Epoch {epoch+1:3d}/{CONFIG['epochs']} | "
                  f"Train: {train_loss:.4f} (P:{train_details['pinball']:.4f} D:{train_details['direction']:.4f}) | "
                  f"Val: {val_loss:.4f} | Bal.Acc: {avg_bacc:.4f} | AUC: {avg_auc:.4f}")

        if patience_counter >= CONFIG['early_stop_patience']:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Final test evaluation (run ONCE, Point 1)
    print("\n" + "=" * 70)
    print("FINAL TEST SET EVALUATION (Untouched Data)")
    print("=" * 70)

    model.load_state_dict(torch.load('artifacts/hybrid_mamba3_weights.pth', map_location=device))
    test_loss, test_metrics = evaluate(model, test_loader, criterion, device)

    print(f"Test Loss: {test_loss:.4f}\n")
    for key in sorted(test_metrics.keys()):
        print(f"  {key:40s}: {test_metrics[key]:.4f}")

    # Save results
    results = {
        'config': CONFIG,
        'test_loss': test_loss,
        'test_metrics': test_metrics,
        'timestamp': datetime.now().isoformat(),
    }
    # Convert non-serializable types
    results['config']['horizons'] = dict(results['config']['horizons'])
    with open('artifacts/training_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\nTraining complete. Artifacts saved to artifacts/")
    return test_metrics


# ============================================================================
# Helpers
# ============================================================================

def _get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def _build_model():
    return HybridArchitecture(
        input_dim=FEATURES_PER_ASSET,
        d_model=CONFIG['d_model'],
        d_state=CONFIG['d_state'],
        num_assets=NUM_ASSETS,
        num_quantiles=3,
        forecast_horizons=NUM_HORIZONS,
        num_direction_classes=3,
        dropout_rate=CONFIG['dropout'],
    )


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    set_seed(CONFIG['seed'])

    features_df, reg_targets_df, dir_targets_df = load_data(CONFIG['csv_path'])
    print(f"Total rows: {len(features_df)}")
    print(f"Features per row: {features_df.shape[1]}")
    print(f"Features per asset node: {FEATURES_PER_ASSET}")

    # 1. Walk-forward validation to establish confidence intervals
    fold_results = run_walk_forward(features_df, reg_targets_df, dir_targets_df)

    # 2. Train final model on the full 60/20/20 split
    test_metrics = train_final_model(features_df, reg_targets_df, dir_targets_df)
