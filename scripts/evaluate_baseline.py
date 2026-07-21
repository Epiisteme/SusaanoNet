"""
Baseline Model Evaluator (Point 8)
====================================
Establishes simple baselines that the Mamba model must beat.
Uses the same leak-free 60/20/20 split and identical metrics.

Baselines:
  1. Always-Up: predict class 2 for every sample
  2. Always-Down: predict class 0 for every sample
  3. Previous-Return Direction: predict the sign of the most recent return
  4. Moving-Average Momentum: 20-period SMA crossover signal
  5. Logistic Regression: sklearn linear classifier
  6. Gradient-Boosted Trees: XGBoost classifier
"""
import sys
import os
# Allow imports from the parent directory (the root of the project)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, balanced_accuracy_score, matthews_corrcoef,
    average_precision_score, classification_report,
)
import json
import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("WARNING: xgboost not installed. Skipping XGBoost baseline.")
    print("Install with: pip install xgboost")

from core.features import (
    ASSETS, ASSET_FEATURE_COLS, GLOBAL_FEATURE_COLS,
    get_feature_columns, get_target_columns,
)


HORIZONS = {'15m': 1, '1h': 4, '4h': 16}
HORIZON_NAMES = list(HORIZONS.keys())


def load_data(csv_path):
    """Load and split data with the same 60/20/20 scheme as training."""
    df = pd.read_csv(csv_path)
    if 'datetime' in df.columns:
        df = df.drop(columns=['datetime'])

    feature_cols = get_feature_columns()
    target_info = get_target_columns(HORIZONS)

    features = df[feature_cols].values
    dir_targets = df[target_info['classification']].values
    reg_targets = df[target_info['regression']].values

    n = len(features)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)

    # Fit scaler on train only
    scaler = StandardScaler()
    X_train = scaler.fit_transform(features[:train_end])
    X_val = scaler.transform(features[train_end:val_end])
    X_test = scaler.transform(features[val_end:])

    y_dir_train = dir_targets[:train_end]
    y_dir_val = dir_targets[train_end:val_end]
    y_dir_test = dir_targets[val_end:]

    y_reg_test = reg_targets[val_end:]

    return X_train, X_val, X_test, y_dir_train, y_dir_val, y_dir_test, y_reg_test


def compute_metrics(y_true, y_pred, y_probs=None):
    """Compute metrics for a single asset-horizon combination."""
    results = {}
    results['balanced_acc'] = balanced_accuracy_score(y_true, y_pred)
    results['mcc'] = matthews_corrcoef(y_true, y_pred)

    # Actionable accuracy (non-neutral predictions)
    actionable = y_pred != 1
    if actionable.sum() > 0:
        results['actionable_acc'] = (y_pred[actionable] == y_true[actionable]).mean()
        results['coverage'] = actionable.mean()
    else:
        results['actionable_acc'] = 0.0
        results['coverage'] = 0.0

    # ROC-AUC requires probability scores
    if y_probs is not None:
        try:
            results['roc_auc'] = roc_auc_score(
                y_true, y_probs, multi_class='ovr', average='macro'
            )
        except ValueError:
            results['roc_auc'] = 0.5

        try:
            y_bin = label_binarize(y_true, classes=[0, 1, 2])
            pr_aucs = []
            for c in range(3):
                if y_bin[:, c].sum() > 0:
                    pr_aucs.append(average_precision_score(y_bin[:, c], y_probs[:, c]))
            results['pr_auc'] = np.mean(pr_aucs) if pr_aucs else 0.0
        except ValueError:
            results['pr_auc'] = 0.0
    else:
        results['roc_auc'] = 0.5
        results['pr_auc'] = 0.0

    return results


def evaluate_baseline(name, y_pred_all, y_true_all, y_probs_all=None):
    """Evaluate a baseline across all assets and horizons."""
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")

    all_metrics = {}
    asset_names = ['BTC', 'ETH', 'SOL']
    num_assets = len(asset_names)
    num_horizons = len(HORIZON_NAMES)

    for a_i, asset in enumerate(asset_names):
        for h_i, horizon in enumerate(HORIZON_NAMES):
            col_idx = a_i * num_horizons + h_i
            y_true = y_true_all[:, col_idx].astype(int)
            y_pred = y_pred_all[:, col_idx].astype(int)
            y_probs = y_probs_all[:, col_idx] if y_probs_all is not None else None

            metrics = compute_metrics(y_true, y_pred, y_probs)
            key = f"{asset}_{horizon}"
            all_metrics[key] = metrics

            print(f"  {key:12s} | Bal.Acc: {metrics['balanced_acc']:.4f} | "
                  f"MCC: {metrics['mcc']:+.4f} | AUC: {metrics['roc_auc']:.4f} | "
                  f"Act.Acc: {metrics['actionable_acc']:.4f} ({metrics['coverage']:.1%})")

    return all_metrics


def run_baselines(csv_path='data/multi_asset_15m_structural_v2.csv'):
    """Run all baselines and report results."""
    print("Loading data...")
    X_train, X_val, X_test, y_dir_train, y_dir_val, y_dir_test, y_reg_test = load_data(csv_path)
    n_test = len(X_test)
    n_cols = y_dir_test.shape[1]

    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {n_test}")

    all_baseline_results = {}

    # --- Baseline 1: Always-Down ---
    pred = np.zeros((n_test, n_cols), dtype=int)
    all_baseline_results['always_down'] = evaluate_baseline(
        "Always-Down Baseline", pred, y_dir_test
    )

    # --- Baseline 2: Always-Up ---
    pred = np.full((n_test, n_cols), 2, dtype=int)
    all_baseline_results['always_up'] = evaluate_baseline(
        "Always-Up Baseline", pred, y_dir_test
    )

    # --- Baseline 3: Previous-Return Direction ---
    # Use the last feature's log_return sign as prediction
    num_assets = 3
    num_horizons = len(HORIZON_NAMES)
    pred = np.ones((n_test, n_cols), dtype=int)  # default neutral
    num_asset_feats = len(ASSET_FEATURE_COLS)
    log_return_idx = ASSET_FEATURE_COLS.index('log_return')

    for a_i in range(num_assets):
        feat_idx = a_i * num_asset_feats + log_return_idx
        prev_ret = X_test[:, feat_idx]
        for h_i in range(num_horizons):
            col = a_i * num_horizons + h_i
            pred[:, col] = np.where(prev_ret > 0.001, 2, np.where(prev_ret < -0.001, 0, 1))

    all_baseline_results['prev_return'] = evaluate_baseline(
        "Previous-Return Direction", pred, y_dir_test
    )

    # --- Baseline 4: Moving-Average Momentum ---
    # If close > 20-period SMA -> up, else -> down (approximated from features)
    pred = np.ones((n_test, n_cols), dtype=int)
    rsi_idx = ASSET_FEATURE_COLS.index('rsi')
    for a_i in range(num_assets):
        feat_idx = a_i * num_asset_feats + rsi_idx
        rsi_vals = X_test[:, feat_idx]
        for h_i in range(num_horizons):
            col = a_i * num_horizons + h_i
            # RSI > 0 (mean) suggests upward momentum in scaled space
            pred[:, col] = np.where(rsi_vals > 0, 2, np.where(rsi_vals < -0.5, 0, 1))

    all_baseline_results['momentum'] = evaluate_baseline(
        "Momentum (RSI-based) Baseline", pred, y_dir_test
    )

    # --- Baseline 5: Logistic Regression ---
    print("\nTraining Logistic Regression (per asset-horizon)...")
    pred = np.zeros((n_test, n_cols), dtype=int)
    probs = np.zeros((n_test, n_cols, 3))

    for col in range(n_cols):
        y_tr = y_dir_train[:, col].astype(int)
        y_va = y_dir_val[:, col].astype(int)

        # Combine train + val for final fit (still no test leakage)
        X_fit = np.vstack([X_train, X_val])
        y_fit = np.concatenate([y_tr, y_va])

        lr = LogisticRegression(max_iter=1000, C=0.1, solver='lbfgs', multi_class='multinomial')
        lr.fit(X_fit, y_fit)

        pred[:, col] = lr.predict(X_test)
        probs[:, col] = lr.predict_proba(X_test)

    all_baseline_results['logistic_regression'] = evaluate_baseline(
        "Logistic Regression", pred, y_dir_test, probs
    )

    # --- Baseline 6: XGBoost ---
    if HAS_XGBOOST:
        print("\nTraining XGBoost (per asset-horizon)...")
        pred = np.zeros((n_test, n_cols), dtype=int)
        probs = np.zeros((n_test, n_cols, 3))

        for col in range(n_cols):
            y_tr = y_dir_train[:, col].astype(int)
            y_va = y_dir_val[:, col].astype(int)

            xgb = XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                eval_metric='mlogloss',
                early_stopping_rounds=20,
                verbosity=0,
                use_label_encoder=False,
            )
            xgb.fit(
                X_train, y_tr,
                eval_set=[(X_val, y_va)],
                verbose=False,
            )

            pred[:, col] = xgb.predict(X_test)
            probs[:, col] = xgb.predict_proba(X_test)

        all_baseline_results['xgboost'] = evaluate_baseline(
            "XGBoost (Gradient-Boosted Trees)", pred, y_dir_test, probs
        )

    # --- Summary ---
    print("\n" + "=" * 60)
    print("  BASELINE COMPARISON SUMMARY (Test Set Balanced Accuracy)")
    print("=" * 60)

    for baseline_name, results in all_baseline_results.items():
        avg_bacc = np.mean([m['balanced_acc'] for m in results.values()])
        avg_mcc = np.mean([m['mcc'] for m in results.values()])
        avg_auc = np.mean([m.get('roc_auc', 0.5) for m in results.values()])
        print(f"  {baseline_name:25s} | Bal.Acc: {avg_bacc:.4f} | "
              f"MCC: {avg_mcc:+.4f} | AUC: {avg_auc:.4f}")

    # Save results
    serializable = {}
    for k, v in all_baseline_results.items():
        serializable[k] = {kk: {kkk: float(vvv) for kkk, vvv in vv.items()} for kk, vv in v.items()}

    with open('artifacts/baseline_results.json', 'w') as f:
        json.dump(serializable, f, indent=2)

    print("\nBaseline results saved to artifacts/baseline_results.json")
    return all_baseline_results


if __name__ == '__main__':
    run_baselines()
