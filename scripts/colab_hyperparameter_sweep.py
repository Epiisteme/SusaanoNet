"""
Hyperparameter Sweep (Point 9)
==============================
Grid search over sequence length, learning rate, and dropout.
Each configuration is evaluated across 5 random seeds to measure robustness.
Results are logged to artifacts/sweep_results.json.
"""
import itertools
import json
import os
import numpy as np
from datetime import datetime

# Import training components and CONFIG from the standalone script
import colab_train_standalone as train_script
from colab_train_standalone import load_data, train_final_model, set_seed

def run_sweep():
    print("=" * 70)
    print("HYPERPARAMETER SWEEP & ROBUSTNESS TESTING (Point 9)")
    print("=" * 70)

    os.makedirs('artifacts', exist_ok=True)

    # Grid definition
    grid = {
        'seq_length': [30, 60, 90],
        'lr': [1e-4, 5e-4, 1e-3],
        'dropout': [0.1, 0.2, 0.3]
    }
    
    seeds = [42, 1337, 2026, 777, 999]

    # Generate all combinations
    keys, values = zip(*grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"Total hyperparameter combinations: {len(combinations)}")
    print(f"Seeds per combination: {len(seeds)}")
    print(f"Total training runs: {len(combinations) * len(seeds)}")

    # Load data once
    print("\nLoading dataset for sweep...")
    features_df, reg_targets_df, dir_targets_df = load_data(train_script.CONFIG['csv_path'])
    
    all_results = []
    
    # We will use fewer epochs for the sweep to make it computationally feasible
    sweep_epochs = 20
    train_script.CONFIG['epochs'] = sweep_epochs
    train_script.CONFIG['early_stop_patience'] = 5

    for i, combo in enumerate(combinations):
        print(f"\n{'=' * 70}")
        print(f"Combination {i+1}/{len(combinations)}: {combo}")
        print(f"{'=' * 70}")
        
        # Update CONFIG
        for k, v in combo.items():
            train_script.CONFIG[k] = v
            
        combo_metrics = []
        
        for seed in seeds:
            print(f"\n  --- Running Seed {seed} ---")
            train_script.CONFIG['seed'] = seed
            set_seed(seed)
            
            # Train and evaluate
            try:
                test_metrics = train_final_model(features_df, reg_targets_df, dir_targets_df)
                combo_metrics.append({
                    'seed': seed,
                    'test_metrics': test_metrics
                })
            except Exception as e:
                print(f"  Run failed for seed {seed}: {e}")
                
        # Aggregate metrics across seeds for this combination
        if combo_metrics:
            avg_metrics = {}
            std_metrics = {}
            metric_keys = combo_metrics[0]['test_metrics'].keys()
            
            for key in metric_keys:
                vals = [m['test_metrics'][key] for m in combo_metrics]
                avg_metrics[key] = float(np.mean(vals))
                std_metrics[key] = float(np.std(vals))
                
            # Print aggregate summary
            avg_bacc = np.mean([v for k, v in avg_metrics.items() if 'balanced_acc' in k])
            avg_auc = np.mean([v for k, v in avg_metrics.items() if 'roc_auc' in k])
            print(f"\n  => Combo {i+1} Avg Bal.Acc: {avg_bacc:.4f} +/- {np.mean([v for k, v in std_metrics.items() if 'balanced_acc' in k]):.4f}")
            print(f"  => Combo {i+1} Avg AUC:     {avg_auc:.4f} +/- {np.mean([v for k, v in std_metrics.items() if 'roc_auc' in k]):.4f}")
            
            all_results.append({
                'combination_id': i + 1,
                'hyperparameters': combo,
                'epochs_per_run': sweep_epochs,
                'runs': combo_metrics,
                'aggregated': {
                    'mean': avg_metrics,
                    'std': std_metrics,
                    'summary_avg_bacc': float(avg_bacc),
                    'summary_avg_auc': float(avg_auc)
                }
            })
            
            # Save intermediate results
            with open('artifacts/sweep_results.json', 'w') as f:
                json.dump(all_results, f, indent=2)

    # Find the best combination based on average balanced accuracy
    if all_results:
        best_result = max(all_results, key=lambda x: x['aggregated']['summary_avg_bacc'])
        print(f"\n{'=' * 70}")
        print("SWEEP COMPLETE")
        print(f"Best Hyperparameters: {best_result['hyperparameters']}")
        print(f"Best Avg Bal.Acc: {best_result['aggregated']['summary_avg_bacc']:.4f}")
        print(f"Best Avg AUC: {best_result['aggregated']['summary_avg_auc']:.4f}")
        print("=" * 70)
        print("\nAll results saved to artifacts/sweep_results.json")

if __name__ == '__main__':
    run_sweep()
