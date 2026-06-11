"""
Step 4 — Evaluation & Reporting
----------------------------------
Metrics computed per model (val + test):
  MSE, RMSE, MAE, R², Adj-R²
Plus 5-fold CV: RMSE mean±std, R² mean±std

Outputs:
  outputs/results/metrics_table.csv
  outputs/figures/08_metrics_comparison.png
  outputs/figures/09_cv_robustness.png
  outputs/figures/10_pred_vs_actual_best.png
  outputs/figures/11_learning_curves.png   (LSTM + NN)
"""

import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULTS_DIR, FIGURES_DIR, MODELS_DIR

sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams.update({'figure.dpi': 120, 'font.size': 10})

MODEL_ORDER = [
    'MLR', 'GBR', 'XGBoost', 'LightGBM', 'CatBoost',
    'RandomForest', 'Stacking', 'Bagging', 'WeightedVoting',
    'LSTM', 'NN',
]


def _save(fig, name):
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {name}")


def adj_r2(r2, n, k):
    if n <= k + 1:
        return float('nan')
    return 1 - (1 - r2) * (n - 1) / (n - k - 1)


def metrics_row(y_true, y_pred, n_features, split_label, model_name) -> dict:
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    ar2  = adj_r2(r2, len(y_true), n_features)
    return {
        'Model': model_name, 'Split': split_label,
        'MSE':      round(mse,  4),
        'RMSE':     round(rmse, 4),
        'MAE':      round(mae,  4),
        'R2':       round(r2,   4),
        'Adj_R2':   round(ar2,  4),
    }


# ── build full metrics table ──────────────────────────────────────────────────

def build_table():
    bundle      = joblib.load(os.path.join(RESULTS_DIR, 'preprocessed.pkl'))
    predictions = joblib.load(os.path.join(RESULTS_DIR, 'predictions.pkl'))
    n_features  = bundle['n_features']

    rows = []
    for model_name in MODEL_ORDER:
        if model_name not in predictions:
            print(f"  WARNING: {model_name} not found in predictions — skipping.")
            continue
        p = predictions[model_name]
        rows.append(metrics_row(p['val_true'],  p['val_pred'],  n_features, 'Val',  model_name))
        rows.append(metrics_row(p['test_true'], p['test_pred'], n_features, 'Test', model_name))

    df_metrics = pd.DataFrame(rows)

    # Attach CV columns (from predictions dict)
    cv_rows = []
    for model_name in MODEL_ORDER:
        if model_name not in predictions:
            continue
        p = predictions[model_name]
        cv_rows.append({
            'Model':        model_name,
            'CV_RMSE_mean': round(p.get('cv_rmse_mean', float('nan')), 4),
            'CV_RMSE_std':  round(p.get('cv_rmse_std',  float('nan')), 4),
            'CV_R2_mean':   round(p.get('cv_r2_mean',   float('nan')), 4),
            'CV_R2_std':    round(p.get('cv_r2_std',    float('nan')), 4),
        })
    df_cv = pd.DataFrame(cv_rows)

    # Merge CV into main table
    df_all = df_metrics.merge(df_cv, on='Model', how='left')

    out_path = os.path.join(RESULTS_DIR, 'metrics_table.csv')
    df_all.to_csv(out_path, index=False)
    print(f"  Saved metrics table → {out_path}")

    # Pretty print
    pd.set_option('display.max_columns', 20)
    pd.set_option('display.width', 200)
    print("\n" + "=" * 100)
    print("METRICS TABLE")
    print("=" * 100)
    for split in ['Val', 'Test']:
        sub = df_all[df_all['Split'] == split].drop(columns='Split')
        print(f"\n── {split} Set ──")
        print(sub.to_string(index=False))
    print("\n── 5-Fold CV (on Training Set) ──")
    print(df_cv.to_string(index=False))
    print("=" * 100 + "\n")

    return df_all, df_cv


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_metrics_comparison(df_all):
    metrics = ['MSE', 'RMSE', 'MAE', 'R2', 'Adj_R2']
    splits  = ['Val', 'Test']
    # Lower is better for error metrics; higher is better for R² metrics
    lower_is_better = {'MSE', 'RMSE', 'MAE'}

    fig, axes = plt.subplots(len(metrics), 2, figsize=(16, 4 * len(metrics)))

    for row_i, metric in enumerate(metrics):
        for col_i, split in enumerate(splits):
            ax  = axes[row_i][col_i]
            sub = df_all[df_all['Split'] == split][['Model', metric]].dropna(subset=[metric])

            # Sort: best model at top of the horizontal bar chart
            ascending = metric in lower_is_better
            sub = sub.sort_values(metric, ascending=not ascending)  # worst at bottom, best at top after invert

            models = sub['Model'].values
            vals   = sub[metric].values

            bars = ax.barh(models, vals, color='#2980b9', edgecolor='white', alpha=0.85)
            ax.set_title(f'{metric} — {split}', fontsize=9, fontweight='bold')
            ax.set_xlabel(metric, fontsize=8)

            # Value annotations
            x_pad = (vals.max() - vals.min()) * 0.01 if vals.max() != vals.min() else vals.max() * 0.01
            for bar, v in zip(bars, vals):
                ax.text(bar.get_width() + x_pad,
                        bar.get_y() + bar.get_height() / 2,
                        f'{v:.4f}', va='center', fontsize=7)

            # Best model at top
            ax.invert_yaxis()

    fig.suptitle('Model Performance Comparison — Validation & Test Sets',
                 fontsize=13, fontweight='bold', y=1.01)
    fig.tight_layout()
    _save(fig, '08_metrics_comparison.png')


def plot_cv_robustness(df_cv):
    df_plot = df_cv.dropna(subset=['CV_RMSE_mean'])
    if df_plot.empty:
        print("  No CV data to plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    models = df_plot['Model'].values
    x = np.arange(len(models))

    axes[0].bar(x, df_plot['CV_RMSE_mean'], yerr=df_plot['CV_RMSE_std'],
                capsize=4, color='steelblue', edgecolor='white', alpha=0.85)
    axes[0].set_xticks(x); axes[0].set_xticklabels(models, rotation=30, ha='right', fontsize=8)
    axes[0].set_ylabel('RMSE'); axes[0].set_title('5-Fold CV — RMSE (mean ± std)')

    axes[1].bar(x, df_plot['CV_R2_mean'], yerr=df_plot['CV_R2_std'],
                capsize=4, color='mediumseagreen', edgecolor='white', alpha=0.85)
    axes[1].set_xticks(x); axes[1].set_xticklabels(models, rotation=30, ha='right', fontsize=8)
    axes[1].set_ylabel('R²'); axes[1].set_title('5-Fold CV — R² (mean ± std)')
    axes[1].set_ylim(max(0, df_plot['CV_R2_mean'].min() - 0.05), 1.01)

    fig.suptitle('Cross-Validation Robustness Check (TimeSeriesSplit, k=5)',
                 fontweight='bold')
    fig.tight_layout()
    _save(fig, '09_cv_robustness.png')


def plot_pred_vs_actual(predictions, df_all):
    """Scatter plot of predicted vs actual AQI for the best model (by test R²)."""
    test_rows = df_all[df_all['Split'] == 'Test']
    best_model = test_rows.loc[test_rows['R2'].idxmax(), 'Model']
    print(f"  Best model by Test R²: {best_model}")

    p = predictions[best_model]
    y_true = p['test_true']
    y_pred = p['test_pred']

    # Sample for plot clarity
    rng = np.random.default_rng(42)
    idx = rng.choice(len(y_true), size=min(5000, len(y_true)), replace=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].scatter(y_true[idx], y_pred[idx], alpha=0.25, s=8, color='steelblue')
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    axes[0].plot(lims, lims, 'r--', linewidth=1.5, label='Perfect fit')
    axes[0].set_xlabel('Actual AQI'); axes[0].set_ylabel('Predicted AQI')
    axes[0].set_title(f'{best_model} — Predicted vs Actual (Test)')
    axes[0].legend()

    residuals = y_true - y_pred
    axes[1].hist(residuals, bins=60, color='salmon', edgecolor='white', linewidth=0.4)
    axes[1].axvline(0, color='black', linestyle='--', linewidth=1)
    axes[1].set_xlabel('Residual (Actual − Predicted)'); axes[1].set_ylabel('Count')
    axes[1].set_title(f'{best_model} — Residual Distribution')

    fig.suptitle(f'Best Model: {best_model}', fontweight='bold')
    fig.tight_layout()
    _save(fig, '10_pred_vs_actual_best.png')


def plot_learning_curves():
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    for ax, name, pkl in [
        (axes[0], 'LSTM',       'lstm_history.pkl'),
        (axes[1], 'NN (MLP)',   'nn_history.pkl'),
    ]:
        path = os.path.join(RESULTS_DIR, pkl)
        if not os.path.exists(path):
            ax.set_title(f'{name} — history not found'); continue
        h = joblib.load(path)
        epochs = range(1, len(h['loss']) + 1)
        ax.plot(epochs, h['loss'],     label='Train loss', color='steelblue')
        ax.plot(epochs, h['val_loss'], label='Val loss',   color='orange', linestyle='--')
        ax.set_xlabel('Epoch'); ax.set_ylabel('MSE Loss')
        ax.set_title(f'{name} — Learning Curve')
        ax.legend()

    fig.suptitle('Deep Learning Learning Curves', fontweight='bold')
    fig.tight_layout()
    _save(fig, '11_learning_curves.png')


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=== EVALUATION ===")
    df_all, df_cv = build_table()
    predictions   = joblib.load(os.path.join(RESULTS_DIR, 'predictions.pkl'))

    plot_metrics_comparison(df_all)
    plot_cv_robustness(df_cv)
    plot_pred_vs_actual(predictions, df_all)
    plot_learning_curves()

    print("Evaluation complete.\n")


if __name__ == '__main__':
    main()
