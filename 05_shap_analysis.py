"""
Step 5 — SHAP Interpretability
---------------------------------
Runs SHAP on:
  • Best model by Test R² (usually GBR / WeightedVoting)
  • WeightedVoting ensemble (always, as reference)

Plots saved to outputs/figures/:
  12_shap_summary_best.png        — beeswarm
  13_shap_bar_best.png            — mean |SHAP| feature importance
  14_shap_dependence_top3.png     — dependence plots for top-3 features
  15_shap_summary_voting.png      — beeswarm for WeightedVoting
"""

import os
import sys
import warnings
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULTS_DIR, FIGURES_DIR, MODELS_DIR

sns.set_theme(style='whitegrid')
plt.rcParams.update({'figure.dpi': 120, 'font.size': 10})

SHAP_SAMPLE = 3000   # number of test rows used for SHAP (speed vs coverage)


def _save(fig, name):
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {name}")


def _get_best_model_name(df_all):
    test_rows = df_all[df_all['Split'] == 'Test']
    return test_rows.loc[test_rows['R2'].idxmax(), 'Model']


def _load_tabular_model(name: str):
    """Load a sklearn-compatible model from pkl."""
    path = os.path.join(MODELS_DIR, f'{name}.pkl')
    if not os.path.exists(path):
        return None
    return joblib.load(path)


# Mapping from result-dict name → pkl filename
MODEL_PKL_MAP = {
    'MLR':           'mlr',
    'GBR':           'gbr',
    'XGBoost':       'xgb',
    'LightGBM':      'lgb',
    'CatBoost':      'cb',
    'RandomForest':  'rf',
    'Stacking':      'stacking',
    'Bagging':       'bagging',
    'WeightedVoting':'voting',
}


def _build_explainer(model, X_bg, model_name: str):
    """Return a SHAP explainer appropriate for the model type."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        # Tree-based models (single tree or ensemble of trees)
        if model_name in ('GBR', 'XGBoost', 'LightGBM', 'CatBoost', 'RandomForest', 'Bagging'):
            try:
                return shap.TreeExplainer(model)
            except Exception:
                pass
        # Stacking / VotingRegressor → fall back to Explainer with masker
        return shap.Explainer(model.predict, shap.maskers.Independent(X_bg, max_samples=500))


def run_shap_for(model_name: str, X_test, feature_names, tag: str):
    """Compute SHAP values and produce three plots for a given model."""
    model = _load_tabular_model(MODEL_PKL_MAP.get(model_name, model_name.lower()))
    if model is None:
        print(f"  Model file not found for {model_name} — skipping SHAP.")
        return

    rng  = np.random.default_rng(42)
    idx  = rng.choice(len(X_test), size=min(SHAP_SAMPLE, len(X_test)), replace=False)
    X_bg = X_test[idx]

    print(f"  Building explainer for {model_name} …")
    explainer  = _build_explainer(model, X_bg, model_name)

    print(f"  Computing SHAP values ({X_bg.shape[0]} samples) …")
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        shap_values = explainer(X_bg)

    vals       = shap_values.values          # (n_samples, n_features)
    feat_names = np.array(feature_names)

    # ── beeswarm summary ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        vals, X_bg,
        feature_names=feat_names,
        show=False, plot_size=None,
        max_display=20,
    )
    plt.title(f'SHAP Summary (Beeswarm) — {model_name}', fontweight='bold', pad=10)
    fig = plt.gcf()
    _save(fig, f'12_shap_summary_{tag}.png')

    # ── bar importance ────────────────────────────────────────────────────────
    mean_abs = np.abs(vals).mean(axis=0)
    order    = np.argsort(mean_abs)[::-1][:20]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(feat_names[order][::-1], mean_abs[order][::-1],
            color='steelblue', edgecolor='white', alpha=0.85)
    ax.set_xlabel('Mean |SHAP value|')
    ax.set_title(f'Feature Importance (Mean |SHAP|) — {model_name}', fontweight='bold')
    fig.tight_layout()
    _save(fig, f'13_shap_bar_{tag}.png')

    # ── dependence plots: top 3 features ─────────────────────────────────────
    top3 = feat_names[order[:3]]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, feat in zip(axes, top3):
        feat_idx = list(feat_names).index(feat)
        shap.dependence_plot(
            feat_idx, vals, X_bg,
            feature_names=feat_names,
            ax=ax, show=False,
            alpha=0.4,
        )
        ax.set_title(f'{feat}', fontsize=9, fontweight='bold')
    fig.suptitle(f'SHAP Dependence Plots — Top-3 Features ({model_name})',
                 fontweight='bold', y=1.02)
    fig.tight_layout()
    _save(fig, f'14_shap_dependence_{tag}.png')

    return order, feat_names, mean_abs


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=== SHAP ANALYSIS ===")

    # Load metrics table to find best model
    metrics_path = os.path.join(RESULTS_DIR, 'metrics_table.csv')
    if not os.path.exists(metrics_path):
        print("  metrics_table.csv not found — run 04_evaluation.py first.")
        return

    df_all = pd.read_csv(metrics_path)
    bundle = joblib.load(os.path.join(RESULTS_DIR, 'preprocessed.pkl'))

    X_test       = bundle['X_test']
    feature_names = bundle['feature_names']

    best_name = _get_best_model_name(df_all)
    print(f"  Best model: {best_name}")

    # SHAP for best model
    run_shap_for(best_name, X_test, feature_names, tag='best')

    # SHAP for WeightedVoting (always, as the paper's proposed ensemble)
    if best_name != 'WeightedVoting':
        print("\n  Also running SHAP for WeightedVoting ensemble …")
        run_shap_for('WeightedVoting', X_test, feature_names, tag='voting')

    # ── comparative feature importance (all tree models, bar chart) ───────────
    print("\n  Building comparative feature importance chart …")
    tree_models = ['GBR', 'XGBoost', 'LightGBM', 'CatBoost', 'RandomForest']
    imp_dict = {}
    for mn in tree_models:
        m = _load_tabular_model(MODEL_PKL_MAP[mn])
        if m is None:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                exp  = shap.TreeExplainer(m)
                rng  = np.random.default_rng(42)
                idx  = rng.choice(len(X_test), size=min(SHAP_SAMPLE, len(X_test)), replace=False)
                sv   = exp.shap_values(X_test[idx])
            imp_dict[mn] = np.abs(sv).mean(axis=0)
        except Exception as e:
            print(f"    {mn} SHAP failed: {e}")

    if imp_dict:
        df_imp = pd.DataFrame(imp_dict, index=feature_names)
        top_feats = df_imp.mean(axis=1).nlargest(15).index

        fig, ax = plt.subplots(figsize=(12, 7))
        df_imp.loc[top_feats].plot.barh(ax=ax, width=0.75, alpha=0.85)
        ax.set_xlabel('Mean |SHAP value|')
        ax.set_title('Comparative Feature Importance Across Tree Models (Top 15)', fontweight='bold')
        ax.invert_yaxis()
        ax.legend(title='Model', loc='lower right')
        fig.tight_layout()
        _save(fig, '15_shap_comparative.png')

    print("SHAP analysis complete.\n")


if __name__ == '__main__':
    main()
