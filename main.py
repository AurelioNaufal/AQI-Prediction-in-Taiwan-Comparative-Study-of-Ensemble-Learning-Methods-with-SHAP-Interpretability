"""
main.py — Orchestrates all pipeline steps in order.

Usage (from project root, with .venv activated):
    python main.py [--skip-eda] [--skip-models] [--skip-eval] [--skip-shap]

Steps:
    1. Preprocessing   → outputs/results/preprocessed.pkl
    2. EDA             → outputs/figures/01-07_*.png
    3. Model Training  → outputs/models/*.pkl  +  outputs/results/predictions.pkl
    4. Evaluation      → outputs/results/metrics_table.csv  +  figures 08-11
    5. SHAP Analysis   → outputs/figures/12-15_*.png
"""

import argparse
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def banner(step: str):
    print("\n" + "=" * 60)
    print(f"  {step}")
    print("=" * 60)


def _run_module(alias, filename):
    import importlib.util
    spec = importlib.util.spec_from_file_location(alias, filename)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main()


def step1_preprocessing(): banner("STEP 1 — Preprocessing"); _run_module("pre",    "01_preprocessing.py")
def step2_eda():            banner("STEP 2 — EDA");          _run_module("eda",    "02_eda.py")
def step3_models():         banner("STEP 3 — Model Training"); _run_module("models", "03_models.py")
def step4_evaluation():     banner("STEP 4 — Evaluation");    _run_module("eval",   "04_evaluation.py")
def step5_shap():           banner("STEP 5 — SHAP Analysis"); _run_module("shap_a", "05_shap_analysis.py")


def main():
    parser = argparse.ArgumentParser(description='AQI Prediction Pipeline')
    parser.add_argument('--skip-eda',    action='store_true', help='Skip EDA step')
    parser.add_argument('--skip-models', action='store_true', help='Skip model training')
    parser.add_argument('--skip-eval',   action='store_true', help='Skip evaluation')
    parser.add_argument('--skip-shap',   action='store_true', help='Skip SHAP analysis')
    args = parser.parse_args()

    t_start = time.time()

    step1_preprocessing()

    if not args.skip_eda:
        step2_eda()

    if not args.skip_models:
        step3_models()

    if not args.skip_eval:
        step4_evaluation()

    if not args.skip_shap:
        step5_shap()

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  Pipeline complete in {elapsed/60:.1f} min.")
    print(f"  Results → outputs/")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
