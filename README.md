# Taiwan AQI Prediction — MLDA Final Project

Hourly Air Quality Index (AQI) regression for Taiwan (Jan–Aug 2024) using 11 machine learning models.  
Pollutant sensors (SO₂, CO, O₃, PM₂.₅, PM₁₀, NO₂, NOₓ, NO, wind speed/direction) + dominant pollutant category → AQI.

## Models

| Type | Models |
|------|--------|
| Baseline | Multiple Linear Regression (MLR) |
| Tree-based | Gradient Boosting (GBR), XGBoost, LightGBM, CatBoost, Random Forest |
| Ensemble | Stacking, Bagging, Weighted Voting |
| Deep Learning | LSTM, Neural Network (MLP) |

Hyperparameters for GBR/XGB/LGB/CatBoost/RF tuned with **Optuna** (30 trials, TPE sampler).  
Evaluation uses a strict **temporal 70 / 15 / 15 train-val-test split** (no shuffling) to prevent leakage.

## Project Structure

```
.
├── config.py                        # Paths, constants, feature lists
├── main.py                          # Pipeline orchestrator
├── 01_preprocessing.py              # Data cleaning, OHE, scaling, split
├── 02_eda.py                        # 7 EDA figures incl. Taiwan choropleth
├── 03_models.py                     # Train all 11 models + CV
├── 04_evaluation.py                 # Metrics table + comparison plots
├── 05_shap_analysis.py              # SHAP analysis for tree models
├── requirements.txt
├── air_quality_2024.csv             # Dataset (2024, Jan–Aug)
├── Taiwanmap_shp/                   # Taiwan county shapefile
│   └── tw.shp (+ .dbf, .shx, ...)
├── outputs/
│   ├── figures/                     # All generated plots (01–15)
│   ├── models/                      # Saved model files (.pkl / .keras)
│   └── results/                     # predictions.pkl, metrics_table.csv
└── MLDA_FinalProject_Code.ipynb     # Self-contained notebook (all steps)
```

## Setup

**Python 3.10+** required. TensorFlow 2.15 does not support Python 3.12.

```bash
# Clone / download the repo, then:
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

> **Note:** `geopandas` on Windows sometimes needs the binary wheel.  
> If the pip install fails, try: `pip install geopandas --find-links https://girder.github.io/large_image_wheels`

## Running the Pipeline

Place `air_quality_2024.csv` and the `Taiwanmap_shp/` folder in the project root, then:

```bash
# Full pipeline (~45–60 min, mostly model training + Optuna)
python main.py

# Skip optional steps to save time
python main.py --skip-eda
python main.py --skip-models --skip-shap

# Or run individual steps
python 01_preprocessing.py
python 02_eda.py
python 03_models.py
python 04_evaluation.py
python 05_shap_analysis.py
```

Steps must run **in order** (each step depends on outputs from the previous one).

### Expected Runtime

| Step | Approx. time |
|------|-------------|
| Preprocessing | < 1 min |
| EDA | 1–2 min |
| Model training (with Optuna) | 40–55 min |
| Evaluation | < 1 min |
| SHAP analysis | 2–5 min |

## Notebook

`MLDA_FinalProject_Code.ipynb` contains all five steps in a single notebook with explanations.  
Open in JupyterLab or VS Code and run cells top to bottom (same prerequisites apply).

## Key Results

- Best model: **CatBoost** (R² ≈ 0.9996, RMSE ≈ 0.54)
- Near-perfect R² is expected — AQI is a deterministic piecewise-linear function of the same-hour inputs the model receives. ML adds value through missing sensor imputation, adapting to Taiwan EPA's actual breakpoints, and SHAP-based feature importance.
- Top features (SHAP): `pm2.5`, `pm2.5_avg`, `o3_8hr`, `pol_PM25`

## Data

Source: Taiwan Environmental Protection Administration (EPA) open data.  
Subset: 2024 Jan–Aug, hourly station readings across all counties.  
The `county`, `latitude`, `longitude` columns are used only for the choropleth map and are excluded from model features.
