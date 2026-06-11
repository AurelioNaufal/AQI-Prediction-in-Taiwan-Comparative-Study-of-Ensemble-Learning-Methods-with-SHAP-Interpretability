"""
Step 3 — Model Training
--------------------------
Models (11 total):
  Tabular  : MLR, GBR, XGBoost, LightGBM, CatBoost, RandomForest,
             Stacking, Bagging, WeightedVoting (Boosting Ensemble)
  Deep     : LSTM, NN (MLP)

Tuning     : Optuna (30 trials, 25% subsample) for GBR/XGB/LGB/CB/RF
CV         : 5-fold TimeSeriesSplit on training set (tabular models only)
Outputs    : outputs/models/<model>.pkl  |  outputs/results/predictions.pkl
"""

import os
import sys
import time
import warnings
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import joblib
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    RESULTS_DIR, MODELS_DIR,
    RANDOM_SEED, OPTUNA_TRIALS, OPTUNA_TIMEOUT, OPTUNA_SUBSAMPLE,
    LOOKBACK, LSTM_EPOCHS, LSTM_BATCH_SIZE, NN_EPOCHS, NN_BATCH_SIZE,
)

# ── utilities ─────────────────────────────────────────────────────────────────

def adj_r2(r2: float, n: int, k: int) -> float:
    if n <= k + 1:
        return float('nan')
    return 1 - (1 - r2) * (n - 1) / (n - k - 1)


def compute_metrics(y_true, y_pred, n_features: int) -> dict:
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    ar2  = adj_r2(r2, len(y_true), n_features)
    return dict(mse=mse, rmse=rmse, mae=mae, r2=r2, adj_r2=ar2)


def run_cv(model, X_train, y_train, n_features: int) -> dict:
    """5-fold TimeSeriesSplit cross-validation on training data."""
    from sklearn.model_selection import TimeSeriesSplit, cross_val_score
    tscv = TimeSeriesSplit(n_splits=5)
    rmse_scores = -cross_val_score(model, X_train, y_train,
                                   cv=tscv, scoring='neg_root_mean_squared_error',
                                   n_jobs=-1)
    r2_scores   =  cross_val_score(model, X_train, y_train,
                                   cv=tscv, scoring='r2',
                                   n_jobs=-1)
    return dict(
        cv_rmse_mean=rmse_scores.mean(),
        cv_rmse_std=rmse_scores.std(),
        cv_r2_mean=r2_scores.mean(),
        cv_r2_std=r2_scores.std(),
    )


def subsample(X, y, frac=OPTUNA_SUBSAMPLE, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=int(len(X) * frac), replace=False)
    idx.sort()
    return X[idx], y[idx]


def _save_model(model, name: str):
    path = os.path.join(MODELS_DIR, f'{name}.pkl')
    joblib.dump(model, path)


def _load_data():
    bundle = joblib.load(os.path.join(RESULTS_DIR, 'preprocessed.pkl'))
    return (
        bundle['X_train'], bundle['y_train'],
        bundle['X_val'],   bundle['y_val'],
        bundle['X_test'],  bundle['y_test'],
        bundle['train_df'], bundle['val_df'], bundle['test_df'],
        bundle['feature_names'], bundle['n_features'],
    )


# ── 1. Multiple Linear Regression (baseline) ──────────────────────────────────

def train_mlr(X_train, y_train, X_val, X_test, y_train_r, y_val, y_test, n_features):
    print("\n[1/11] Multiple Linear Regression …")
    from sklearn.linear_model import LinearRegression
    model = LinearRegression(n_jobs=-1)
    model.fit(X_train, y_train_r)

    val_pred  = model.predict(X_val)
    test_pred = model.predict(X_test)
    cv_res    = run_cv(LinearRegression(n_jobs=-1), X_train, y_train_r, n_features)
    _save_model(model, 'mlr')
    return model, val_pred, test_pred, cv_res


# ── 2. Gradient Boosting (sklearn GBR) ───────────────────────────────────────

def tune_gbr(X_sub, y_sub, X_val, y_val):
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import mean_squared_error

    def objective(trial):
        params = dict(
            n_estimators  = trial.suggest_int('n_estimators', 50, 300),
            learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            max_depth     = trial.suggest_int('max_depth', 3, 7),
            min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20),
            subsample     = trial.suggest_float('subsample', 0.5, 1.0),
            random_state  = RANDOM_SEED,
        )
        m = GradientBoostingRegressor(**params)
        m.fit(X_sub, y_sub)
        return mean_squared_error(y_val, m.predict(X_val))

    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    study.optimize(objective, n_trials=OPTUNA_TRIALS, timeout=OPTUNA_TIMEOUT, show_progress_bar=False)
    return study.best_params


def train_gbr(X_train, y_train, X_val, X_test, y_val, y_test, n_features):
    print("\n[2/11] Gradient Boosting Regressor (Optuna) …")
    from sklearn.ensemble import GradientBoostingRegressor
    X_sub, y_sub = subsample(X_train, y_train)
    best = tune_gbr(X_sub, y_sub, X_val, y_val)
    best['random_state'] = RANDOM_SEED
    print(f"  Best params: {best}")
    model = GradientBoostingRegressor(**best)
    model.fit(X_train, y_train)
    val_pred  = model.predict(X_val)
    test_pred = model.predict(X_test)
    cv_res = run_cv(GradientBoostingRegressor(**best), X_train, y_train, n_features)
    _save_model(model, 'gbr')
    return model, val_pred, test_pred, cv_res, best


# ── 3. XGBoost ────────────────────────────────────────────────────────────────

def tune_xgb(X_sub, y_sub, X_val, y_val):
    import xgboost as xgb
    from sklearn.metrics import mean_squared_error

    def objective(trial):
        params = dict(
            n_estimators      = trial.suggest_int('n_estimators', 50, 300),
            learning_rate     = trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            max_depth         = trial.suggest_int('max_depth', 3, 8),
            subsample         = trial.suggest_float('subsample', 0.5, 1.0),
            colsample_bytree  = trial.suggest_float('colsample_bytree', 0.5, 1.0),
            reg_alpha         = trial.suggest_float('reg_alpha', 1e-4, 1.0, log=True),
            reg_lambda        = trial.suggest_float('reg_lambda', 1e-4, 1.0, log=True),
            random_state      = RANDOM_SEED,
            verbosity         = 0,
            n_jobs            = -1,
        )
        m = xgb.XGBRegressor(**params)
        m.fit(X_sub, y_sub, eval_set=[(X_val, y_val)],
              verbose=False)
        return mean_squared_error(y_val, m.predict(X_val))

    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    study.optimize(objective, n_trials=OPTUNA_TRIALS, timeout=OPTUNA_TIMEOUT, show_progress_bar=False)
    return study.best_params


def train_xgb(X_train, y_train, X_val, X_test, y_val, y_test, n_features):
    print("\n[3/11] XGBoost (Optuna) …")
    import xgboost as xgb
    X_sub, y_sub = subsample(X_train, y_train)
    best = tune_xgb(X_sub, y_sub, X_val, y_val)
    best.update({'random_state': RANDOM_SEED, 'verbosity': 0, 'n_jobs': -1})
    print(f"  Best params: {best}")
    model = xgb.XGBRegressor(**best)
    model.fit(X_train, y_train)
    val_pred  = model.predict(X_val)
    test_pred = model.predict(X_test)
    cv_res = run_cv(xgb.XGBRegressor(**best), X_train, y_train, n_features)
    _save_model(model, 'xgb')
    return model, val_pred, test_pred, cv_res, best


# ── 4. LightGBM ───────────────────────────────────────────────────────────────

def tune_lgb(X_sub, y_sub, X_val, y_val):
    import lightgbm as lgb
    from sklearn.metrics import mean_squared_error

    def objective(trial):
        params = dict(
            n_estimators     = trial.suggest_int('n_estimators', 50, 400),
            learning_rate    = trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            max_depth        = trial.suggest_int('max_depth', 3, 8),
            num_leaves       = trial.suggest_int('num_leaves', 15, 127),
            min_child_samples= trial.suggest_int('min_child_samples', 5, 50),
            subsample        = trial.suggest_float('subsample', 0.5, 1.0),
            colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0),
            random_state     = RANDOM_SEED,
            n_jobs           = -1,
            verbose          = -1,
        )
        m = lgb.LGBMRegressor(**params)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m.fit(X_sub, y_sub,
                  eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(-1)])
        return mean_squared_error(y_val, m.predict(X_val))

    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    study.optimize(objective, n_trials=OPTUNA_TRIALS, timeout=OPTUNA_TIMEOUT, show_progress_bar=False)
    return study.best_params


def train_lgb(X_train, y_train, X_val, X_test, y_val, y_test, n_features):
    print("\n[4/11] LightGBM (Optuna) …")
    import lightgbm as lgb
    X_sub, y_sub = subsample(X_train, y_train)
    best = tune_lgb(X_sub, y_sub, X_val, y_val)
    best.update({'random_state': RANDOM_SEED, 'n_jobs': -1, 'verbose': -1})
    print(f"  Best params: {best}")
    model = lgb.LGBMRegressor(**best)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model.fit(X_train, y_train)
    val_pred  = model.predict(X_val)
    test_pred = model.predict(X_test)
    cv_res = run_cv(lgb.LGBMRegressor(**best), X_train, y_train, n_features)
    _save_model(model, 'lgb')
    return model, val_pred, test_pred, cv_res, best


# ── 5. CatBoost ───────────────────────────────────────────────────────────────

def tune_cb(X_sub, y_sub, X_val, y_val):
    from catboost import CatBoostRegressor
    from sklearn.metrics import mean_squared_error

    def objective(trial):
        params = dict(
            iterations    = trial.suggest_int('iterations', 50, 300),
            learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            depth         = trial.suggest_int('depth', 3, 8),
            l2_leaf_reg   = trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
            random_seed   = RANDOM_SEED,
            silent        = True,
        )
        m = CatBoostRegressor(**params)
        m.fit(X_sub, y_sub, eval_set=(X_val, y_val), use_best_model=True)
        return mean_squared_error(y_val, m.predict(X_val))

    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    study.optimize(objective, n_trials=OPTUNA_TRIALS, timeout=OPTUNA_TIMEOUT, show_progress_bar=False)
    return study.best_params


def train_cb(X_train, y_train, X_val, X_test, y_val, y_test, n_features):
    print("\n[5/11] CatBoost (Optuna) …")
    from catboost import CatBoostRegressor
    X_sub, y_sub = subsample(X_train, y_train)
    best = tune_cb(X_sub, y_sub, X_val, y_val)
    best.update({'random_seed': RANDOM_SEED, 'silent': True})
    print(f"  Best params: {best}")
    model = CatBoostRegressor(**best)
    model.fit(X_train, y_train)
    val_pred  = model.predict(X_val)
    test_pred = model.predict(X_test)
    cv_res = run_cv(CatBoostRegressor(**best), X_train, y_train, n_features)
    _save_model(model, 'cb')
    return model, val_pred, test_pred, cv_res, best


# ── 6. Random Forest ──────────────────────────────────────────────────────────

def tune_rf(X_sub, y_sub, X_val, y_val):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error

    def objective(trial):
        params = dict(
            n_estimators     = trial.suggest_int('n_estimators', 50, 300),
            max_depth        = trial.suggest_int('max_depth', 4, 20),
            min_samples_split= trial.suggest_int('min_samples_split', 2, 20),
            min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10),
            max_features     = trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.5]),
            random_state     = RANDOM_SEED,
            n_jobs           = -1,
        )
        m = RandomForestRegressor(**params)
        m.fit(X_sub, y_sub)
        return mean_squared_error(y_val, m.predict(X_val))

    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    study.optimize(objective, n_trials=OPTUNA_TRIALS, timeout=OPTUNA_TIMEOUT, show_progress_bar=False)
    return study.best_params


def train_rf(X_train, y_train, X_val, X_test, y_val, y_test, n_features):
    print("\n[6/11] Random Forest (Optuna) …")
    from sklearn.ensemble import RandomForestRegressor
    X_sub, y_sub = subsample(X_train, y_train)
    best = tune_rf(X_sub, y_sub, X_val, y_val)
    best.update({'random_state': RANDOM_SEED, 'n_jobs': -1})
    print(f"  Best params: {best}")
    model = RandomForestRegressor(**best)
    model.fit(X_train, y_train)
    val_pred  = model.predict(X_val)
    test_pred = model.predict(X_test)
    cv_res = run_cv(RandomForestRegressor(**best), X_train, y_train, n_features)
    _save_model(model, 'rf')
    return model, val_pred, test_pred, cv_res, best


# ── 7. Stacking Ensemble ──────────────────────────────────────────────────────

def train_stacking(gbr, xgb_m, lgb_m, cb_m,
                   X_train, y_train, X_val, X_test, y_val, y_test, n_features):
    print("\n[7/11] Stacking Ensemble (GBR+XGB+LGB+CB → Ridge) …")
    from sklearn.ensemble import StackingRegressor
    from sklearn.linear_model import Ridge

    estimators = [('gbr', gbr), ('xgb', xgb_m), ('lgb', lgb_m), ('cb', cb_m)]
    # StackingRegressor uses cross_val_predict internally to build meta-features,
    # which requires every sample to appear in exactly one test fold.
    # TimeSeriesSplit violates this (early samples never appear in test folds),
    # so we use KFold (cv=5) here instead.
    model = StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(),
        cv=5,          # KFold — required for cross_val_predict partition constraint
        n_jobs=-1,
        passthrough=False,
    )
    model.fit(X_train, y_train)
    val_pred  = model.predict(X_val)
    test_pred = model.predict(X_test)
    # Nested CV: 5 outer folds × 5 inner folds × 4 base models = 100 fits — slow but correct.
    print("  Running nested 5-fold CV for Stacking (this will take a while) ...")
    cv_res = run_cv(
        StackingRegressor(estimators=estimators, final_estimator=Ridge(), cv=5, n_jobs=-1),
        X_train, y_train, n_features
    )
    _save_model(model, 'stacking')
    return model, val_pred, test_pred, cv_res


# ── 8. Bagging Ensemble ───────────────────────────────────────────────────────

def train_bagging(X_train, y_train, X_val, X_test, y_val, y_test, n_features):
    print("\n[8/11] Bagging Ensemble …")
    from sklearn.ensemble import BaggingRegressor
    from sklearn.tree import DecisionTreeRegressor

    model = BaggingRegressor(
        estimator=DecisionTreeRegressor(max_depth=12, random_state=RANDOM_SEED),
        n_estimators=100,
        max_samples=0.8,
        max_features=0.8,
        bootstrap=True,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    val_pred  = model.predict(X_val)
    test_pred = model.predict(X_test)
    cv_res = run_cv(
        BaggingRegressor(
            estimator=DecisionTreeRegressor(max_depth=12, random_state=RANDOM_SEED),
            n_estimators=50, max_samples=0.8, max_features=0.8,
            bootstrap=True, random_state=RANDOM_SEED, n_jobs=-1,
        ),
        X_train, y_train, n_features
    )
    _save_model(model, 'bagging')
    return model, val_pred, test_pred, cv_res


# ── 9. Weighted Voting (Boosting Ensemble) ────────────────────────────────────

def train_voting(gbr, xgb_m, lgb_m, cb_m,
                 X_train, y_train, X_val, X_test, y_val, y_test, n_features):
    print("\n[9/11] Weighted Voting Ensemble (GBR×4 + CB×3 + XGB×2 + LGB×1) …")
    from sklearn.ensemble import VotingRegressor

    model = VotingRegressor(
        estimators=[('gbr', gbr), ('cb', cb_m), ('xgb', xgb_m), ('lgb', lgb_m)],
        weights=[4, 3, 2, 1],
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    val_pred  = model.predict(X_val)
    test_pred = model.predict(X_test)
    cv_res = run_cv(
        VotingRegressor(
            estimators=[('gbr', gbr), ('cb', cb_m), ('xgb', xgb_m), ('lgb', lgb_m)],
            weights=[4, 3, 2, 1], n_jobs=-1,
        ),
        X_train, y_train, n_features
    )
    _save_model(model, 'voting')
    return model, val_pred, test_pred, cv_res


# ── 10. LSTM ──────────────────────────────────────────────────────────────────

def _make_sequences(df_split, feature_cols, target_col, lookback):
    """Build LSTM input sequences per station, then pool together."""
    Xs, ys = [], []
    for _, grp in df_split.groupby('siteid', sort=False):
        grp = grp.sort_values('date')
        X = grp[feature_cols].values.astype(np.float32)
        y = grp[target_col].values.astype(np.float32)
        for i in range(lookback, len(grp)):
            Xs.append(X[i - lookback:i])
            ys.append(y[i])
    if not Xs:
        return np.empty((0, lookback, len(feature_cols))), np.empty(0)
    return np.stack(Xs), np.array(ys)


def train_lstm(train_df, val_df, test_df, feature_cols, n_features):
    print("\n[10/11] LSTM …")
    import tensorflow as tf
    tf.random.set_seed(RANDOM_SEED)

    from config import TARGET
    X_tr, y_tr = _make_sequences(train_df, feature_cols, TARGET, LOOKBACK)
    X_vl, y_vl = _make_sequences(val_df,   feature_cols, TARGET, LOOKBACK)
    X_ts, y_ts = _make_sequences(test_df,  feature_cols, TARGET, LOOKBACK)
    print(f"  Sequences — train: {X_tr.shape}, val: {X_vl.shape}, test: {X_ts.shape}")

    inp = tf.keras.Input(shape=(LOOKBACK, n_features))
    x   = tf.keras.layers.LSTM(64, return_sequences=True)(inp)
    x   = tf.keras.layers.Dropout(0.2)(x)
    x   = tf.keras.layers.LSTM(32)(x)
    x   = tf.keras.layers.Dropout(0.2)(x)
    x   = tf.keras.layers.Dense(16, activation='relu')(x)
    out = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mse')

    cb_es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    cb_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5, verbose=0)

    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_vl, y_vl),
        epochs=LSTM_EPOCHS,
        batch_size=LSTM_BATCH_SIZE,
        callbacks=[cb_es, cb_lr],
        verbose=1,
    )

    val_pred  = model.predict(X_vl, verbose=0).flatten()
    test_pred = model.predict(X_ts, verbose=0).flatten()

    model_path = os.path.join(MODELS_DIR, 'lstm.keras')
    model.save(model_path)
    joblib.dump(history.history, os.path.join(RESULTS_DIR, 'lstm_history.pkl'))

    # CV not computed for LSTM (computationally prohibitive); return placeholder
    cv_res = dict(cv_rmse_mean=float('nan'), cv_rmse_std=float('nan'),
                  cv_r2_mean=float('nan'),   cv_r2_std=float('nan'))
    return model, val_pred, test_pred, cv_res, y_vl, y_ts


# ── 11. NN (MLP) ──────────────────────────────────────────────────────────────

def train_nn(X_train, y_train, X_val, X_test, y_val, y_test, n_features):
    print("\n[11/11] Neural Network (MLP) …")
    import tensorflow as tf
    tf.random.set_seed(RANDOM_SEED)

    inp = tf.keras.Input(shape=(n_features,))
    x   = tf.keras.layers.Dense(128, activation='relu')(inp)
    x   = tf.keras.layers.Dropout(0.3)(x)
    x   = tf.keras.layers.Dense(64, activation='relu')(x)
    x   = tf.keras.layers.Dropout(0.3)(x)
    x   = tf.keras.layers.Dense(32, activation='relu')(x)
    out = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mse')

    cb_es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
    cb_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5, verbose=0)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=NN_EPOCHS,
        batch_size=NN_BATCH_SIZE,
        callbacks=[cb_es, cb_lr],
        verbose=1,
    )

    val_pred  = model.predict(X_val,  verbose=0).flatten()
    test_pred = model.predict(X_test, verbose=0).flatten()

    model.save(os.path.join(MODELS_DIR, 'nn.keras'))
    joblib.dump(history.history, os.path.join(RESULTS_DIR, 'nn_history.pkl'))

    cv_res = dict(cv_rmse_mean=float('nan'), cv_rmse_std=float('nan'),
                  cv_r2_mean=float('nan'),   cv_r2_std=float('nan'))
    return model, val_pred, test_pred, cv_res


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=== MODEL TRAINING ===")
    t0 = time.time()

    (X_train, y_train,
     X_val,   y_val,
     X_test,  y_test,
     train_df, val_df, test_df,
     feature_names, n_features) = _load_data()

    print(f"  X_train: {X_train.shape}  X_val: {X_val.shape}  X_test: {X_test.shape}")

    predictions = {}   # {name: {val_pred, test_pred, val_true, test_true, cv}}

    def _record(name, val_pred, test_pred, cv_res, val_true=None, test_true=None):
        predictions[name] = dict(
            val_pred  = val_pred,
            test_pred = test_pred,
            val_true  = val_true  if val_true  is not None else y_val,
            test_true = test_true if test_true is not None else y_test,
            **cv_res,
        )

    # 1. MLR
    mlr, vp, tp, cv = train_mlr(X_train, y_train, X_val, X_test, y_train, y_val, y_test, n_features)
    _record('MLR', vp, tp, cv)

    # 2. GBR
    gbr, vp, tp, cv, gbr_params = train_gbr(X_train, y_train, X_val, X_test, y_val, y_test, n_features)
    _record('GBR', vp, tp, cv)

    # 3. XGBoost
    xgb_m, vp, tp, cv, _ = train_xgb(X_train, y_train, X_val, X_test, y_val, y_test, n_features)
    _record('XGBoost', vp, tp, cv)

    # 4. LightGBM
    lgb_m, vp, tp, cv, _ = train_lgb(X_train, y_train, X_val, X_test, y_val, y_test, n_features)
    _record('LightGBM', vp, tp, cv)

    # 5. CatBoost
    cb_m, vp, tp, cv, _ = train_cb(X_train, y_train, X_val, X_test, y_val, y_test, n_features)
    _record('CatBoost', vp, tp, cv)

    # 6. Random Forest
    rf_m, vp, tp, cv, _ = train_rf(X_train, y_train, X_val, X_test, y_val, y_test, n_features)
    _record('RandomForest', vp, tp, cv)

    # 7. Stacking
    stk, vp, tp, cv = train_stacking(gbr, xgb_m, lgb_m, cb_m,
                                     X_train, y_train, X_val, X_test, y_val, y_test, n_features)
    _record('Stacking', vp, tp, cv)

    # 8. Bagging
    bag, vp, tp, cv = train_bagging(X_train, y_train, X_val, X_test, y_val, y_test, n_features)
    _record('Bagging', vp, tp, cv)

    # 9. Weighted Voting
    vot, vp, tp, cv = train_voting(gbr, xgb_m, lgb_m, cb_m,
                                   X_train, y_train, X_val, X_test, y_val, y_test, n_features)
    _record('WeightedVoting', vp, tp, cv)

    # 10. LSTM
    lstm_m, vp, tp, cv, y_vl_lstm, y_ts_lstm = train_lstm(
        train_df, val_df, test_df, feature_names, n_features
    )
    _record('LSTM', vp, tp, cv, val_true=y_vl_lstm, test_true=y_ts_lstm)

    # 11. NN
    nn_m, vp, tp, cv = train_nn(X_train, y_train, X_val, X_test, y_val, y_test, n_features)
    _record('NN', vp, tp, cv)

    # Save predictions
    pred_path = os.path.join(RESULTS_DIR, 'predictions.pkl')
    joblib.dump(predictions, pred_path)
    print(f"\n  Saved predictions → {pred_path}")

    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed/60:.1f} min.\n")


if __name__ == '__main__':
    main()
