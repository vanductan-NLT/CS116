"""Hyper-parameter optimisation for the report: Optuna (TPE) vs exhaustive GridSearch on the same split.

Both tune XGBoost (GPU) on fold 0 of the shared split with the final feature set; the best Optuna params are
then verified with full 5-fold CV via run_cv.py (see logs/cv_results.csv, name xgb_optuna).
Results: logs/optuna_trials.csv, logs/grid_trials.csv
"""
import argparse
import itertools
import time

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from config import LOG_DIR, TARGET
from data_loading import get_folds, load_raw
from feature_engineering import add_target_encoding, build_static_features

ap = argparse.ArgumentParser()
ap.add_argument("--trials", type=int, default=30)
a = ap.parse_args()

train, test = load_raw()
y = train[TARGET].values
X, X_test = build_static_features(train, test, {"aux", "delay", "fd_freq", "route_full", "original"})
folds = get_folds(train)
tr, va = np.where(folds != 0)[0], np.where(folds == 0)[0]
X_tr, X_va, _ = add_target_encoding(X.iloc[tr], y[tr], X.iloc[va], X_test, ["Flight Distance"], seed=0)
dtr, dva = xgb.DMatrix(X_tr, y[tr]), xgb.DMatrix(X_va, y[va])
BASE = dict(objective="binary:logistic", eval_metric="auc", tree_method="hist", device="cuda", nthread=4)


def fit(params, rounds=20000):
    bst = xgb.train({**BASE, **params}, dtr, rounds, evals=[(dva, "va")], early_stopping_rounds=200, verbose_eval=False)
    return bst.best_score, bst.best_iteration


# ---- Optuna / TPE
def objective(trial):
    p = dict(learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
             max_depth=trial.suggest_int("max_depth", 4, 10),
             min_child_weight=trial.suggest_float("min_child_weight", 1, 50, log=True),
             subsample=trial.suggest_float("subsample", 0.5, 1.0),
             colsample_bytree=trial.suggest_float("colsample_bytree", 0.2, 0.8),
             reg_lambda=trial.suggest_float("reg_lambda", 0.1, 30, log=True),
             reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 5, log=True),
             gamma=trial.suggest_float("gamma", 1e-3, 2, log=True))
    auc, it = fit(p)
    trial.set_user_attr("best_iteration", it)
    return auc


t0 = time.time()
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
study.enqueue_trial(dict(learning_rate=0.02, max_depth=7, min_child_weight=5, subsample=0.8, colsample_bytree=0.4,
                         reg_lambda=1.0, reg_alpha=1e-3, gamma=1e-3))  # hand-tuned reference
study.optimize(objective, n_trials=a.trials)
df = study.trials_dataframe()
df.to_csv(LOG_DIR / "optuna_trials.csv", index=False)
print(f"Optuna: {a.trials} trials in {time.time() - t0:.0f}s, best fold-0 AUC {study.best_value:.6f}")
print("best params:", study.best_params)

# ---- GridSearch over the two most influential dimensions (same budget order of magnitude)
t0, rows = time.time(), []
for depth, lr, col in itertools.product([5, 7, 9], [0.02, 0.05], [0.3, 0.5]):
    auc, it = fit(dict(max_depth=depth, learning_rate=lr, colsample_bytree=col, subsample=0.8, min_child_weight=5, reg_lambda=1.0))
    rows.append(dict(max_depth=depth, learning_rate=lr, colsample_bytree=col, auc=auc, best_iteration=it))
    print(f"grid depth={depth} lr={lr} col={col}: {auc:.6f} ({it} rounds)", flush=True)
grid = pd.DataFrame(rows).sort_values("auc", ascending=False)
grid.to_csv(LOG_DIR / "grid_trials.csv", index=False)
print(f"GridSearch: {len(rows)} configs in {time.time() - t0:.0f}s, best fold-0 AUC {grid.auc.iloc[0]:.6f}")
