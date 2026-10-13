"""Auxiliary-task features: for each service rating, predict it from all other columns.

Label-free (never touches `satisfaction`), so it is fit once on train+test with out-of-fold predictions.
The expected rating E[r | rest] and the residual r - E[r | rest] describe how "surprising" a passenger's
rating is given their profile — a signal the GBDTs struggle to build from raw splits.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold

from config import DATA_DIR, RATING_COLS

_CACHE = DATA_DIR / "aux_task_features.parquet"


def build_aux_task_features(X_full: pd.DataFrame) -> pd.DataFrame:
    """X_full: raw-encoded features for concat(train, test) (21 base columns). Returns aux_* frame."""
    if _CACHE.exists():
        return pd.read_parquet(_CACHE)
    out = {}
    kf = KFold(5, shuffle=True, random_state=0)
    for r in RATING_COLS:
        X, y = X_full.drop(columns=[r]), X_full[r].values.astype("float32")
        pred = np.zeros(len(X), dtype="float32")
        for tr, va in kf.split(X):
            m = xgb.XGBRegressor(n_estimators=400, learning_rate=0.08, max_depth=8, subsample=0.8,
                                 colsample_bytree=0.7, tree_method="hist", device="cuda")
            m.fit(X.iloc[tr], y[tr], verbose=False)
            pred[va] = m.predict(X.iloc[va])
        out[f"aux_exp_{r}"] = pred
        out[f"aux_res_{r}"] = (y - pred).astype("float32")
        print(f"aux {r}: rmse={np.sqrt(np.mean((y - pred) ** 2)):.4f}", flush=True)
    df = pd.DataFrame(out)
    df.to_parquet(_CACHE, index=False)
    return df
