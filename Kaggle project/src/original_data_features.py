"""Features derived from the original Airline Passenger Satisfaction dataset (129,880 rows, public).

The competition data was synthesized from it, so the original acts as an external "teacher":
- teacher: a model trained ONLY on original rows, predicting comp train/test (label-free w.r.t. comp target)
- lookup: smoothed satisfaction rate in the original for each value of a column
Both are computed once and cached; neither uses competition labels, so no fold leakage.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd

from config import CAT_COLS, DATA_DIR

LOOKUP_COLS = ["Flight Distance", "Age", "Online boarding", "Inflight wifi service", "Class", "Type of Travel"]
_CACHE = DATA_DIR / "original_features.parquet"


def load_original() -> pd.DataFrame:
    o = pd.concat([pd.read_csv(DATA_DIR / "original" / f) for f in ("train.csv", "test.csv")], ignore_index=True)
    o["satisfaction"] = (o["satisfaction"] == "satisfied").astype(int)
    return o.drop(columns=["Unnamed: 0", "id", "Inflight service"])


def _encode(df: pd.DataFrame) -> pd.DataFrame:
    X = df.copy()
    for c in CAT_COLS:
        X[c] = X[c].astype("category")
    X["Arrival Delay in Minutes"] = X["Arrival Delay in Minutes"].fillna(X["Departure Delay in Minutes"])
    return X


def build_original_features(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Return a frame aligned to concat(train, test) with orig_* columns."""
    if _CACHE.exists():
        return pd.read_parquet(_CACHE)
    orig = load_original()
    feat_cols = [c for c in orig.columns if c != "satisfaction"]
    full = pd.concat([train[feat_cols], test[feat_cols]], ignore_index=True)

    # Teacher: average of 5 seeds trained on all original rows
    Xo, yo = _encode(orig[feat_cols]), orig["satisfaction"].values
    Xf = _encode(full)
    for c in CAT_COLS:
        Xf[c] = pd.Categorical(Xf[c], categories=Xo[c].cat.categories)
    teacher = np.zeros(len(full))
    for seed in range(5):
        m = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=63, subsample=0.8,
                               subsample_freq=1, colsample_bytree=0.6, random_state=seed, verbose=-1)
        m.fit(Xo, yo)
        teacher += m.predict_proba(Xf)[:, 1] / 5
    out = pd.DataFrame({"orig_teacher": teacher.astype("float32")})
    out["orig_teacher_logit"] = np.log(np.clip(teacher, 1e-6, 1 - 1e-6) / np.clip(1 - teacher, 1e-6, 1)).astype("float32")

    # Lookups: smoothed per-value satisfaction rate in the original data
    prior, k = yo.mean(), 20.0
    for c in LOOKUP_COLS:
        s = orig.groupby(c)["satisfaction"].agg(["sum", "count"])
        rate = (s["sum"] + prior * k) / (s["count"] + k)
        out[f"orig_rate_{c}"] = full[c].map(rate).fillna(prior).astype("float32").values
        out[f"orig_cnt_{c}"] = full[c].map(s["count"]).fillna(0).astype("float32").values
    out.to_parquet(_CACHE, index=False)
    return out
