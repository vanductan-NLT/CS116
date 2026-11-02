"""Label-free "teacher" features from the 20k-row source dataset (uditjain13/credit-card-fraud-detection-2026).

The source is near-additive (logistic regression beats LightGBM on it, Hemingweb's EDA), so two teachers are fit
on the source rows only and applied to competition train+test: a one-hot logistic regression and a small LightGBM.
Nothing from the competition labels is used, so the columns are fold-safe static features.
Output: data/original_features.parquet with columns id, orig_lr, orig_lgb (cached; group "original" in FE).
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import BOOL_COLS, CAT_COLS, DATA_DIR, ID_COL, NUM_COLS, TARGET
from data_loading import load_raw

OUT = DATA_DIR / "original_features.parquet"


def load_original() -> pd.DataFrame:
    o = pd.read_csv(next(DATA_DIR.glob("original/*.csv")))
    for c in CAT_COLS:
        o[c] = o[c].astype(str)
    for c in BOOL_COLS:
        o[c] = o[c].astype(str).str.lower().eq("true").astype("int8")
    o[TARGET] = o[TARGET].astype(int)
    return o


def main():
    o = load_original()
    train, test = load_raw()
    feats = NUM_COLS + BOOL_COLS + CAT_COLS
    X_o, y_o = o[feats], o[TARGET].values
    both = pd.concat([train[feats], test[feats]], ignore_index=True)

    lr = Pipeline([("prep", ColumnTransformer([("num", StandardScaler(), NUM_COLS + BOOL_COLS),
                                               ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS)])),
                   ("lr", LogisticRegression(C=1.0, max_iter=2000))])
    print("source LR 5-fold AUC:", round(roc_auc_score(y_o, cross_val_predict(lr, X_o, y_o, cv=5, method="predict_proba")[:, 1]), 5))
    lr.fit(X_o, y_o)
    p_lr = lr.predict_proba(both)[:, 1]

    import lightgbm as lgb
    Xo_l, both_l = X_o.copy(), both.copy()
    for c in CAT_COLS:
        cats = pd.Categorical(pd.concat([Xo_l[c], both_l[c]])).categories
        Xo_l[c] = pd.Categorical(Xo_l[c], categories=cats)
        both_l[c] = pd.Categorical(both_l[c], categories=cats)
    m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=50,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.7, verbose=-1)
    print("source LGBM 5-fold AUC:", round(roc_auc_score(y_o, cross_val_predict(m, Xo_l, y_o, cv=5, method="predict_proba")[:, 1]), 5))
    m.fit(Xo_l, y_o)
    p_lgb = m.predict_proba(both_l)[:, 1]

    ids = np.concatenate([train[ID_COL].values, test[ID_COL].values])
    out = pd.DataFrame({ID_COL: ids, "orig_lr": p_lr.astype("float32"), "orig_lgb": p_lgb.astype("float32")})
    out.to_parquet(OUT, index=False)
    y = train[TARGET].values
    print("teacher AUC on competition train: LR", round(roc_auc_score(y, p_lr[: len(train)]), 5),
          "| LGBM", round(roc_auc_score(y, p_lgb[: len(train)]), 5))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
