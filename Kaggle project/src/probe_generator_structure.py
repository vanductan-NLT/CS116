"""Probe for structure the synthetic generator may have left behind (EDA for the report).

1. Exact matches between competition rows and ORIGINAL rows on the 17 discrete columns
   (13 ratings + 4 categoricals): if the generator perturbed original rows, the matched original label
   should predict the competition label far better than the marginal rate.
2. Duplicate discrete tuples inside train(+test): in-fold target rate of the tuple as a feature.
3. Row-order structure: does the target / a feature drift with `id`?
Prints AUCs so each hypothesis is accepted or rejected with a number.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from config import CAT_COLS, RATING_COLS, TARGET
from data_loading import get_folds, load_raw
from original_data_features import load_original

train, test = load_raw()
orig = load_original()
y = train[TARGET].values
DISC = RATING_COLS + CAT_COLS
folds = get_folds(train)


def key(df, cols):
    return pd.util.hash_pandas_object(df[cols].astype(str), index=False).values


# ---- 1. exact discrete-tuple matches against the original dataset
for cols, label in [(DISC, "17 discrete"), (DISC + ["Age"], "17 discrete + Age"),
                    (DISC + ["Age", "Flight Distance"], "17 discrete + Age + FD")]:
    k_tr, k_or = key(train, cols), key(orig, cols)
    stats = pd.DataFrame({"k": k_or, "y": orig[TARGET].values}).groupby("k")["y"].agg(["mean", "count"])
    m = pd.Series(k_tr).map(stats["mean"])
    hit = m.notna().values
    print(f"[orig match on {label}] train rows matched: {hit.mean():.3%} ({hit.sum()})", end="")
    if hit.sum() > 1000:
        print(f"  AUC(orig label rate -> y | matched) = {roc_auc_score(y[hit], m[hit]):.4f}", end="")
    print()

# ---- 2. duplicate discrete tuples inside the competition data (in-fold target rate)
for cols, label in [(DISC, "17 discrete"), (DISC + ["Age"], "+Age"), (DISC + ["Age", "Flight Distance"], "+Age+FD")]:
    k_tr = key(train, cols)
    k_te = key(test, cols)
    print(f"[dup tuples on {label}] train rows whose tuple repeats in train: "
          f"{pd.Series(k_tr).duplicated(keep=False).mean():.3%}; test tuples seen in train: "
          f"{np.isin(k_te, k_tr).mean():.3%}")
    te_rate = np.full(len(train), np.nan)
    for f in range(5):
        tr, va = folds != f, folds == f
        st = pd.DataFrame({"k": k_tr[tr], "y": y[tr]}).groupby("k")["y"].agg(["sum", "count"])
        rate = (st["sum"] + 0.44 * 5) / (st["count"] + 5)
        te_rate[va] = pd.Series(k_tr[va]).map(rate).fillna(0.44).values
    print(f"      AUC(in-fold tuple target rate) = {roc_auc_score(y, te_rate):.4f}")

# ---- 3. row-order structure
ids = train["id"].values
block = ids // 10000
rate_by_block = pd.Series(y).groupby(block).mean()
print(f"[id structure] target rate by 10k-id block: std={rate_by_block.std():.4f} "
      f"(binomial noise ~{np.sqrt(0.44*0.56/10000):.4f}); AUC(id)={roc_auc_score(y, ids):.4f}")
for c in ["Flight Distance", "Age", "Online boarding"]:
    print(f"      corr(id, {c}) = {np.corrcoef(ids, train[c].values)[0,1]:+.4f}")
