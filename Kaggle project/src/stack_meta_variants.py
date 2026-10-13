"""Compare level-2 meta-learners on the same member pool with the same nested folds (seed 7).

Variants:
  lr            plain family-regularised logistic regression (reference)
  lr_x<col>     LR whose member weights may differ per level of a raw categorical column
                (columns Z and Z * onehot(col) side by side = segment-wise experts)
  lgbm          shallow, heavily regularised LightGBM on logits + raw categoricals
Everything label-free on the feature side; only the meta folds see labels.
"""
import argparse
import time

import lightgbm as lgb
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from config import SUB_DIR, TARGET
from data_loading import load_raw, write_submission
from stack_oof import family, load_members, logit

ap = argparse.ArgumentParser()
ap.add_argument("--variants", default="lr,lr_xClass,lr_xType of Travel,lr_xCustomer Type,lgbm")
ap.add_argument("--exclude", default="xgb_v1_pl")
ap.add_argument("--out", default="", help="write submission for this variant name")
a = ap.parse_args()

train, test = load_raw()
y = train[TARGET].values
names, oof, tst = load_members(["all"], a.exclude.split(","))
scale = np.array([{"tabpfn": 3.0, "tfm": 1.5}.get(family(n), 1.0) for n in names])
Z, Zt = (logit(oof) * scale).astype(np.float32), (logit(tst) * scale).astype(np.float32)
del oof, tst
print(f"{len(names)} members")
skf = list(StratifiedKFold(5, shuffle=True, random_state=7).split(Z, y))


def onehot(col):
    tr = train[col].astype("category")
    te = test[col].astype(str).astype("category").cat.set_categories(tr.cat.categories)
    return np.eye(len(tr.cat.categories), dtype=np.float32)[tr.cat.codes], \
        np.eye(len(tr.cat.categories), dtype=np.float32)[te.cat.codes]


def interact(Zm, oh):  # [Z, Z*oh_1, ..., Z*oh_k]  (drop first level to avoid full collinearity)
    return np.hstack([Zm] + [Zm * oh[:, [j]] for j in range(1, oh.shape[1])])


def run_lr(Xtr, Xte, C=0.003):
    out = np.zeros(len(y))
    for a_, b in skf:
        out[b] = LogisticRegression(C=C, max_iter=5000).fit(Xtr[a_], y[a_]).decision_function(Xtr[b])
    final = LogisticRegression(C=C, max_iter=5000).fit(Xtr, y).decision_function(Xte)
    return out, 1 / (1 + np.exp(-final))


def run_lgbm(Xtr, Xte):
    params = dict(objective="binary", learning_rate=0.02, num_leaves=7, min_child_samples=3000,
                  feature_fraction=0.3, bagging_fraction=0.7, bagging_freq=1, lambda_l2=50.0, verbose=-1, n_jobs=8)
    out, rounds = np.zeros(len(y)), []
    for a_, b in skf:
        inner_tr, inner_va = a_[: int(0.9 * len(a_))], a_[int(0.9 * len(a_)):]  # early stop on a slice of meta-train
        m = lgb.train(params, lgb.Dataset(Xtr[inner_tr], y[inner_tr]), 3000,
                      valid_sets=[lgb.Dataset(Xtr[inner_va], y[inner_va])],
                      callbacks=[lgb.early_stopping(200, verbose=False)])
        rounds.append(m.best_iteration)
        out[b] = m.predict(Xtr[b], num_iteration=m.best_iteration)
    m = lgb.train(params, lgb.Dataset(Xtr, y), int(np.mean(rounds)))
    return out, m.predict(Xte)


for v in a.variants.split(","):
    t0 = time.time()
    if v == "lr":
        meta, pred = run_lr(Z, Zt)
    elif v.startswith("lr_x"):
        oh_tr, oh_te = onehot(v[4:])
        meta, pred = run_lr(interact(Z, oh_tr), interact(Zt, oh_te))
    elif v == "lgbm":
        cats_tr = np.column_stack([train[c].astype("category").cat.codes for c in ["Class", "Type of Travel", "Customer Type"]])
        cats_te = np.column_stack([test[c].astype("category").cat.codes for c in ["Class", "Type of Travel", "Customer Type"]])
        meta, pred = run_lgbm(np.hstack([Z, cats_tr]), np.hstack([Zt, cats_te]))
    else:
        raise ValueError(v)
    folds = " ".join(f"{roc_auc_score(y[b], meta[b]):.5f}" for _, b in skf)
    print(f"{v:22s} nested CV {roc_auc_score(y, meta):.6f}  folds [{folds}]  ({time.time() - t0:.0f}s)", flush=True)
    if a.out and v == a.out:
        write_submission(test, pred, SUB_DIR / f"meta_{v.replace(' ', '_')}.csv")
        print("wrote", SUB_DIR / f"meta_{v.replace(' ', '_')}.csv")
