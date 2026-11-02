"""Level-2 stacking of saved OOF predictions.

Meta-model: L2-regularised logistic regression on logit(OOF). Scored with nested CV on a *different*
fold seed than the members (meta folds seed 7), so the reported AUC is honest.
Family regularisation (goodpjw2008, Part 11): scaling a column by s divides its effective L2 penalty by s^2,
so TabPFN x3 / other TFM x1.5 trusts the foundation-model families more a priori.

Example:
  python src/stack_oof.py --members lgbm_v1,xgb_v1,cat_v1 --out stack_v1
  python src/stack_oof.py --members all --exclude pub_gl_f --family-reg --out stack_all
"""
import argparse
import os
import time

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from config import OOF_DIR, SUB_DIR, TARGET, TEST_PRED_DIR
from data_loading import load_raw, write_submission


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-15, 1 - 1e-15)
    return np.clip(np.log(p / (1 - p)), -30, 30)


def family(n):
    n = n.lower()
    if "tabpfn" in n or "pfn" in n:
        return "tabpfn"
    if any(k in n for k in ("tfm_", "tabicl", "tabm", "kumo", "causilo", "limix", "mitra", "exaone", "tabfm")):
        return "tfm"
    if "mlp" in n:
        return "mlp"
    return "gbdt"


def load_members(spec, exclude):
    names = sorted(p.stem for p in OOF_DIR.glob("*.npy")) if spec == ["all"] else spec
    names = [n for n in names if not any(e in n for e in exclude) and not n.endswith("_by_fold")]
    oof = np.column_stack([np.load(OOF_DIR / f"{n}.npy") for n in names]).astype(np.float32)
    test = np.column_stack([np.load(TEST_PRED_DIR / f"{n}.npy") for n in names]).astype(np.float32)
    return names, oof, test


def _fit_fold(Z, y, a, b, C):
    return b, LogisticRegression(C=C, max_iter=5000).fit(Z[a], y[a]).decision_function(Z[b])


def nested_cv(Z, y, C, seed=7, n_jobs=5):
    """Return nested OOF decision values of the meta-model (meta folds on `seed`)."""
    skf = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(Z, y))
    out = np.zeros(len(y))
    # threads, not processes: avoids pickling/memmapping the 1-2 GB logit matrix to the temp drive
    for b, v in Parallel(n_jobs=min(n_jobs, os.cpu_count()), prefer="threads")(
            delayed(_fit_fold)(Z, y, a, b, C) for a, b in skf):
        out[b] = v
    return out, skf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", required=True, help="comma list of OOF names, or 'all'")
    ap.add_argument("--exclude", default="", help="comma list of substrings to drop")
    ap.add_argument("--C", type=float, default=0.003)
    ap.add_argument("--family-reg", action="store_true")
    ap.add_argument("--meta-seed", type=int, default=7)
    ap.add_argument("--out", default="")
    ap.add_argument("--quiet", action="store_true", help="skip per-member AUC listing")
    a = ap.parse_args()

    train, test = load_raw()
    y = train[TARGET].values
    names, oof, test_pred = load_members(a.members.split(","), [e for e in a.exclude.split(",") if e])
    t0 = time.time()
    if not a.quiet:
        for i, n in enumerate(names):
            print(f"  {n:45s} {roc_auc_score(y, oof[:, i]):.6f}")
    fam = np.array([family(n) for n in names])
    print({f: int((fam == f).sum()) for f in ("gbdt", "mlp", "tabpfn", "tfm")}, f"{len(names)} members")
    print(f"simple mean of probs : {roc_auc_score(y, oof.mean(1)):.6f}")

    scale = np.array([{"tabpfn": 3.0, "tfm": 1.5}.get(f, 1.0) for f in fam]) if a.family_reg else np.ones(len(names))
    Z, Zt = logit(oof) * scale, logit(test_pred) * scale
    del oof, test_pred
    meta_oof, skf = nested_cv(Z, y, a.C, a.meta_seed)
    folds = " ".join(f"{roc_auc_score(y[b], meta_oof[b]):.5f}" for _, b in skf)
    print(f"LR stack nested CV   : {roc_auc_score(y, meta_oof):.6f}  folds [{folds}]  "
          f"(C={a.C}, family_reg={a.family_reg}, seed={a.meta_seed}, {time.time() - t0:.0f}s)")

    if a.out:
        final = LogisticRegression(C=a.C, max_iter=5000).fit(Z, y)
        top = sorted(zip(names, final.coef_[0]), key=lambda x: -abs(x[1]))[:15]
        print("top |weights|:", [(n, round(float(w), 3)) for n, w in top])
        pred = 1 / (1 + np.exp(-final.decision_function(Zt)))
        write_submission(test, pred, SUB_DIR / f"{a.out}.csv")
        np.save(SUB_DIR / f"{a.out}_meta_oof.npy", meta_oof)
        print(f"wrote submissions/{a.out}.csv")


if __name__ == "__main__":
    main()
