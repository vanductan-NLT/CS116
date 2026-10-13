"""Run K-fold CV for one model/feature configuration and save OOF + averaged test predictions.

Example:
  python src/run_cv.py --name lgbm_base --model lgbm
  python src/run_cv.py --name lgbm_fe1 --model lgbm --groups delay,fd_freq --te "Flight Distance"
"""
import argparse
import json
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from config import CAT_COLS, LOG_DIR, OOF_DIR, TARGET, TEST_PRED_DIR
from data_loading import get_folds, load_raw
from feature_engineering import add_target_encoding, build_static_features
from model_zoo import make_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True, help="output file stem in oof/ and test_preds/")
    p.add_argument("--model", required=True, choices=["lr", "knn", "rf", "lgbm", "xgb", "cat"])
    p.add_argument("--groups", default="", help="comma list of static feature groups")
    p.add_argument("--te", default="", help="comma list of columns to target-encode in-fold")
    p.add_argument("--params", default="{}", help="JSON dict of model params")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--fold-seed", type=int, default=42)
    p.add_argument("--only-folds", default="", help="debug: comma list of fold ids to run")
    p.add_argument("--pseudo", default="", help="name of a run whose <name>_by_fold.npy test preds become soft labels")
    p.add_argument("--pseudo-thr", type=float, default=0.0, help="keep test rows with |p-0.5| >= thr (0 = all)")
    p.add_argument("--pseudo-weight", type=float, default=1.0, help="sample weight of pseudo-labelled rows")
    p.add_argument("--pseudo-hard", action="store_true", help="round soft labels to 0/1")
    p.add_argument("--self-distill", type=float, default=0.0,
                   help="alpha>0: train on (1-alpha)*y + alpha*inner-OOF teacher prob (teacher fit inside the fold)")
    return p.parse_args()


def main():
    a = parse_args()
    groups = {g for g in a.groups.split(",") if g}
    te_cols = [c for c in a.te.split(",") if c]
    params = json.loads(a.params)

    train, test = load_raw()
    y = train[TARGET].values
    X, X_test = build_static_features(train, test, groups)
    folds = get_folds(train, a.folds, a.fold_seed)
    run_folds = [int(f) for f in a.only_folds.split(",") if f] or list(range(a.folds))

    oof = np.full(len(X), np.nan)
    test_pred = np.zeros(len(X_test))
    # per-fold test predictions: fold k's model never saw fold k, so they can serve as fold-safe pseudo-labels
    test_by_fold = np.zeros((a.folds, len(X_test)), dtype="float32")
    t0 = time.time()
    for k in run_folds:
        tr, va = np.where(folds != k)[0], np.where(folds == k)[0]
        X_tr, X_va, X_te = X.iloc[tr], X.iloc[va], X_test
        y_tr, w_tr = y[tr].astype("float32"), np.ones(len(tr), dtype="float32")
        if a.pseudo:
            # fold k's teacher never saw fold k, so adding test rows here keeps the validation honest
            pl = np.load(TEST_PRED_DIR / f"{a.pseudo}_by_fold.npy")[k].astype("float32")
            sel = np.abs(pl - 0.5) >= a.pseudo_thr
            X_tr = pd.concat([X_tr, X_test[sel]], ignore_index=True)
            y_tr = np.concatenate([y_tr, np.round(pl[sel]) if a.pseudo_hard else pl[sel]])
            w_tr = np.concatenate([w_tr, np.full(sel.sum(), a.pseudo_weight, dtype="float32")])
        if te_cols:
            X_tr, X_va, X_te = add_target_encoding(X_tr, y_tr, X_va, X_te, te_cols, seed=k)
        if a.self_distill > 0:
            # label-noise smoothing: inner 4-fold teacher of the same model class, fit only on this fold's train rows
            inner = np.zeros(len(X_tr), dtype="float32")
            for ia, ib in StratifiedKFold(4, shuffle=True, random_state=100 + k).split(X_tr, np.round(y_tr)):
                t = make_model(a.model, params, CAT_COLS).fit(X_tr.iloc[ia], y_tr[ia], X_tr.iloc[ib], np.round(y_tr[ib]))
                inner[ib] = t.predict(X_tr.iloc[ib])
            y_tr = (1 - a.self_distill) * y_tr + a.self_distill * inner
        model = make_model(a.model, params, CAT_COLS).fit(X_tr, y_tr, X_va, y[va], sample_weight=w_tr)
        oof[va] = model.predict(X_va)
        test_by_fold[k] = model.predict(X_te)
        test_pred += test_by_fold[k] / len(run_folds)
        print(f"fold {k}: AUC={roc_auc_score(y[va], oof[va]):.6f}  ({time.time() - t0:.0f}s)", flush=True)

    mask = ~np.isnan(oof)
    cv = roc_auc_score(y[mask], oof[mask])
    print(f"[{a.name}] CV AUC = {cv:.6f}  n_features={X_tr.shape[1]}  time={time.time() - t0:.0f}s")
    if len(run_folds) == a.folds:  # only persist complete runs
        np.save(OOF_DIR / f"{a.name}.npy", oof)
        np.save(TEST_PRED_DIR / f"{a.name}.npy", test_pred)
        np.save(TEST_PRED_DIR / f"{a.name}_by_fold.npy", test_by_fold)
    with open(LOG_DIR / "cv_results.csv", "a", encoding="utf-8") as f:
        pseudo = f"{a.pseudo} thr={a.pseudo_thr} w={a.pseudo_weight} hard={a.pseudo_hard}" if a.pseudo else ""
        pseudo += f" self_distill={a.self_distill}" if a.self_distill else ""
        f.write(f"{datetime.now():%Y-%m-%d %H:%M},{a.name},{a.model},{cv:.6f},{len(run_folds)}/{a.folds},"
                f"\"{a.groups}\",\"{a.te}\",\"{a.params.replace(chr(34), chr(39))}\",\"{pseudo}\"\n")


if __name__ == "__main__":
    main()
