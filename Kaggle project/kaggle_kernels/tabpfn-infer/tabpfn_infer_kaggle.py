"""Kaggle script kernel: TabPFN-3.5 in-context inference with large context + route-group feature engineering.

Recipe: Chris Deotte, "Feature Engineering with TFMs" (S6E10 discussion #749007): raw columns + 9 features built
around Flight Distance as a group id lifted TabPFN-3.5 from 0.96115 to 0.96149. Here:
  - 8 label-free columns come from the attached dataset vanductan/s6e10-route-fe-features
    (per-FD means of 6 columns, FD count, original-data teacher prediction)
  - the 9th, target encoding of Flight Distance, is computed INSIDE each fold (inner-OOF for context rows,
    context-fitted for query rows) so no validation label leaks into the context.
fit_mode='fit_with_cache' + int8 KV cache lets a T4 hold ~400k context rows.
Phase A screens raw vs raw+FE on fold 0 (100k context); phase B runs the better input set on all folds.
Outputs: oof_fold{k}.npy, oof_idx_fold{k}.npy, test_fold{k}.npy, screen.json, done.json.
Requires Kaggle secret TABPFN_TOKEN, GPU, internet, inputs: competition + the features dataset.
"""
import json
import os
import subprocess
import sys
import time

FOLDS = [int(f) for f in os.environ.get("FOLDS", "0,1,2,3,4").split(",")]
CTX = int(os.environ.get("CTX", "400000"))
CTX_SCREEN = int(os.environ.get("CTX_SCREEN", "100000"))
N_EST = int(os.environ.get("N_EST", "1"))
FORCE_FE = os.environ.get("FORCE_FE", "")  # "1"/"0" to skip the screen

try:
    from kaggle_secrets import UserSecretsClient
    token = UserSecretsClient().get_secret("TABPFN_TOKEN").strip()
except Exception as e:  # noqa: BLE001
    sys.exit(f"Kaggle secret TABPFN_TOKEN is not attached ({e!r}).")
os.environ["TABPFN_TOKEN"] = token
os.environ["TABPFN_NO_BROWSER"] = "1"
os.environ["TABPFN_MAX_BATCHED_TEST_ROWS"] = "32768"
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tabpfn>=9.2.0"], check=True)

import glob  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from tabpfn import TabPFNClassifier  # noqa: E402

hits = glob.glob("/kaggle/input/**/playground-series-s6e10/train.csv", recursive=True)
if not hits:
    sys.exit("Competition data not attached: Add Input -> Competitions -> Predicting Airline Satisfaction.")
D = os.path.dirname(hits[0])
train, test = pd.read_csv(f"{D}/train.csv"), pd.read_csv(f"{D}/test.csv")
fe_hits = glob.glob("/kaggle/input/**/fe_features.parquet", recursive=True)
if not fe_hits:
    sys.exit("Feature dataset not attached: Add Input -> Datasets -> 'S6E10 label-free route features'.")
FE = pd.read_parquet(fe_hits[0]).set_index("id")
FE_NO_LLR = [c for c in FE.columns if c != "llr"]
FE_TR_ALL, FE_TE_ALL = FE.loc[train["id"]], FE.loc[test["id"]]
FE_TR, FE_TE = FE_TR_ALL[FE_NO_LLR].values.astype(np.float32), FE_TE_ALL[FE_NO_LLR].values.astype(np.float32)
HAS_LLR = "llr" in FE.columns
LLR_TR = FE_TR_ALL["llr"].values.astype(np.float32)[:, None] if HAS_LLR else None
LLR_TE = FE_TE_ALL["llr"].values.astype(np.float32)[:, None] if HAS_LLR else None
CATS = ["Gender", "Customer Type", "Type of Travel", "Class"]
y = train["satisfaction"].astype(int).values
FD_TR, FD_TE = train["Flight Distance"].values, test["Flight Distance"].values


def build_raw(df):
    X = df.drop(columns=["id", "satisfaction"], errors="ignore").copy()
    for c in CATS:
        X[c] = X[c].astype("category").cat.codes
    X["Arrival Delay in Minutes"] = X["Arrival Delay in Minutes"].fillna(-1)
    X["fd10"] = X["Flight Distance"] // 10
    return X.values.astype(np.float32), [X.columns.get_loc(c) for c in CATS + ["Flight Distance", "fd10"]]


X_raw, cat_idx = build_raw(train)
X_raw_te, _ = build_raw(test)
folds = np.zeros(len(train), dtype=int)
for k, (_, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(train, y)):
    folds[va] = k


def smoothed_mean(keys, yy, smoothing=20.0):
    prior = yy.mean()
    s = pd.DataFrame({"k": keys, "y": yy}).groupby("k")["y"].agg(["sum", "count"])
    return (s["sum"] + prior * smoothing) / (s["count"] + smoothing), prior


def te_fd(ctx_idx, query_sets):
    """Fold-safe target encoding of Flight Distance: inner-OOF for context rows, context-fitted for queries."""
    enc_ctx = np.zeros(len(ctx_idx), dtype=np.float32)
    for a, b in StratifiedKFold(5, shuffle=True, random_state=0).split(ctx_idx, y[ctx_idx]):
        m, prior = smoothed_mean(FD_TR[ctx_idx[a]], y[ctx_idx[a]])
        enc_ctx[b] = pd.Series(FD_TR[ctx_idx[b]]).map(m).fillna(prior).values
    m, prior = smoothed_mean(FD_TR[ctx_idx], y[ctx_idx])
    return enc_ctx, [pd.Series(q).map(m).fillna(prior).values.astype(np.float32) for q in query_sets]


def with_fe(X, fe, te, llr=None):
    parts = [X, fe, te[:, None]] + ([llr] if llr is not None else [])
    return np.hstack(parts)


def make_clf(seed):
    return TabPFNClassifier(n_estimators=N_EST, device="cuda", ignore_pretraining_limits=True, random_state=seed,
                            categorical_features_indices=cat_idx, fit_mode="fit_with_cache",
                            kv_cache_precision="int8", memory_saving_mode=True)


def predict(clf, Xq, chunk=16384):
    return np.concatenate([clf.predict_proba(Xq[i:i + chunk])[:, 1] for i in range(0, len(Xq), chunk)])


def run_fold(k, n_ctx, use_fe, va_subset=None, seed=None):
    """use_fe: False = raw, True = raw + 9 FE columns, 'llr' = raw + 9 FE columns + generator-surrogate LLR."""
    tr, va = np.where(folds != k)[0], np.where(folds == k)[0]
    if va_subset is not None:
        va = va_subset
    ctx = np.random.default_rng(k).choice(tr, min(n_ctx, len(tr)), replace=False)
    if use_fe:
        te_ctx, (te_va, te_te) = te_fd(ctx, [FD_TR[va], FD_TE])
        lt, lv, ls = (LLR_TR[ctx], LLR_TR[va], LLR_TE) if use_fe == "llr" else (None, None, None)
        Xc, Xv, Xt = (with_fe(X_raw[ctx], FE_TR[ctx], te_ctx, lt), with_fe(X_raw[va], FE_TR[va], te_va, lv),
                      with_fe(X_raw_te, FE_TE, te_te, ls))
    else:
        Xc, Xv, Xt = X_raw[ctx], X_raw[va], X_raw_te
    clf = make_clf(k if seed is None else seed)
    clf.fit(Xc, y[ctx])
    return clf, va, Xv, Xt, len(ctx)


# ---- phase A: screen raw vs raw+FE on fold 0 with a reduced context
if FORCE_FE in ("0", "1", "llr"):
    USE_FE = {"0": False, "1": True, "llr": "llr"}[FORCE_FE]
else:
    screen = {}
    va0 = np.where(folds == 0)[0]
    va_s = np.random.default_rng(1).choice(va0, 30000, replace=False)
    variants = [False, True] + (["llr"] if HAS_LLR else [])
    for use_fe in variants:
        t0 = time.time()
        clf, va, Xv, _, n = run_fold(0, CTX_SCREEN, use_fe, va_subset=va_s, seed=0)
        screen[use_fe] = roc_auc_score(y[va], predict(clf, Xv))
        print(f"screen variant={use_fe}: ctx {n} -> fold-0 AUC {screen[use_fe]:.6f} on 30k rows ({time.time() - t0:.0f}s, "
              f"peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.1f} GB)", flush=True)
        del clf
        torch.cuda.empty_cache()
    USE_FE = max(screen, key=screen.get)
    json.dump({str(k): v for k, v in screen.items()}, open("screen.json", "w"))
print(f"phase B uses variant={USE_FE}", flush=True)

results = {}
for FOLD in FOLDS:
    t0 = time.time()
    clf, va, Xv, Xt, n = run_fold(FOLD, CTX, USE_FE)
    print(f"fold {FOLD}: context {n} cached in {time.time() - t0:.0f}s, peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
    oof = predict(clf, Xv)
    auc = roc_auc_score(y[va], oof)
    print(f"fold {FOLD}: AUC={auc:.6f} ({time.time() - t0:.0f}s)", flush=True)
    np.save(f"oof_fold{FOLD}.npy", oof)
    np.save(f"oof_idx_fold{FOLD}.npy", va)
    np.save(f"test_fold{FOLD}.npy", predict(clf, Xt))
    results[FOLD] = {"auc": auc, "ctx": int(n), "fe": bool(USE_FE), "seconds": time.time() - t0}
    json.dump(results, open("done.json", "w"))
    print(f"fold {FOLD} done in {time.time() - t0:.0f}s", flush=True)
    del clf
    torch.cuda.empty_cache()
