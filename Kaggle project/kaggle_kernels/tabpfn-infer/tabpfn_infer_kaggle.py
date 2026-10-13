"""Kaggle script kernel: TabPFN-3.5 in-context inference with large context + generator-aware (BPE) inputs.

Recipe from the S6E9 2nd-place writeup: no fine-tuning, fit_mode='fit_with_cache' with an int8 KV cache so a
T4 holds ~430k context rows; inputs = raw columns + GPT-2 BPE tokens / digits of the numeric strings.
Per fold k: context = up to CTX rows sampled from the other folds -> OOF for fold k and test preds (averaged).
Outputs: oof_fold{k}.npy, oof_idx_fold{k}.npy, test_fold{k}.npy, done.json (fold AUCs).
Requires Kaggle secret TABPFN_TOKEN, GPU, internet. FOLDS / CTX / N_EST via env.
"""
import json
import os
import subprocess
import sys
import time

FOLDS = [int(f) for f in os.environ.get("FOLDS", "0,1,2,3,4").split(",")]
CTX = int(os.environ.get("CTX", "400000"))
CTX_SCREEN = int(os.environ.get("CTX_SCREEN", "100000"))  # phase A: fold-0 screen of raw vs BPE inputs
N_EST = int(os.environ.get("N_EST", "1"))

try:
    from kaggle_secrets import UserSecretsClient
    token = UserSecretsClient().get_secret("TABPFN_TOKEN").strip()
except Exception as e:  # noqa: BLE001
    sys.exit(f"Kaggle secret TABPFN_TOKEN is not attached ({e!r}).")
os.environ["TABPFN_TOKEN"] = token
os.environ["TABPFN_NO_BROWSER"] = "1"
os.environ["TABPFN_MAX_BATCHED_TEST_ROWS"] = "32768"
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tabpfn>=9.2.0", "tokenizers"], check=True)

import glob  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from tabpfn import TabPFNClassifier  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

hits = glob.glob("/kaggle/input/**/playground-series-s6e10/train.csv", recursive=True)
if not hits:
    sys.exit("Competition data not attached: Add Input -> Competitions -> Predicting Airline Satisfaction.")
D = os.path.dirname(hits[0])
train, test = pd.read_csv(f"{D}/train.csv"), pd.read_csv(f"{D}/test.csv")
CATS = ["Gender", "Customer Type", "Type of Travel", "Class"]
y = train["satisfaction"].astype(int).values
tok = Tokenizer.from_pretrained("gpt2")


def build(df, bpe):
    X = df.drop(columns=["id", "satisfaction"], errors="ignore").copy()
    for c in CATS:
        X[c] = X[c].astype("category").cat.codes
    X["Arrival Delay in Minutes"] = X["Arrival Delay in Minutes"].fillna(-1)
    X["fd10"] = X["Flight Distance"] // 10
    cats = CATS + ["Flight Distance", "fd10"]
    if bpe:
        for c, tag in [("Flight Distance", "fd"), ("Age", "age"), ("Departure Delay in Minutes", "dep"),
                       ("Arrival Delay in Minutes", "arr")]:
            vals = X[c].astype(int)
            enc = {v: tok.encode(f" {v}").tokens for v in vals.unique()}
            X[f"bpe_{tag}_first"] = vals.map(lambda v: enc[v][0]).astype("category").cat.codes
            X[f"bpe_{tag}_last"] = vals.map(lambda v: enc[v][-1]).astype("category").cat.codes
            X[f"bpe_{tag}_shape"] = vals.map(lambda v: "-".join(str(len(t.lstrip("Ġ"))) for t in enc[v])).astype("category").cat.codes
            X[f"bpe_{tag}_n"] = vals.map(lambda v: len(enc[v]))
            cats += [f"bpe_{tag}_first", f"bpe_{tag}_last", f"bpe_{tag}_shape"]
        fd = X["Flight Distance"].astype(int)
        for d in range(4):
            X[f"fd_digit_{d}"] = (fd // 10**d) % 10
    return X.values.astype(np.float32), [X.columns.get_loc(c) for c in cats]


folds = np.zeros(len(train), dtype=int)
for k, (_, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(train, y)):
    folds[va] = k


def predict(clf, Xq, chunk=16384):
    return np.concatenate([clf.predict_proba(Xq[i:i + chunk])[:, 1] for i in range(0, len(Xq), chunk)])


def make_clf(cat_idx, seed):
    return TabPFNClassifier(n_estimators=N_EST, device="cuda", ignore_pretraining_limits=True, random_state=seed,
                            categorical_features_indices=cat_idx, fit_mode="fit_with_cache",
                            kv_cache_precision="int8", memory_saving_mode=True)


# ---- phase A: screen the two input sets on fold 0 with a reduced context (same context rows, same seed)
screen = {}
tr0, va0 = np.where(folds != 0)[0], np.where(folds == 0)[0]
ctx0 = np.random.default_rng(0).choice(tr0, min(CTX_SCREEN, len(tr0)), replace=False)
va_s = np.random.default_rng(1).choice(va0, 30000, replace=False)
for bpe in (False, True):
    t0 = time.time()
    Xs, cats_s = build(train, bpe)
    clf = make_clf(cats_s, 0)
    clf.fit(Xs[ctx0], y[ctx0])
    screen[bpe] = roc_auc_score(y[va_s], predict(clf, Xs[va_s]))
    print(f"screen bpe={bpe}: ctx {len(ctx0)} -> fold-0 AUC {screen[bpe]:.6f} on 30k rows ({time.time() - t0:.0f}s, "
          f"peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.1f} GB)", flush=True)
    del clf
    torch.cuda.empty_cache()
BPE = screen[True] >= screen[False]
print(f"phase B uses bpe={BPE}", flush=True)
json.dump({"screen_raw": screen[False], "screen_bpe": screen[True]}, open("screen.json", "w"))

X, cat_idx = build(train, BPE)
X_te, _ = build(test, BPE)
results = {}
for FOLD in FOLDS:
    t0 = time.time()
    tr, va = np.where(folds != FOLD)[0], np.where(folds == FOLD)[0]
    ctx = np.random.default_rng(FOLD).choice(tr, min(CTX, len(tr)), replace=False)
    clf = make_clf(cat_idx, FOLD)
    clf.fit(X[ctx], y[ctx])
    print(f"fold {FOLD}: context {len(ctx)} cached in {time.time() - t0:.0f}s, peak VRAM "
          f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
    oof = predict(clf, X[va])
    auc = roc_auc_score(y[va], oof)
    print(f"fold {FOLD}: AUC={auc:.6f} ({time.time() - t0:.0f}s)", flush=True)
    np.save(f"oof_fold{FOLD}.npy", oof)
    np.save(f"oof_idx_fold{FOLD}.npy", va)
    np.save(f"test_fold{FOLD}.npy", predict(clf, X_te))
    results[FOLD] = {"auc": auc, "ctx": int(len(ctx)), "seconds": time.time() - t0}
    print(f"fold {FOLD} done in {time.time() - t0:.0f}s", flush=True)
    json.dump(results, open("done.json", "w"))
    del clf
    torch.cuda.empty_cache()
