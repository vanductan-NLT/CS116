"""Import every public OOF/test pair under data/public_oof/ into oof/pub_*.npy + test_preds/pub_*.npy.

Generic scan per source directory:
  oof_X.npy + test_X.npy        -> member <tag>_X
  *.npz with (oof,test)/(train,test) keys of the right length
  *_oof.csv + *_test.csv (id, pred) -> aligned by id
Exact duplicates (OOF correlation > 0.999995 with an earlier member) and stacks/blends are dropped.
Members keep their authors' names in the tag; all are credited in the report.
"""
import glob
import os
import shutil

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from config import DATA_DIR, OOF_DIR, TARGET, TEST_PRED_DIR
from data_loading import load_raw

B = DATA_DIR / "public_oof"
# order matters: the first copy of a duplicate wins
SOURCES = [
    ("s6e11-oof-predictions", "ac"),            # alexchilton, 28 members with a manifest
    ("s6e11-fraud-series-artifacts", "gp"),     # goodpjw2008, X_oof.npy / X_test.npy pairs
    ("wguesdon-tree-ensemble", "wg"),           # wguesdon, npz archives
    ("s6e11-cross-fitted-artifacts", "md"),     # malyshevdanil, per-fold npz (merged by the npz scan if keyed oof/test)
    ("s6e11-saved-study-records", "mk"),        # masayakawamata, <member>/oof.npy + <member>/test.npy
    ("lazyyeou-bank", "lz"),                    # notebook output of lazyyeou: <m>_oof.npy + <m>_full.npy (refit)
    ("nilanjan-96key", "ns"),                   # notebook output of Nilanjan_Saha123: <m>_oof.npy + <m>_full.npy
]
# level-2 outputs / feature matrices are not members; "__s<k>" are single seeds whose bag is also published
EXCLUDE = ("stack", "blend", "knn_feats", "__s")
MIN_AUC = 0.93  # S6E11 single models sit around 0.945-0.950

train, test = load_raw()
y = train[TARGET].values
N_TR, N_TE = len(train), len(test)
TR_ID, TE_ID = train["id"].values, test["id"].values

kept_names, kept_oof = [], []


def ok(o, t):
    return o.shape == (N_TR,) and t.shape == (N_TE,) and np.isfinite(o).all() and np.isfinite(t).all()


def add(name, o, t):
    o, t = np.asarray(o, dtype="float32").ravel(), np.asarray(t, dtype="float32").ravel()
    if any(x in name.lower() for x in EXCLUDE) or not ok(o, t):
        return
    if o.min() < 0 or o.max() > 1:  # logits -> probabilities
        o, t = 1 / (1 + np.exp(-o)), 1 / (1 + np.exp(-t))
    auc = roc_auc_score(y, o)
    if auc < MIN_AUC:
        print(f"  drop weak {name} auc={auc:.4f}")
        return
    if kept_oof:
        corr = np.corrcoef(np.vstack([np.stack(kept_oof[-400:]), o[None]]))[-1, :-1]
        if corr.max() > 0.999995:
            print(f"  dup {name} == {kept_names[-400:][int(corr.argmax())]}")
            return
    kept_names.append(name)
    kept_oof.append(o)
    np.save(OOF_DIR / f"pub_{name}.npy", o)
    np.save(TEST_PRED_DIR / f"pub_{name}.npy", t)


def scan(d, tag):
    # alexchilton ships hash-named files plus a manifest {hash: {"name": ...}}: use the readable name
    # so that family detection (realmlp/tabm/...) and the report credits work
    names = {}
    if (d / "oof_manifest.json").exists():
        import json
        names = {k: v["name"].lower().replace(" ", "_").replace("+", "plus") for k, v in json.load(open(d / "oof_manifest.json")).items()}
    for f in sorted(glob.glob(str(d / "oof_*.npy"))):
        x = os.path.basename(f)[4:-4]
        tf = d / f"test_{x}.npy"
        if tf.exists():
            add(f"{tag}_{names.get(x, x)}", np.load(f), np.load(tf))
    # <member>_oof.npy + <member>_test.npy (or _full.npy = full-data refit), possibly in a preds/ subfolder
    for f in sorted(glob.glob(str(d / "**" / "*_oof.npy"), recursive=True)):
        x, fd = os.path.basename(f)[:-8], os.path.dirname(f)
        tf = os.path.join(fd, f"{x}_test.npy")
        tf = tf if os.path.exists(tf) else os.path.join(fd, f"{x}_full.npy")
        if os.path.exists(tf):
            add(f"{tag}_{x}", np.load(f), np.load(tf))
    # masayakawamata: one directory per member holding oof.npy + test.npy (name = directory name)
    for f in sorted(glob.glob(str(d / "**" / "oof.npy"), recursive=True)):
        tf = os.path.join(os.path.dirname(f), "test.npy")
        if os.path.exists(tf):
            add(f"{tag}_{os.path.basename(os.path.dirname(f))}", np.load(f), np.load(tf))
    # malyshevdanil: one npz per fold with valid_index / valid / test -> assemble OOF, average the 5 test arrays
    fold_files = sorted(glob.glob(str(d / "**" / "*_fold0.npz"), recursive=True))
    for f0 in fold_files:
        x = os.path.basename(f0)[:-10]
        parts = [f0.replace("_fold0.npz", f"_fold{k}.npz") for k in range(5)]
        if not all(os.path.exists(p) for p in parts):
            continue
        o, t = np.full(N_TR, np.nan, dtype="float32"), np.zeros(N_TE, dtype="float32")
        for p in parts:
            z = np.load(p)
            o[z["valid_index"]] = z["valid"]
            t += z["test"] / 5
        add(f"{tag}_{x}", o, t)
    for f in sorted(glob.glob(str(d / "*.npz"))):
        base = os.path.basename(f)[:-4]
        if base.endswith("_test"):
            continue  # handled together with its *_oof.npz twin below
        z = np.load(f)
        # wguesdon: oof (or seed_oof with several seeds) + fold_test (5, n_test) -> average the fold models
        if "fold_test" in z.files and ("oof" in z.files or "seed_oof" in z.files):
            o = z["oof"] if "oof" in z.files else z["seed_oof"].mean(axis=1)
            add(f"{tag}_{base}", o, z["fold_test"].mean(axis=0))
            continue
        keys = ("oof", "test") if "oof" in z.files else ("train", "test") if "train" in z.files else None
        if keys and z[keys[0]].ndim == 1:
            add(f"{tag}_{base}", z[keys[0]], z[keys[1]])
            continue
        # one archive holding many members as oof_<x> / test_<x> pairs (goodpjw2008 series_*.npz)
        for k in z.files:
            if k.startswith("oof_") and f"test_{k[4:]}" in z.files and z[k].ndim == 1:
                add(f"{tag}_{base}_{k[4:]}", z[k], z[f"test_{k[4:]}"])
        # twin archives X_oof.npz / X_test.npz with the same member keys (goodpjw2008 stack182_lr_lgb)
        if base.endswith("_oof") and os.path.exists(f[:-8] + "_test.npz"):
            zt = np.load(f[:-8] + "_test.npz")
            for k in z.files:
                if k in zt.files and z[k].ndim == 1:
                    add(f"{tag}_{base[:-4]}_{k}", z[k], zt[k])
    for f in sorted(glob.glob(str(d / "*_oof.csv"))):
        tf = str(f).replace("_oof.csv", "_test.csv")
        if os.path.exists(tf):
            o, t = pd.read_csv(f).set_index("id"), pd.read_csv(tf).set_index("id")
            col = [c for c in o.columns if c not in ("fold",)][0]
            add(f"{tag}_{os.path.basename(f)[:-8]}", o[col].loc[TR_ID].values, t[col].loc[TE_ID].values)


for f in list(OOF_DIR.glob("pub_*.npy")) + list(TEST_PRED_DIR.glob("pub_*.npy")):
    f.unlink()
for sub, tag in SOURCES:
    before = len(kept_names)
    scan(B / sub, tag)
    print(f"{sub}: +{len(kept_names) - before}")
print(f"total members kept: {len(kept_names)}")
