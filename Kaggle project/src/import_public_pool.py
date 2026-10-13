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
    ("s6e10-124-public-oof-and-test-predictions", "cd"),
    ("s6e10-tfm-oof-and-test-predictions", "tfm"),
    ("s6e10-ablation-members-oof", "kk"),
    ("ps-s6e10-airline-satisfaction-artifacts", "st"),
    ("s6e10-forge-own-pairs", "am"),
    ("s6e10-goodpjw2008-extra-members-oof", "gp"),
    ("s6e10-route-tfm-nn-oof", "km"),
    ("oof-dataset", "ab"),
    ("s6e10-catboost-all-categorical-oof-and-test", "dk"),
    ("s6e10-golem-oof-library", "gl"),
    ("s6e10-stack-oof-predictions", "sa"),
    ("s6e10-busyaprime-five-members-oof", "bp"),
]
EXCLUDE = ("stack", "blend", "knn_feats")  # level-2 outputs / feature matrices are not members

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
    if auc < 0.95:
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
    for f in sorted(glob.glob(str(d / "oof_*.npy"))):
        x = os.path.basename(f)[4:-4]
        tf = d / f"test_{x}.npy"
        if tf.exists():
            add(f"{tag}_{x}", np.load(f), np.load(tf))
    for f in sorted(glob.glob(str(d / "*.npz"))):
        z = np.load(f)
        keys = ("oof", "test") if "oof" in z.files else ("train", "test") if "train" in z.files else None
        if keys and z[keys[0]].ndim == 1:
            add(f"{tag}_{os.path.basename(f)[:-4]}", z[keys[0]], z[keys[1]])
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
# sources with bespoke layouts (parquet / member csv) reuse the first importer's loaders
import import_public_oof  # noqa: E402,F401  (runs its loaders -> pub_* files; duplicates are removed next)
for f in sorted(OOF_DIR.glob("pub_*.npy")):
    n = f.stem[4:]
    if n not in kept_names:
        add(n, np.load(f), np.load(TEST_PRED_DIR / f.name))
    if n not in kept_names:  # rejected as duplicate/weak -> remove the files the bespoke loader wrote
        f.unlink()
        (TEST_PRED_DIR / f.name).unlink()
print(f"total members kept: {len(kept_names)}")
