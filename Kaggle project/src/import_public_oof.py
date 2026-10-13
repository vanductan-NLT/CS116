"""Convert public OOF/test prediction libraries (downloaded into data/public_oof/) into oof/pub_*.npy.

All sources use the shared split StratifiedKFold(5, shuffle=True, random_state=42) except busyaprime
(seed 33 – still out-of-fold). Every member is credited to its Kaggle author in the report.
Format handling follows kratosyan's "S6E10 | Stacking every public OOF library" notebook.
"""
import glob
import os

import numpy as np
import pandas as pd

from config import DATA_DIR, OOF_DIR, TEST_PRED_DIR
from data_loading import load_raw

B = DATA_DIR / "public_oof"
train, test = load_raw()
TR_ID, TE_ID = train["id"].values, test["id"].values
saved = []


def add(name, o, t):
    o, t = np.asarray(o, float), np.asarray(t, float)
    if o.shape != (len(TR_ID),) or t.shape != (len(TE_ID),) or not (np.isfinite(o).all() and np.isfinite(t).all()):
        print("  skip (bad shape/NaN):", name)
        return
    np.save(OOF_DIR / f"pub_{name}.npy", o)
    np.save(TEST_PRED_DIR / f"pub_{name}.npy", t)
    saved.append(name)


def by_id(df, ids, col):
    return df.set_index("id")[col].loc[ids].values


def source(fn):
    """Run one loader; a missing/changed source must not break the others."""
    try:
        fn()
    except Exception as e:  # noqa: BLE001 - public files change between versions
        print(f"  source {fn.__name__} failed: {e!r}")
    return fn


@source
def megayak_library():
    d = B / "s6e10-oof-library"
    mo, mt = pd.read_parquet(d / "oof_predictions.parquet"), pd.read_parquet(d / "test_predictions.parquet")
    groups = {}
    for c in mo.columns:
        if c not in ("id", "fold", "satisfaction"):
            groups.setdefault(c.rsplit("_s", 1)[0] if c.startswith("realmlp") else c, []).append(c)
    for g, cols in groups.items():
        add("mg_" + g, by_id(mo.assign(_m=mo[cols].mean(1)), TR_ID, "_m"), by_id(mt.assign(_m=mt[cols].mean(1)), TE_ID, "_m"))


@source
def golem():
    d = B / "s6e10-golem-oof-library"
    for f in sorted(glob.glob(str(d / "oof_*.npy"))):
        k = os.path.basename(f)[4:-4]
        add("gl_" + k, np.load(f), np.load(d / f"test_{k}.npy"))


@source
def arhancanli():
    d = B / "s6e10-oof-test-predictions"
    ao, at = pd.read_parquet(d / "oof_predictions.parquet"), pd.read_parquet(d / "test_predictions.parquet")
    for c in [c for c in ao.columns if c not in ("id", "fold", "satisfaction")]:
        add("ar_" + c, by_id(ao, TR_ID, c), by_id(at, TE_ID, c))


@source
def sachith():
    for f in sorted(glob.glob(str(B / "s6e10-stack-oof-predictions" / "*.npz"))):
        z = np.load(f)
        add("sa_" + os.path.basename(f)[:-4], z["oof"], z["test"])


@source
def busyaprime():
    for f in sorted(glob.glob(str(B / "s6e10-busyaprime-five-members-oof" / "bp_*.npz"))):
        z = np.load(f)
        add(os.path.basename(f)[:-4], z["oof"], z["test"])


@source
def najiama():
    d = B / "s6e10-oof"
    add("nj_01", by_id(pd.read_csv(d / "01_blend_oof.csv"), TR_ID, "satisfaction"),
        by_id(pd.read_csv(d / "01_submission.csv"), TE_ID, "satisfaction"))


@source
def tabpfn_outputs():
    for nm, slug in [("pfn_samanyu_route_og", "s6e10-tabpfn-route-og"), ("pfn_samanyu_seeds", "s6e10-tabpfn-seeds"),
                     ("pfn_hemingweb_raw", "s6e10-exp01-tabpfn35-full-context")]:
        d = B / slug
        add(nm, by_id(pd.read_csv(d / "oof_exp01_tabpfn35.csv"), TR_ID, "pred"),
            by_id(pd.read_csv(d / "test_exp01_tabpfn35.csv"), TE_ID, "satisfaction"))


@source
def mitudru():
    d = B / "s6e10-analysis-first-equality-blocks-stack"
    mo = pd.read_parquet(d / "oof_predictions.parquet").set_index("id").loc[TR_ID]
    mt = pd.read_parquet(d / "test_predictions.parquet").set_index("id").loc[TE_ID]
    for c in mt.columns:
        add("mi_" + c, mo[c].values, mt[c].values)


@source
def wangxintong():
    for slug, c in [("s6e10-catboost-rating-crosses-cv-0-96080", "rating_segments"),
                    ("s6e10-lightgbm-context-stats-cv-0-96065", "context_numeric")]:
        d = B / slug
        add("wx_" + c, by_id(pd.read_parquet(d / "oof_predictions.parquet"), TR_ID, c),
            by_id(pd.read_parquet(d / "test_predictions.parquet"), TE_ID, c))


@source
def member_csvs():
    for slug, tag in [("s6e10-xgb-realmlp-catboost-lb-0-96123", "th"), ("s6e10-honest-oof-stack", "mh")]:
        d = B / slug
        o = pd.read_csv(d / "oof_members.csv").set_index("id").loc[TR_ID]
        t = pd.read_csv(d / "test_members.csv").set_index("id").loc[TE_ID]
        for c in t.columns:
            add(f"{tag}_{c}", o[c].values, t[c].values)


print(f"saved {len(saved)} public members")
