"""VRAM probe for TabPFN-3.5 on this GPU: how large a context fits for fit + chunked predict."""
import argparse
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

import tabpfn.architectures.shared.scaled_dot_product_attention as _sdpa
from torch.nn.attention import SDPBackend

ap = argparse.ArgumentParser()
ap.add_argument("--ctx", type=int, default=30000)
ap.add_argument("--n-val", type=int, default=20000)
ap.add_argument("--n-est", type=int, default=1)
ap.add_argument("--precision", default="auto", choices=["auto", "fp16", "fp32"])
ap.add_argument("--sdpa-patch", action="store_true", help="drop the MATH kernel (Windows speed fix)")
ap.add_argument("--no-cats", action="store_true", help="do not declare categorical columns")
ap.add_argument("--fit-mode", default="fit_preprocessors")
ap.add_argument("--kv", default=None, help="kv_cache_precision for fit_with_cache, e.g. int8 / float16")
ap.add_argument("--chunk", type=int, default=2048, help="query rows per predict call")
ap.add_argument("--bpe", action="store_true", help="add GPT-2 BPE token / digit columns")
ap.add_argument("--route", action="store_true", help="add label-free route-profile columns (per-FD means, residuals, counts)")
a = ap.parse_args()
if a.sdpa_patch:
    _sdpa._SDPA_BACKENDS[:] = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION]
from tabpfn import TabPFNClassifier  # noqa: E402

from config import TARGET  # noqa: E402
from data_loading import get_folds, load_raw  # noqa: E402
from tabpfn_finetune import build_matrix, predict_chunked  # noqa: E402

PREC = {"auto": "auto", "fp16": torch.float16, "fp32": torch.float32}[a.precision]

torch.cuda.set_per_process_memory_fraction(0.9)  # never spill into Windows shared memory
train, _ = load_raw()
y = train[TARGET].values
X, cat_idx = build_matrix(train, bpe=a.bpe)
if a.route:  # label-free transductive columns computed over train+test (same idea as the S6E9 "GAR" columns)
    from feature_engineering import build_static_features
    _, test_df = load_raw()
    R, _ = build_static_features(train, test_df, {"fd_freq", "route_full"})
    R = R[[c for c in R.columns if c.startswith("rf_") or c == "fd_count"]].values.astype(np.float32)
    X = np.hstack([X, R])
folds = get_folds(train)
tr, va = np.where(folds != 0)[0], np.where(folds == 0)[0]
rng = np.random.default_rng(0)
ctx, val = rng.choice(tr, a.ctx, replace=False), rng.choice(va, a.n_val, replace=False)

t0 = time.time()
extra = {"fit_mode": a.fit_mode}
if a.kv:
    extra["kv_cache_precision"] = a.kv
clf = TabPFNClassifier(n_estimators=a.n_est, device="cuda", ignore_pretraining_limits=True, memory_saving_mode=True,
                       categorical_features_indices=None if a.no_cats else cat_idx, inference_precision=PREC, random_state=0,
                       **extra)
clf.fit(X[ctx], y[ctx])
t1 = time.time()
p = predict_chunked(clf, X[val], chunk=a.chunk)
print(f"ctx={a.ctx} n_est={a.n_est} bpe={a.bpe} prec={a.precision} fit_mode={a.fit_mode} kv={a.kv} chunk={a.chunk}: "
      f"AUC {roc_auc_score(y[val], p):.5f} nan={np.isnan(p).sum()} std={np.nanstd(p):.4f} | fit {t1 - t0:.0f}s, "
      f"predict {time.time() - t1:.0f}s ({a.n_val / (time.time() - t1):.0f} rows/s) | "
      f"peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")
