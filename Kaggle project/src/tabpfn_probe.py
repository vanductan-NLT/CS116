"""VRAM probe for TabPFN-3.5 on this GPU: how large a context fits for fit + chunked predict."""
import argparse
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

import tabpfn.architectures.shared.scaled_dot_product_attention as _sdpa
from torch.nn.attention import SDPBackend

_sdpa._SDPA_BACKENDS[:] = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION]  # Windows: avoid MATH kernel
from tabpfn import TabPFNClassifier  # noqa: E402

from config import TARGET  # noqa: E402
from data_loading import get_folds, load_raw  # noqa: E402
from tabpfn_finetune import build_matrix, predict_chunked  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ctx", type=int, default=30000)
ap.add_argument("--n-val", type=int, default=20000)
ap.add_argument("--n-est", type=int, default=1)
a = ap.parse_args()

torch.cuda.set_per_process_memory_fraction(0.9)  # never spill into Windows shared memory
train, _ = load_raw()
y = train[TARGET].values
X, cat_idx = build_matrix(train)
folds = get_folds(train)
tr, va = np.where(folds != 0)[0], np.where(folds == 0)[0]
rng = np.random.default_rng(0)
ctx, val = rng.choice(tr, a.ctx, replace=False), rng.choice(va, a.n_val, replace=False)

t0 = time.time()
clf = TabPFNClassifier(n_estimators=a.n_est, device="cuda", ignore_pretraining_limits=True, memory_saving_mode=True,
                       categorical_features_indices=cat_idx, inference_precision=torch.float16, random_state=0)
clf.fit(X[ctx], y[ctx])
t1 = time.time()
p = predict_chunked(clf, X[val], chunk=2048)
print(f"ctx={a.ctx} n_est={a.n_est}: AUC {roc_auc_score(y[val], p):.5f} | fit {t1 - t0:.0f}s, predict {time.time() - t1:.0f}s "
      f"({a.n_val / (time.time() - t1):.0f} rows/s) | peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")
