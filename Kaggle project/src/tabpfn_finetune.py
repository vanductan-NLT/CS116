"""Fine-tune TabPFN-3.5 per outer fold on the shared split and save OOF/test predictions (member `tabpfn_ft`).

Why: every public TabPFN member is in-context only (no weight updates). Fine-tuning on ~560k labelled rows is
the one direction nobody reported; if it generalises, it is a new "family" for the stack.
Constraints: RTX 4050 6 GB -> small fine-tuning contexts, subsampled inference context, Windows SDPA fix.

Example (feasibility, fold 0 only, 10-minute budget):
  python src/tabpfn_finetune.py --only-folds 0 --time-limit 600 --n-val 30000 --name tabpfn_ft_probe
Full run:
  python src/tabpfn_finetune.py --time-limit 2400 --name tabpfn_ft
"""
import argparse
import os
import time
from pathlib import Path

# validation/final inference predicts this many query rows per forward pass (default 32768 blows up VRAM)
os.environ.setdefault("TABPFN_MAX_BATCHED_TEST_ROWS", "2048")

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from config import CAT_COLS, LOG_DIR, OOF_DIR, TARGET, TEST_PRED_DIR
from data_loading import get_folds, load_raw

# Windows torch wheels lack FlashAttention; without this TabPFN falls back to the MATH kernel (~90x slower).
import tabpfn.architectures.shared.scaled_dot_product_attention as _sdpa  # noqa: E402
from torch.nn.attention import SDPBackend  # noqa: E402

_sdpa._SDPA_BACKENDS[:] = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION]

from tabpfn import TabPFNClassifier  # noqa: E402
from tabpfn.finetuning import FinetunedTabPFNClassifier  # noqa: E402


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="tabpfn_ft")
    p.add_argument("--only-folds", default="")
    p.add_argument("--time-limit", type=int, default=2400, help="seconds of fine-tuning per fold")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--ctx-samples", type=int, default=16000, help="context+query rows per fine-tuning step")
    p.add_argument("--infer-ctx", type=int, default=120000, help="context rows at inference (VRAM bound)")
    p.add_argument("--n-est", type=int, default=2, help="estimators at final inference")
    p.add_argument("--n-val", type=int, default=0, help="probe mode: score only this many validation rows")
    p.add_argument("--baseline", action="store_true", help="also score the non-finetuned model on the same rows")
    return p.parse_args()


def build_matrix(df: pd.DataFrame):
    """Raw 21 columns + FD//10; Flight Distance and FD//10 declared categorical (community's best TabPFN setup)."""
    X = df.drop(columns=["id", TARGET], errors="ignore").copy()
    for c in CAT_COLS:
        X[c] = X[c].astype("category").cat.codes
    X["Arrival Delay in Minutes"] = X["Arrival Delay in Minutes"].fillna(-1)
    X["fd10"] = X["Flight Distance"] // 10
    cat_idx = [X.columns.get_loc(c) for c in CAT_COLS + ["Flight Distance", "fd10"]]
    return X.values.astype(np.float32), cat_idx


def predict_chunked(clf, X, chunk=1024):
    return np.concatenate([clf.predict_proba(X[i:i + chunk])[:, 1] for i in range(0, len(X), chunk)])


def main():
    a = parse()
    train, test = load_raw()
    y = train[TARGET].values
    X, cat_idx = build_matrix(train)
    X_te, _ = build_matrix(test)
    folds = get_folds(train)
    run_folds = [int(f) for f in a.only_folds.split(",") if f] or list(range(5))
    oof, test_pred = np.full(len(X), np.nan), np.zeros(len(X_te))
    torch.cuda.set_per_process_memory_fraction(0.9)  # never spill into Windows shared memory

    for k in run_folds:
        t0 = time.time()
        tr, va = np.where(folds != k)[0], np.where(folds == k)[0]
        if a.n_val:
            va = np.random.default_rng(k).choice(va, a.n_val, replace=False)
        clf = FinetunedTabPFNClassifier(
            device="cuda", epochs=a.epochs, time_limit=a.time_limit, learning_rate=a.lr,
            n_finetune_ctx_plus_query_samples=a.ctx_samples, n_inference_subsample_samples=a.infer_ctx,
            n_estimators_finetune=1, n_estimators_validation=1, n_estimators_final_inference=a.n_est,
            early_stopping=True, early_stopping_patience=4, random_state=k, use_activation_checkpointing=True,
            save_checkpoint_interval=None,
        )
        # explicit small validation set: early stopping signal without scoring 100k+ rows every epoch
        v = np.random.default_rng(1000 + k).choice(va, min(10000, len(va)), replace=False)
        clf.fit(X[tr], y[tr], X_val=X[v], y_val=y[v], output_dir=Path(LOG_DIR / f"tabpfn_ft_fold{k}"))
        print(f"fold {k}: fine-tuned in {time.time() - t0:.0f}s, peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GB",
              flush=True)
        oof[va] = predict_chunked(clf, X[va])
        print(f"fold {k}: AUC={roc_auc_score(y[va], oof[va]):.6f} (n={len(va)}, {time.time() - t0:.0f}s)", flush=True)
        if not a.n_val:
            test_pred += predict_chunked(clf, X_te) / len(run_folds)

        if a.baseline:  # same inference recipe without fine-tuning, for an apples-to-apples delta
            base = TabPFNClassifier(n_estimators=a.n_est, device="cuda", ignore_pretraining_limits=True,
                                    categorical_features_indices=cat_idx, memory_saving_mode=True,
                                    random_state=k)
            ctx = np.random.default_rng(k).choice(tr, min(a.infer_ctx, len(tr)), replace=False)
            base.fit(X[ctx], y[ctx])
            print(f"fold {k}: baseline (no finetune, ctx={len(ctx)}) AUC={roc_auc_score(y[va], predict_chunked(base, X[va])):.6f}",
                  flush=True)
        del clf
        torch.cuda.empty_cache()

    mask = ~np.isnan(oof)
    print(f"[{a.name}] AUC on scored rows = {roc_auc_score(y[mask], oof[mask]):.6f}")
    if len(run_folds) == 5 and not a.n_val:
        np.save(OOF_DIR / f"{a.name}.npy", oof)
        np.save(TEST_PRED_DIR / f"{a.name}.npy", test_pred)


if __name__ == "__main__":
    main()
