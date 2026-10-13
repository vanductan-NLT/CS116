"""Kaggle script kernel: fine-tune TabPFN-3.5 on one outer fold of the shared split and save OOF/test preds.

Run once per FOLD (0..4) on a Kaggle GPU (P100 16 GB / T4). Requires:
  - Kaggle secret TABPFN_TOKEN (Prior Labs API key, license accepted) attached to the kernel
  - internet enabled (pip install + weight download)
Outputs: oof_fold{F}.npy (prob for the fold's validation rows, train order ids), test_fold{F}.npy
"""
import os
import subprocess
import sys
import time

FOLDS = [int(f) for f in os.environ.get("FOLDS", "0,1,2,3,4").split(",")]
TIME_LIMIT = int(os.environ.get("TIME_LIMIT", "2700"))     # seconds of fine-tuning per fold (5 folds < 12 h session)
CTX_SAMPLES = int(os.environ.get("CTX_SAMPLES", "24000"))  # context+query rows per step (16 GB budget)
INFER_CTX = int(os.environ.get("INFER_CTX", "50000"))      # context rows at inference (halved on OOM)

try:
    from kaggle_secrets import UserSecretsClient
    token = UserSecretsClient().get_secret("TABPFN_TOKEN").strip()
except Exception as e:  # noqa: BLE001
    sys.exit(f"Kaggle secret TABPFN_TOKEN is not attached to this kernel ({e!r}); attach it and re-run.")
# expose the key both ways the library looks for it: env var and the cached token file
os.environ["TABPFN_TOKEN"] = token
from pathlib import Path  # noqa: E402
tok_file = Path.home() / ".cache" / "tabpfn" / "auth_token"
tok_file.parent.mkdir(parents=True, exist_ok=True)
tok_file.write_text(token)
print(f"token length {len(token)} written to {tok_file}", flush=True)
os.environ["TABPFN_NO_BROWSER"] = "1"
os.environ["TABPFN_MAX_BATCHED_TEST_ROWS"] = "4096"  # query rows per forward pass during validation/inference
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tabpfn>=9.2.0"], check=True)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from tabpfn.finetuning import FinetunedTabPFNClassifier  # noqa: E402

# Diagnose the license gate explicitly so a failure names its cause in the log
import tabpfn.browser_auth as _auth  # noqa: E402
from tabpfn.settings import settings as _settings  # noqa: E402

_api = _settings.tabpfn.auth_api_url
print("verify_token:", _auth.verify_token(token, _api),
      "| license 3.5 accepted:", _auth.check_license_accepted(token, _api, _auth._get_license_name("tabpfn_3_5")), flush=True)

import glob  # noqa: E402

# Kaggle mounts inputs either at /kaggle/input/<slug> or /kaggle/input/competitions/<slug>
hits = [p for p in glob.glob("/kaggle/input/**/playground-series-s6e10/train.csv", recursive=True)]
if not hits:
    sys.exit("Competition data not attached: in the editor use 'Add Input' -> Competitions -> Predicting Airline Satisfaction.")
D = os.path.dirname(hits[0])
train, test = pd.read_csv(f"{D}/train.csv"), pd.read_csv(f"{D}/test.csv")
CATS = ["Gender", "Customer Type", "Type of Travel", "Class"]
y = train["satisfaction"].astype(int).values


def build(df):
    X = df.drop(columns=["id", "satisfaction"], errors="ignore").copy()
    for c in CATS:
        X[c] = X[c].astype("category").cat.codes
    X["Arrival Delay in Minutes"] = X["Arrival Delay in Minutes"].fillna(-1)
    X["fd10"] = X["Flight Distance"] // 10
    return X.values.astype(np.float32)


X, X_te = build(train), build(test)
folds = np.zeros(len(X), dtype=int)
for k, (_, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y)):
    folds[va] = k
def predict(clf, Xq, chunk=2048):
    """Chunked inference; on CUDA OOM halve the inference context and retry."""
    while True:
        try:
            return np.concatenate([clf.predict_proba(Xq[i:i + chunk])[:, 1] for i in range(0, len(Xq), chunk)])
        except Exception as e:  # noqa: BLE001
            if "out of memory" not in str(e).lower() or clf.n_inference_subsample_samples <= 20000:
                raise
            clf.n_inference_subsample_samples //= 2
            torch.cuda.empty_cache()
            print(f"OOM -> inference context reduced to {clf.n_inference_subsample_samples}", flush=True)


for FOLD in FOLDS:
    tr, va = np.where(folds != FOLD)[0], np.where(folds == FOLD)[0]
    t0 = time.time()
    # No per-epoch validation: inference while the optimiser state is resident is what OOMs on small GPUs.
    # The time limit is the stopping rule; the honest score is computed below on the held-out fold.
    clf = FinetunedTabPFNClassifier(device="cuda", epochs=100, time_limit=TIME_LIMIT, learning_rate=1e-5,
                                    validation_split_ratio=None, early_stopping=False,
                                    n_finetune_ctx_plus_query_samples=CTX_SAMPLES, n_inference_subsample_samples=INFER_CTX,
                                    n_estimators_finetune=1, n_estimators_validation=1, n_estimators_final_inference=2,
                                    random_state=FOLD, use_activation_checkpointing=True, save_checkpoint_interval=None)
    clf.fit(X[tr], y[tr])
    torch.cuda.empty_cache()
    print(f"fold {FOLD}: fine-tuned in {time.time() - t0:.0f}s, peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.1f} GB",
          flush=True)
    oof = predict(clf, X[va])
    print(f"fold {FOLD}: AUC={roc_auc_score(y[va], oof):.6f} ({time.time() - t0:.0f}s)", flush=True)
    np.save(f"oof_fold{FOLD}.npy", oof)
    np.save(f"oof_idx_fold{FOLD}.npy", va)
    np.save(f"test_fold{FOLD}.npy", predict(clf, X_te))
    print(f"fold {FOLD} done in {time.time() - t0:.0f}s", flush=True)
    del clf
    torch.cuda.empty_cache()
