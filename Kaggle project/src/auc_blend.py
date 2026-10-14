"""Level-3 AUC-direct blend (S6E9 2nd place, Team Alicia): non-negative weights over members' rank-logits that
minimise a smoothed pairwise-AUC loss, evaluated exactly over all pos/neg pairs with histograms + FFT.

    L(w) = mean over (pos i, neg j) of sigmoid(-(s_i - s_j) / tau),   s = X w,  w >= 0, sum w = 1
Scored with the same nested meta folds (seed 7) as stack_oof.py so it is comparable to the LR stack.

Example: python src/auc_blend.py --members all --exclude base_lr,base_knn,base_rf,base_lgbm_default,xgb_v1_pl,teacher5 --out auc_blend_v1
"""
import argparse
import time

import numpy as np
from scipy.optimize import minimize
from scipy.signal import fftconvolve
from scipy.special import expit
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from config import SUB_DIR, TARGET
from data_loading import load_raw, write_submission
from stack_oof import load_members, logit

TAU, H, LO, N = 0.1, 0.0005, -16.0, 64001
LAG = np.arange(-N + 1, N) * H
K_SIG = expit(-LAG / TAU)                      # sigma(-lag/tau), lag = s_pos - s_neg
K_DSIG = K_SIG * (1 - K_SIG)                   # sigma'(-lag/tau)


def rank_logit(v):
    return logit((rankdata(v) - 0.5) / len(v))


def hist(s):
    """Linear-interpolation histogram on the grid (sums to 1) and the per-row (bin, frac) for the gradient."""
    q = np.clip((s - LO) / H, 0, N - 1 - 1e-9)
    lo = np.floor(q).astype(int)
    fr = q - lo
    h = np.bincount(lo, weights=1 - fr, minlength=N + 1) + np.bincount(lo + 1, weights=fr, minlength=N + 1)
    return h[:N] / len(s), lo, fr


def loss_and_grad(w, X, y):
    s = X @ w
    sp, sn = s[y == 1], s[y == 0]
    hp, lop, frp = hist(sp)
    hn, lon, frn = hist(sn)
    F = fftconvolve(hn, K_SIG)[N - 1:2 * N - 1]        # F(a) = sum_b hn(b) sigma(-(g_a - g_b)/tau)
    loss = hp @ F
    # dL/ds_i for positives: -(1/(tau*n+)) * G(s_i),  G = conv(hn, sigma') ;  negatives: +(1/(tau*n-)) * G2(s_j)
    G = fftconvolve(hn, K_DSIG)[N - 1:2 * N - 1]
    G2 = fftconvolve(hp, K_DSIG[::-1])[N - 1:2 * N - 1]
    ds = np.zeros(len(s))
    ds[y == 1] = -(G[lop] * (1 - frp) + G[lop + 1] * frp) / (TAU * len(sp))
    ds[y == 0] = (G2[lon] * (1 - frn) + G2[lon + 1] * frn) / (TAU * len(sn))
    return loss, X.T @ ds


def fit(X, y, w0):
    res = minimize(loss_and_grad, w0 / w0.sum(), args=(X, y), jac=True, method="SLSQP",
                   bounds=[(0, 1)] * X.shape[1], constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
                   options={"maxiter": 200, "ftol": 1e-12})
    return res.x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", required=True)
    ap.add_argument("--exclude", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    train, test = load_raw()
    y = train[TARGET].values
    names, oof, tst = load_members(a.members.split(","), [e for e in a.exclude.split(",") if e])
    X = np.column_stack([rank_logit(oof[:, i]) for i in range(oof.shape[1])])
    Xt = np.column_stack([rank_logit(tst[:, i]) for i in range(tst.shape[1])])
    Z = logit(oof)
    print(f"{len(names)} members", flush=True)

    skf = list(StratifiedKFold(5, shuffle=True, random_state=7).split(X, y))
    meta_lr, meta_auc = np.zeros(len(y)), np.zeros(len(y))
    t0 = time.time()
    for k, (tr, va) in enumerate(skf):
        lr = LogisticRegression(C=0.003, max_iter=5000).fit(Z[tr], y[tr])
        meta_lr[va] = lr.decision_function(Z[va])
        w0 = np.clip(lr.coef_[0], 1e-6, None)        # warm start: non-negative part of the LR weights
        w = fit(X[tr], y[tr], w0)
        meta_auc[va] = X[va] @ w
        print(f"fold {k}: LR {roc_auc_score(y[va], meta_lr[va]):.6f}  AUC-blend {roc_auc_score(y[va], meta_auc[va]):.6f}  "
              f"({time.time() - t0:.0f}s, {int((w > 1e-4).sum())} active)", flush=True)
    print(f"nested: LR stack {roc_auc_score(y, meta_lr):.6f} | AUC-direct blend {roc_auc_score(y, meta_auc):.6f}")
    if a.out:
        lr = LogisticRegression(C=0.003, max_iter=5000).fit(Z, y)
        w = fit(X, y, np.clip(lr.coef_[0], 1e-6, None))
        pred = Xt @ w
        write_submission(test, (pred - pred.min()) / (pred.max() - pred.min()), SUB_DIR / f"{a.out}.csv")
        print("wrote", SUB_DIR / f"{a.out}.csv")


if __name__ == "__main__":
    main()
