"""Feature builders.

Two kinds of features:
- `build_static_features`: label-free transforms computed on train+test together (no leakage).
- `add_target_encoding`: label-based encodings that MUST be fit inside each CV fold.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

from config import CAT_COLS, RATING_COLS, TARGET

FD = "Flight Distance"


def build_static_features(train: pd.DataFrame, test: pd.DataFrame, groups: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return feature frames for train/test. `groups` selects optional feature families."""
    full = pd.concat([train.drop(columns=[TARGET]), test], ignore_index=True)
    X = full.drop(columns=["id"]).copy()
    for c in CAT_COLS:  # tiny-cardinality categoricals -> integer codes
        X[c] = X[c].astype("category").cat.codes.astype("int8")
    X["Arrival Delay in Minutes"] = X["Arrival Delay in Minutes"].fillna(X["Departure Delay in Minutes"])

    if "aux" in groups:  # must use only the 21 base columns -> compute before adding anything else
        from aux_task_features import build_aux_task_features
        aux = build_aux_task_features(X.copy())

    if "delay" in groups:
        dep, arr = X["Departure Delay in Minutes"], X["Arrival Delay in Minutes"]
        X["delay_diff"] = arr - dep
        X["delay_total_log"] = np.log1p(dep + arr)
        X["has_delay"] = ((dep + arr) > 0).astype("int8")

    if "rating_stats" in groups:
        R = X[RATING_COLS]
        X["rating_mean"] = R.replace(0, np.nan).mean(axis=1)
        X["rating_min"] = R.min(axis=1)
        X["rating_std"] = R.std(axis=1)
        X["n_zero_ratings"] = (R == 0).sum(axis=1)

    if "fd_freq" in groups:  # Flight Distance repeats like a route id
        X["fd_count"] = X[FD].map(X[FD].value_counts()).astype("int32")

    if "route_profile" in groups:  # label-free profile of each "route" (distance value)
        for c in ["Online boarding", "Inflight wifi service", "Class", "Type of Travel", "Age"]:
            X[f"fd_mean_{c}"] = X.groupby(FD)[c].transform("mean").astype("float32")
            X[f"dev_{c}"] = (X[c] - X[f"fd_mean_{c}"]).astype("float32")

    if "cluster_pca" in groups:  # unsupervised view of the 13 ratings (syllabus: KMeans + PCA)
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        R = StandardScaler().fit_transform(X[RATING_COLS].values.astype("float32"))
        pcs = PCA(n_components=4, random_state=0).fit_transform(R)
        for j in range(4):
            X[f"pca_{j}"] = pcs[:, j].astype("float32")
        km = KMeans(n_clusters=12, n_init=3, random_state=0).fit(R[::5])
        X["kmeans_12"] = km.predict(R).astype("int8")
        X["kmeans_dist"] = np.min(km.transform(R), axis=1).astype("float32")

    if "llr" in groups:  # generator-surrogate log-likelihood ratio from distilgpt2 fine-tuned on the ORIGINAL data only
        from config import DATA_DIR
        llr = np.concatenate([np.load(DATA_DIR / "llr_train.npy"), np.load(DATA_DIR / "llr_test.npy")]).astype("float32")
        X["llr"] = llr
        X["llr_clip"] = np.clip(llr, -20, 20)

    if "tuple_keys" in groups:  # hashed discrete tuples (49% of rows share one) for in-fold target encoding
        X["tuple17"] = pd.util.hash_pandas_object(full[RATING_COLS + CAT_COLS].astype(str), index=False).values.astype("int64")
        X["tuple_ratings"] = pd.util.hash_pandas_object(full[RATING_COLS].astype(str), index=False).values.astype("int64")
        X["tuple17_count"] = X["tuple17"].map(X["tuple17"].value_counts()).astype("int32")

    if "bpe" in groups:  # generator-aware: group numeric values by their GPT-2 BPE tokenisation (S6E9 3rd place)
        from tokenizers import Tokenizer
        tok = Tokenizer.from_pretrained("gpt2")
        for c, tag in [(FD, "fd"), ("Age", "age"), ("Departure Delay in Minutes", "dep")]:
            vals = full[c].fillna(-1).astype(int)
            enc = {v: tok.encode(f" {v}").tokens for v in vals.unique()}
            first = vals.map(lambda v: enc[v][0]).astype("category").cat.codes.astype("int16")
            last = vals.map(lambda v: enc[v][-1]).astype("category").cat.codes.astype("int16")
            shape = vals.map(lambda v: "-".join(str(len(t.lstrip("Ġ"))) for t in enc[v])).astype("category").cat.codes
            X[f"bpe_{tag}_first"] = first
            X[f"bpe_{tag}_last"] = last
            X[f"bpe_{tag}_shape"] = shape.astype("int16")
            X[f"bpe_{tag}_first_x_shape"] = (first.astype("int32") * 64 + shape.astype("int32")).astype("int32")

    if "fd_combo" in groups:  # integer keys meant for in-fold target encoding (route x context)
        X["fd_x_class"] = (X[FD] * 3 + X["Class"]).astype("int32")
        X["fd_x_travel"] = (X[FD] * 2 + X["Type of Travel"]).astype("int32")
        X["fd_x_cust"] = (X[FD] * 2 + X["Customer Type"]).astype("int32")

    if "route_full" in groups:  # per-distance mean of EVERY other column + residual (community: +0.0012)
        base_cols = [c for c in full.columns if c not in ("id", FD)]
        g = X.groupby(FD)
        for c in base_cols:
            m = g[c].transform("mean").astype("float32")
            X[f"rf_mean_{c}"] = m
            X[f"rf_res_{c}"] = (X[c] - m).astype("float32")
        X["rf_count"] = g[FD].transform("size").astype("int32")

    if "original" in groups:  # teacher + lookups from the public original dataset
        from original_data_features import build_original_features
        X = pd.concat([X, build_original_features(train, test)], axis=1)

    if "aux" in groups:
        X = pd.concat([X, aux], axis=1)

    if "cat_cross" in groups:  # joint code of the 4 categoricals (24 combos)
        X["cat_cross"] = (X["Gender"] * 12 + X["Customer Type"] * 6 + X["Type of Travel"] * 3 + X["Class"]).astype("int16")

    n = len(train)
    return X.iloc[:n].reset_index(drop=True), X.iloc[n:].reset_index(drop=True)


def _smoothed_mean(keys: pd.Series, y: np.ndarray, smoothing: float) -> pd.Series:
    """Bayesian-smoothed target mean per key."""
    prior = y.mean()
    stats = pd.DataFrame({"k": keys.values, "y": y}).groupby("k")["y"].agg(["sum", "count"])
    return (stats["sum"] + prior * smoothing) / (stats["count"] + smoothing)


def add_target_encoding(X_tr, y_tr, X_va, X_te, cols, smoothing=20.0, inner_folds=5, seed=0):
    """Target-encode `cols` without leakage.

    Train rows get inner out-of-fold encodings; valid/test rows get encodings fit on all train rows.
    Returns new copies of (X_tr, X_va, X_te) with `te_<col>` columns appended.
    """
    X_tr, X_va, X_te = X_tr.copy(), X_va.copy(), X_te.copy()
    prior = y_tr.mean()
    soft = not np.all((y_tr == 0) | (y_tr == 1))  # pseudo-labelled rows carry probabilities
    skf = KFold(inner_folds, shuffle=True, random_state=seed) if soft else StratifiedKFold(inner_folds, shuffle=True, random_state=seed)
    for c in cols:
        name = f"te_{c}"
        tr_enc = np.full(len(X_tr), prior, dtype="float32")
        for a, b in skf.split(X_tr, y_tr):
            m = _smoothed_mean(X_tr[c].iloc[a], y_tr[a], smoothing)
            tr_enc[b] = X_tr[c].iloc[b].map(m).fillna(prior).values
        full_map = _smoothed_mean(X_tr[c], y_tr, smoothing)
        X_tr[name] = tr_enc
        X_va[name] = X_va[c].map(full_map).fillna(prior).astype("float32").values
        X_te[name] = X_te[c].map(full_map).fillna(prior).astype("float32").values
    return X_tr, X_va, X_te
