"""Feature engineering for S6E11 (credit-card fraud, synthetic data).

build_static_features(train, test, groups) -> (X, X_test) with label-free features; groups:
  digits    : first digit, last digit, cents and "roundness" of the float columns - synthetic generators leave
              fingerprints in the digits (the trick behind the community's *_dig* members)
  freq      : frequency of each raw value, counted over train+test
  interact  : hand-made ratios / combos (amount vs balance, velocity x txn count, risk x new merchant ...)
  cat_cross : pairwise concatenations of the categorical columns (as new categoricals)
  bool_sum  : number of raised risk flags
add_target_encoding(...) : in-fold target encoding, unchanged from the S6E10 project.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

from config import BOOL_COLS, CAT_COLS, DATA_DIR, FLOAT_COLS, ID_COL, INT_COLS, TARGET

TARGET_DIR_ORIGINAL = DATA_DIR / "original_features.parquet"


def build_static_features(train: pd.DataFrame, test: pd.DataFrame, groups: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_tr = len(train)
    df = pd.concat([train.drop(columns=[TARGET]), test], ignore_index=True)
    X = df.drop(columns=[ID_COL]).copy()
    new_cats = []

    if "digits" in groups:
        for c in FLOAT_COLS:
            v = X[c].astype(float)
            s = v.abs().map(lambda x: f"{x:.2f}")
            X[f"{c}_last_digit"] = s.str[-1].astype("int8")
            X[f"{c}_cents"] = (np.round(v * 100) % 100).astype("int16")
            X[f"{c}_first_digit"] = s.str.lstrip("0.").str[0].fillna("0").astype("int8")
            X[f"{c}_is_round"] = (np.round(v * 100) % 100 == 0).astype("int8")
            X[f"{c}_ndigits"] = s.str.split(".").str[0].str.len().astype("int8")

    if "freq" in groups:
        for c in FLOAT_COLS + INT_COLS:
            X[f"{c}_freq"] = X[c].map(X[c].value_counts()).astype("int32")

    if "interact" in groups:
        X["amount_over_balance"] = X["amount_usd"] / (X["account_balance_usd"].abs() + 1)
        X["amount_log"] = np.log1p(X["amount_usd"].clip(lower=0))
        X["velocity_x_txn"] = X["velocity_score"] * X["txn_count_last_24h"]
        X["risk_x_newmerchant"] = X["merchant_risk_score"] * (1 + X["is_new_merchant"])
        X["dist_over_cardage"] = X["distance_from_home_km"] / (X["card_age_months"] + 1)
        X["hours_x_txn"] = X["hours_since_last_txn"] * X["txn_count_last_24h"]
        X["cvv_x_retry_flag"] = X["cvv_retry_count"] * (1 + X["billing_shipping_mismatch"])
        X["age_minus_cardage_years"] = X["customer_age"] - X["card_age_months"] / 12
        X["night"] = X["time_of_day_hour"].isin([0, 1, 2, 3, 4, 5]).astype("int8")
        X["weekend"] = X["day_of_week"].isin([5, 6]).astype("int8")

    if "cat_cross" in groups:
        for i, a in enumerate(CAT_COLS):
            for b in CAT_COLS[i + 1:]:
                name = f"{a}_x_{b}"
                X[name] = X[a] + "|" + X[b]
                new_cats.append(name)

    if "te_keys" in groups:
        # keys for in-fold target encoding (run_cv --te keys): the generator writes numbers digit by digit, so the
        # fraud rate depends on the digits of a number, not only its size (goodpjw2008 Part 3, Ravi Ramakrishnan)
        for c in FLOAT_COLS:
            v = X[c].astype(float)
            X[f"k_{c}_int"] = np.round(v).astype("int32")
            X[f"k_{c}_r10"] = np.round(v / 10).astype("int32")
            X[f"k_{c}_x10"] = np.round(v * 10).astype("int64")
            X[f"k_{c}_cents"] = (np.round(v * 100) % 100).astype("int16")
            X[f"k_{c}_lead"] = v.abs().map(lambda x: f"{x:.2f}").str.lstrip("0.").str[0].fillna("0").astype("int8")
        top = ["cvv_retry_count", "ip_country_mismatch", "is_foreign_transaction", "is_new_merchant",
               "billing_shipping_mismatch", "auth_method", "used_vpn", "is_ai_generated_scam_attempt",
               "merchant_category", "txn_count_last_24h"]
        for i, a in enumerate(top):
            for b in top[i + 1:]:
                X[f"k_{a}__{b}"] = pd.factorize(X[a].astype(str) + "_" + X[b].astype(str))[0].astype("int32")

    if "te_trie" in groups:
        # "digit-trie" keys (lazyyeou / areebbb recipe): the LM writes each number as a string, so encode its
        # integer part, its fractional part, floor(x/10), floor(10x) and length|leading digits - no pair keys
        # (pair TE measured -0.0007 by two independent notebooks)
        for c in FLOAT_COLS:
            dec = 2 if (np.round(X[c] * 100) % 10 != 0).any() else 1
            s = X[c].map(lambda v, d=dec: f"{v:.{d}f}").str.split(".")
            ip, fp = s.str[0], s.str[1]
            ln = ip.str.len().astype(str)
            X[f"k_{c}_int"] = pd.factorize(ip)[0].astype("int32")
            X[f"k_{c}_frac"] = pd.factorize(fp)[0].astype("int32")
            X[f"k_{c}_ten"] = np.floor(X[c] / 10.0).astype("int64")
            X[f"k_{c}_dec"] = np.floor(X[c] * 10.0 + 1e-6).astype("int64")
            X[f"k_{c}_l1"] = pd.factorize(ln + "|" + ip.str[:1])[0].astype("int32")
            X[f"k_{c}_l2"] = pd.factorize(ln + "|" + ip.str[:2])[0].astype("int32")
            ref = pd.concat([X[c], pd.read_csv(next(DATA_DIR.glob("original/*.csv")))[c]])
            X[f"cnt_{c}"] = X[c].map(ref.value_counts()).fillna(0).astype("float32")

    if "original" in groups:
        # teachers fit on the 20k source rows only (src/original_data_features.py); label-free w.r.t. competition
        orig = pd.read_parquet(TARGET_DIR_ORIGINAL).set_index(ID_COL)
        for c in orig.columns:
            X[c] = orig.loc[df[ID_COL].values, c].values.astype("float32")

    if "bool_sum" in groups:
        X["n_flags"] = X[BOOL_COLS].sum(axis=1).astype("int8")
        X["flags_code"] = sum(X[c].astype(int) * (2 ** i) for i, c in enumerate(BOOL_COLS)).astype("int16")

    # integer codes (fit on train+test, label-free) so every model family accepts the columns;
    # CatModel turns them back into strings and treats them as categorical
    for c in CAT_COLS + new_cats:
        X[c] = X[c].astype("category").cat.codes.astype("int16")
    return X.iloc[:n_tr].reset_index(drop=True), X.iloc[n_tr:].reset_index(drop=True)


def _smoothed_mean(keys: pd.Series, y: np.ndarray, smoothing: float) -> pd.Series:
    """Bayesian-smoothed target mean per key."""
    prior = y.mean()
    stats = pd.DataFrame({"k": keys.values, "y": y}).groupby("k")["y"].agg(["sum", "count"])
    return (stats["sum"] + prior * smoothing) / (stats["count"] + smoothing)


def add_target_encoding(X_tr, y_tr, X_va, X_te, cols, smoothing=20.0, inner_folds=5, seed=0):
    """Target-encode `cols` without leakage: inner-OOF for train rows, fit-on-all-train for valid/test rows."""
    X_tr, X_va, X_te = X_tr.copy(), X_va.copy(), X_te.copy()
    prior = y_tr.mean()
    soft = not np.all((y_tr == 0) | (y_tr == 1))
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
