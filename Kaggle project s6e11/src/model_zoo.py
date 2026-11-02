"""Model factory: every model exposes fit(X_tr, y_tr, X_va, y_va, sample_weight) -> self and predict(X) -> P(y=1).

Labels may be soft (probabilities in [0,1]) when pseudo-labelled test rows are mixed in; GBDT wrappers switch
to a cross-entropy objective in that case.
"""
import numpy as np


def is_soft(y) -> bool:
    return not np.all((y == 0) | (y == 1))
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


class SklearnModel:
    """Wraps sklearn estimators; KNN is trained on a subsample to stay tractable on 560k rows."""

    def __init__(self, kind: str, params: dict, cat_cols: list[str]):
        params = dict(params)
        # extra_cats: "keys" = every k_* column (digit / rounded / pair keys) is one-hot encoded too, so the GLM
        # sees each exact value as its own level (the "GLM on one-hot keys" view of goodpjw2008's series)
        self.extra_cats = params.pop("extra_cats", None)
        self.min_frequency = params.pop("min_frequency", 20)
        self.kind, self.params, self.cat_cols = kind, params, cat_cols

    def fit(self, X_tr, y_tr, X_va=None, y_va=None, sample_weight=None):
        y_tr = np.round(y_tr).astype(int)  # sklearn baselines only support hard labels; weights ignored
        cats = list(self.cat_cols)
        if self.extra_cats == "keys":
            cats += [c for c in X_tr.columns if c.startswith("k_") and c not in cats]
        elif self.extra_cats:
            cats += [c for c in self.extra_cats if c in X_tr.columns and c not in cats]
        num = [c for c in X_tr.columns if c not in cats]
        pre = ColumnTransformer([("num", StandardScaler(), num),
                                 ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=self.min_frequency), cats)])
        if self.kind == "lr":
            est = LogisticRegression(max_iter=self.params.get("max_iter", 2000), C=self.params.get("C", 1.0),
                                     solver=self.params.get("solver", "lbfgs"))
        elif self.kind == "knn":
            est = KNeighborsClassifier(n_neighbors=self.params.get("k", 101), n_jobs=-1)
            n = self.params.get("sample", 100_000)
            idx = np.random.RandomState(0).choice(len(X_tr), min(n, len(X_tr)), replace=False)
            X_tr, y_tr = X_tr.iloc[idx], y_tr[idx]
        elif self.kind == "rf":
            est = RandomForestClassifier(n_estimators=self.params.get("n_estimators", 300),
                                         min_samples_leaf=self.params.get("min_samples_leaf", 5),
                                         max_depth=self.params.get("max_depth", 20),
                                         n_jobs=self.params.get("n_jobs", 4), random_state=0)
            pre = "passthrough"
        else:
            raise ValueError(self.kind)
        self.model = make_pipeline(pre, est) if pre != "passthrough" else est
        self.model.fit(X_tr, y_tr)
        return self

    def predict(self, X):
        return self.model.predict_proba(X)[:, 1]


class LGBMModel:
    def __init__(self, params: dict):
        self.params = {"objective": "binary", "learning_rate": 0.03, "n_estimators": 20000,
                       "num_leaves": 63, "min_child_samples": 50, "subsample": 0.8, "subsample_freq": 1,
                       "colsample_bytree": 0.5, "reg_lambda": 1.0, "verbose": -1, "n_jobs": 16, **params}

    def fit(self, X_tr, y_tr, X_va, y_va, sample_weight=None):
        import lightgbm as lgb
        # LGBMRegressor + cross_entropy objective accepts soft labels and still outputs probabilities
        params = {**self.params, "objective": "cross_entropy"} if is_soft(y_tr) else self.params
        cls = lgb.LGBMRegressor if is_soft(y_tr) else lgb.LGBMClassifier
        self.model = cls(**params)
        self.model.fit(X_tr, y_tr, sample_weight=sample_weight, eval_X=(X_va,), eval_y=(y_va,), eval_metric="auc",
                       callbacks=[lgb.early_stopping(300, verbose=False)])
        return self

    def predict(self, X):
        p = self.model.predict(X) if not hasattr(self.model, "predict_proba") else self.model.predict_proba(X)[:, 1]
        return p


class XGBModel:
    def __init__(self, params: dict):
        self.params = {"learning_rate": 0.03, "n_estimators": 20000, "max_depth": 7, "min_child_weight": 5,
                       "subsample": 0.8, "colsample_bytree": 0.5, "reg_lambda": 1.0, "tree_method": "hist",
                       "device": "cuda", "eval_metric": "auc", "early_stopping_rounds": 300,
                       "objective": "binary:logistic", **params}

    def fit(self, X_tr, y_tr, X_va, y_va, sample_weight=None):
        import xgboost as xgb
        # XGBRegressor with a logistic objective = same model as the classifier but accepts soft labels
        self.model = xgb.XGBRegressor(**self.params)
        self.model.fit(X_tr, y_tr, sample_weight=sample_weight, eval_set=[(X_va, y_va)], verbose=False)
        return self

    def predict(self, X):
        return self.model.predict(X)


class CatModel:
    def __init__(self, params: dict, cat_features: list[str]):
        params = dict(params)
        # extra_cats: numeric columns to also treat as categorical (e.g. Flight Distance as a route id)
        self.cat_features = cat_features + params.pop("extra_cats", [])
        self.params = {"iterations": 20000, "learning_rate": 0.05, "depth": 7, "eval_metric": "AUC",
                       "task_type": "GPU", "od_type": "Iter", "od_wait": 300, "verbose": 0, **params}

    def fit(self, X_tr, y_tr, X_va, y_va, sample_weight=None):
        from catboost import CatBoostClassifier
        cats = [c for c in self.cat_features if c in X_tr.columns]
        X_tr, X_va = X_tr.copy(), X_va.copy()
        for c in cats:
            X_tr[c], X_va[c] = X_tr[c].astype(str), X_va[c].astype(str)
        self.cats = cats
        params = {**self.params, "loss_function": "CrossEntropy"} if is_soft(y_tr) else self.params
        self.model = CatBoostClassifier(**params)
        self.model.fit(X_tr, y_tr, sample_weight=sample_weight, cat_features=cats, eval_set=(X_va, y_va),
                       use_best_model=True)
        return self

    def predict(self, X):
        X = X.copy()
        for c in self.cats:
            X[c] = X[c].astype(str)
        return self.model.predict_proba(X)[:, 1]


class EBMModel:
    """Explainable Boosting Machine (additive model with pairwise terms): the data is near-additive, and the EBM
    members carry the largest stack weights in the public pool, so it is a useful second family next to GBDTs."""

    def __init__(self, params: dict, cat_cols: list[str]):
        self.params = {"max_bins": 256, "interactions": 10, "learning_rate": 0.02, "max_rounds": 10000,
                       "early_stopping_rounds": 100, "n_jobs": 8, "random_state": 0, **params}
        self.cat_cols = cat_cols

    def fit(self, X_tr, y_tr, X_va, y_va, sample_weight=None):
        from interpret.glassbox import ExplainableBoostingClassifier
        feature_types = ["nominal" if c in self.cat_cols else "continuous" for c in X_tr.columns]
        self.model = ExplainableBoostingClassifier(feature_types=feature_types, **self.params)
        self.model.fit(X_tr, np.round(y_tr).astype(int))
        return self

    def predict(self, X):
        return self.model.predict_proba(X)[:, 1]


class RealMLPModel:
    """pytabkit RealMLP (GPU). Hyper-parameters close to the S6E10 public recipe (Vladimir Demidov); the
    validation fold is passed only for early stopping of the epoch count, never for fitting."""

    def __init__(self, params: dict, cat_cols: list[str]):
        self.params = {"device": "cuda", "random_state": 0, "n_epochs": 16, "batch_size": 1024,
                       "hidden_sizes": [256, 256, 256], "lr": 0.04, "p_drop": 0.15, "plr_sigma": 2.0,
                       "use_ls": False, "val_metric_name": "1-auc_ovr", "verbosity": 0, **params}
        self.cat_cols = cat_cols

    def fit(self, X_tr, y_tr, X_va, y_va, sample_weight=None):
        import pandas as pd
        from pytabkit import RealMLP_TD_Classifier
        cats = [c for c in self.cat_cols if c in X_tr.columns]
        Xa = pd.concat([X_tr, X_va], ignore_index=True)
        ya = np.concatenate([np.round(y_tr).astype(int), np.asarray(y_va).astype(int)])
        val_idx = np.arange(len(X_tr), len(Xa))
        self.model = RealMLP_TD_Classifier(**self.params)
        self.model.fit(Xa, ya, val_idxs=val_idx, cat_col_names=cats)
        return self

    def predict(self, X):
        return self.model.predict_proba(X)[:, 1]


def make_model(kind: str, params: dict, cat_cols: list[str]):
    if kind in ("lr", "knn", "rf"):
        return SklearnModel(kind, params, cat_cols)
    if kind == "ebm":
        return EBMModel(params, cat_cols)
    if kind == "realmlp":
        return RealMLPModel(params, cat_cols)
    if kind == "lgbm":
        return LGBMModel(params)
    if kind == "xgb":
        return XGBModel(params)
    if kind == "cat":
        return CatModel(params, cat_cols)
    raise ValueError(kind)
