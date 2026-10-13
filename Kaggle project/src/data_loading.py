"""Load competition CSVs once, cache as parquet, and build the fixed fold assignment."""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from config import CAT_COLS, DATA_DIR, FOLD_SEED, ID_COL, N_FOLDS, TARGET


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train, test). Target is cast to int 0/1; categoricals kept as strings."""
    cache_tr, cache_te = DATA_DIR / "train.parquet", DATA_DIR / "test.parquet"
    if cache_tr.exists() and cache_te.exists():
        return pd.read_parquet(cache_tr), pd.read_parquet(cache_te)
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    train[TARGET] = train[TARGET].astype(int)
    for c in CAT_COLS:
        train[c] = train[c].astype(str)
        test[c] = test[c].astype(str)
    train.to_parquet(cache_tr, index=False)
    test.to_parquet(cache_te, index=False)
    return train, test


def get_folds(train: pd.DataFrame, n_folds: int = N_FOLDS, seed: int = FOLD_SEED) -> np.ndarray:
    """Fold id per train row (0..n_folds-1) from StratifiedKFold on the target."""
    folds = np.zeros(len(train), dtype=int)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for k, (_, va) in enumerate(skf.split(train, train[TARGET])):
        folds[va] = k
    return folds


def write_submission(test: pd.DataFrame, pred: np.ndarray, path) -> None:
    """Write id,satisfaction probability CSV in the sample_submission format."""
    pd.DataFrame({ID_COL: test[ID_COL], TARGET: pred}).to_csv(path, index=False)
