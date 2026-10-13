"""Generate the EDA figures and tables used in the written report into docs/figures/.

Covers the CS116 syllabus items: target balance, univariate distributions, bivariate relation to the target,
univariate AUC per feature, correlation heatmap, missing values, and train/test/original drift.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score

from config import CAT_COLS, DELAY_COLS, RATING_COLS, ROOT, TARGET
from data_loading import load_raw
from original_data_features import load_original

OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", font_scale=0.9)

train, test = load_raw()
orig = load_original()
y = train[TARGET].values
feats = [c for c in train.columns if c not in ("id", TARGET)]

# --- 1. target balance + missing values
fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
train[TARGET].value_counts().rename({0: "not satisfied", 1: "satisfied"}).plot.bar(ax=ax[0], color=["#c44", "#4a4"])
ax[0].set_title("Target balance (train)"); ax[0].set_ylabel("rows")
miss = pd.DataFrame({"train": train.isna().sum(), "test": test.isna().sum()}).loc[feats]
miss[miss.sum(1) > 0].plot.bar(ax=ax[1]); ax[1].set_title("Missing values per column")
fig.tight_layout(); fig.savefig(OUT / "01_target_missing.png", dpi=150); plt.close(fig)

# --- 2. univariate distributions of numeric columns (train vs test vs original)
num = ["Age", "Flight Distance"] + DELAY_COLS
fig, axes = plt.subplots(1, 4, figsize=(15, 3.2))
for ax, c in zip(axes, num):
    for df, lab in [(train, "train"), (test, "test"), (orig, "original")]:
        v = df[c].dropna()
        v = np.log1p(v) if "Delay" in c else v
        ax.hist(v, bins=60, density=True, histtype="step", label=lab)
    ax.set_title(("log1p " if "Delay" in c else "") + c); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(OUT / "02_numeric_distributions.png", dpi=150); plt.close(fig)

# --- 3. ratings: distribution and satisfaction rate per level (bivariate)
fig, axes = plt.subplots(3, 5, figsize=(16, 8.5))
for ax, c in zip(axes.ravel(), RATING_COLS + CAT_COLS[:2]):
    g = train.groupby(c)[TARGET].agg(["mean", "size"])
    ax.bar(g.index.astype(str), g["size"] / len(train), color="#bbb", label="share of rows")
    ax2 = ax.twinx(); ax2.plot(g.index.astype(str), g["mean"], "o-", color="#c33", label="P(satisfied)")
    ax2.set_ylim(0, 1); ax.set_title(c, fontsize=9); ax.tick_params(labelsize=7); ax2.tick_params(labelsize=7)
fig.suptitle("Share of rows (bars) and satisfaction rate (red) per level"); fig.tight_layout()
fig.savefig(OUT / "03_ratings_bivariate.png", dpi=150); plt.close(fig)

# --- 4. univariate AUC per feature (effect size instead of p-values at n=700k)
rows = []
for c in feats:
    v = train[c]
    if c in CAT_COLS:
        v = v.map(train.groupby(c)[TARGET].mean())
    v = v.fillna(v.median())
    auc = roc_auc_score(y, v)
    rows.append((c, max(auc, 1 - auc)))
uni = pd.DataFrame(rows, columns=["feature", "univariate AUC"]).sort_values("univariate AUC", ascending=False)
uni.to_csv(OUT / "04_univariate_auc.csv", index=False)
fig, ax = plt.subplots(figsize=(7, 5.5))
sns.barplot(data=uni, y="feature", x="univariate AUC", ax=ax, color="#4a7")
ax.set_xlim(0.5, 0.9); ax.set_title("Univariate AUC of each feature vs satisfaction")
fig.tight_layout(); fig.savefig(OUT / "04_univariate_auc.png", dpi=150); plt.close(fig)

# --- 5. correlation heatmap (Spearman) of ratings + numeric + target
corr = train[RATING_COLS + num + [TARGET]].corr(method="spearman")
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, cmap="coolwarm", center=0, annot=True, fmt=".2f", annot_kws={"size": 6}, ax=ax)
ax.set_title("Spearman correlation"); fig.tight_layout(); fig.savefig(OUT / "05_correlation.png", dpi=150); plt.close(fig)

# --- 6. Flight Distance behaves like a route id: value counts
vc = pd.concat([train, test])["Flight Distance"].value_counts()
fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
ax[0].hist(vc.values, bins=80, log=True); ax[0].set_title(f"rows per Flight Distance value ({len(vc)} distinct values)")
ax[0].set_xlabel("rows sharing the value"); ax[0].set_ylabel("count (log)")
g = train.groupby("Flight Distance")[TARGET].agg(["mean", "size"]).query("size >= 200")
ax[1].scatter(g.index, g["mean"], s=4, alpha=0.5); ax[1].set_title("satisfaction rate per distance value (n>=200)")
ax[1].set_xlabel("Flight Distance"); ax[1].set_ylabel("P(satisfied)")
fig.tight_layout(); fig.savefig(OUT / "06_flight_distance_route.png", dpi=150); plt.close(fig)

# --- 7. summary table
summary = pd.DataFrame({
    "dtype": train[feats].dtypes.astype(str), "n_unique": train[feats].nunique(),
    "missing_train": train[feats].isna().sum(), "missing_test": test[feats].isna().sum(),
    "univariate_auc": uni.set_index("feature")["univariate AUC"].round(4)})
summary.to_csv(OUT / "00_feature_summary.csv")
print(summary.to_string())
print("figures written to", OUT)
