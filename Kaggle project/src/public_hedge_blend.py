"""Public-LB hedge (final pick #2 only): rank-blend twin_v3 with two credited public-max files.

By 21 Oct 2026 the public top-10% cutoff (0.96187) was creeping up on twin_v3 (0.96189) because
hundreds of teams copy public files tuned on the public split. This blend borrows two of those
files - hanifnoerrofiq's 0.96193 and mayankraaz's LB-surface-ascent v2 0.96196, both re-hosted in
goodpjw2008's dataset - and scored public 0.96195. Their gains are public-split extrapolation that
honest nested CV rejects (goodpjw2008, Part 2 of "Meta Stack on Public OOF": -935e-6), so this file
is never the CV-selected pick; Kaggle scores the better of the two selected files, so it cannot hurt.

Usage: python src/public_hedge_blend.py -> submissions/public_hedge_v1.csv (submit by hand)
"""
import glob
import sys

import pandas as pd
from scipy.stats import rankdata, spearmanr

from config import ROOT

SRC = ROOT / "data" / "public_oof" / "s6e10-goodpjw2008-extra-members-oof"
SUBS = ROOT / "submissions"
PUBLIC = {"hanifnoerrofiq_agentic_1019.csv": 0.25, "mayankraaz_surface_v2_1019.csv": 0.25}  # weights; rest = twin_v3


def load(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df.set_index("id").iloc[:, 0].astype(float)


twin = load(SUBS / "twin_v3_consensus_den30.csv")
rank = lambda s: rankdata(s) / len(s)  # noqa: E731
blend = (1 - sum(PUBLIC.values())) * rank(twin)
for name, w in PUBLIC.items():
    hits = glob.glob(str(SRC / "**" / name), recursive=True)
    if not hits:
        sys.exit(f"{name} missing: run src/daily_refresh.py to download the goodpjw2008 dataset")
    f = load(hits[0]).reindex(twin.index)
    print(f"{name}: rank corr with twin_v3 = {spearmanr(f, twin)[0]:.6f}")
    blend += w * rank(f)
out = SUBS / "public_hedge_v1.csv"
pd.DataFrame({"id": twin.index, "satisfaction": blend}).to_csv(out, index=False)
print(f"written {out}")
