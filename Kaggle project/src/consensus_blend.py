"""Public-hedge submission: rank-blend our twin stack with the one low-correlation public test file.

The public LB on 16 Oct 2026 clustered at 0.96190 = 0.7 * a 0.96188 stack + 0.3 * Den Pugovkin's
"passenger-context foundation ensemble" (0.96183), the only public file whose rank correlation with
the big stacks is ~0.9965 instead of ~0.9999. Den's file ships without OOF, so this blend cannot be
checked by nested CV; it is kept as the public-LB hedge (final pick #2), never as the CV-selected pick.
Sources: dataset nina2025/ps-s6e10-7 (0.96183.csv), recipe by yuvrajlamba01.

Usage: python src/consensus_blend.py  -> submissions/twin_v3_consensus_den30.csv (not submitted)
"""
import subprocess
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import rankdata, spearmanr

from config import ROOT

KAGGLE = str(ROOT / ".venv" / "Scripts" / "kaggle.exe")
PUBLIC_DIR = Path("D:/tmp/nina7")
DEN_FILE = PUBLIC_DIR / "0.96183.csv"
SUBS = ROOT / "submissions"
W_DEN = 0.3


def load(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df.set_index("id")["satisfaction"].astype(float)


if not DEN_FILE.exists():
    PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run([KAGGLE, "datasets", "download", "nina2025/ps-s6e10-7", "-p", str(PUBLIC_DIR), "--unzip"], check=True)

twin = load(SUBS / "twin_v2_dariush.csv")
den = load(DEN_FILE).reindex(twin.index)
if den.isna().any():
    sys.exit("public file does not cover every test id")
print(f"rank corr(twin_v2, den) = {spearmanr(twin, den)[0]:.6f}")

rank = lambda s: rankdata(s) / len(s)  # noqa: E731
blend = (1 - W_DEN) * rank(twin) + W_DEN * rank(den)
out = SUBS / "twin_v3_consensus_den30.csv"
pd.DataFrame({"id": twin.index, "satisfaction": blend}).to_csv(out, index=False)
print(f"written {out} (rank corr with twin_v2 = {spearmanr(blend, twin)[0]:.6f}); submit by hand")
