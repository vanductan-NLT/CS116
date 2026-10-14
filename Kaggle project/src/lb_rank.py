"""Exact public-leaderboard position of our best submission (best/worst case inside the tie group).

Usage: python src/lb_rank.py
"""
import io
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

import pandas as pd

from config import COMPETITION, ROOT

KAGGLE = str(ROOT / ".venv" / "Scripts" / "kaggle.exe")


def run(args):
    r = subprocess.run([KAGGLE, *args], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)
    return (r.stdout or "") + (r.stderr or "")


out_dir = Path(tempfile.mkdtemp(prefix="lb_", dir="D:/tmp"))
run(["competitions", "leaderboard", "-c", COMPETITION, "--download", "-p", str(out_dir), "-q"])
zips = sorted(out_dir.glob("*.zip"))
if not zips:
    sys.exit("could not download the leaderboard (API timeout?) - retry")
z = zipfile.ZipFile(zips[-1])
lb = pd.read_csv(io.BytesIO(z.read(z.namelist()[0])))
scores = lb["Score"].astype(float).sort_values(ascending=False).values
n = len(scores)
print(f"snapshot {zips[-1].stem.split('-')[-1]} | now {datetime.now():%Y-%m-%d %H:%M} | teams = {n}")
print(f"#1 = {scores[0]:.5f} | top-10% cutoff (rank {int(n * 0.1)}) = {scores[int(n * 0.1) - 1]:.5f}")

subs = pd.read_csv(io.StringIO(run(["competitions", "submissions", "-c", COMPETITION, "-v"])))
subs = subs[pd.to_numeric(subs["publicScore"], errors="coerce").notna()].copy()
subs["publicScore"] = subs["publicScore"].astype(float)
print("our scored submissions (top 4):")
print(subs.sort_values("publicScore", ascending=False)[["fileName", "publicScore"]].head(4).to_string(index=False))
best = subs["publicScore"].max()
better, tied = int((scores > best).sum()), int((scores == best).sum())
print(f"\nbest score {best:.5f}: {better} teams strictly better, {tied} teams share this score (incl. us)")
print(f"=> rank {better + 1} (best case) .. {better + tied} (worst case)  of {n}")
print(f"=> top {100 * (better + 1) / n:.1f}% .. top {100 * (better + tied) / n:.1f}%")
