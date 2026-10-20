"""Morning routine: detect new public OOF versions, re-import + re-stack, submit if nested CV improved, report rank.

Steps
  1. For each known public OOF source, compare the newest file date on Kaggle with the local import time;
     re-download sources that changed (only those).
  2. List newest competition notebooks whose title mentions OOF/stack (new sources to inspect by hand).
  3. If anything was re-downloaded: run import_public_pool.py, then stack_oof.py -> stack_daily_<date>.csv.
  4. If nested CV > best so far (logs/daily_state.json): submit, record. Always print our LB rank.
Usage: python src/daily_refresh.py [--force-stack] [--no-submit]
"""
import argparse
import io
import json
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config import COMPETITION, DATA_DIR, LOG_DIR, ROOT, SUB_DIR

KAGGLE = str(ROOT / ".venv" / "Scripts" / "kaggle.exe")
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
STATE = LOG_DIR / "daily_state.json"
B = DATA_DIR / "public_oof"
DATASETS = {  # kaggle dataset ref -> local folder name
    "cdeotte/s6e10-124-public-oof-and-test-predictions": "s6e10-124-public-oof-and-test-predictions",
    "cdeotte/s6e10-tfm-oof-and-test-predictions": "s6e10-tfm-oof-and-test-predictions",
    "kagankoral/s6e10-ablation-members-oof": "s6e10-ablation-members-oof",
    "sadamtorres/ps-s6e10-airline-satisfaction-artifacts": "ps-s6e10-airline-satisfaction-artifacts",
    "amanatar/s6e10-forge-own-pairs": "s6e10-forge-own-pairs",
    "goodpjw2008/s6e10-goodpjw2008-extra-members-oof": "s6e10-goodpjw2008-extra-members-oof",
    "koumeimaki/s6e10-route-tfm-nn-oof": "s6e10-route-tfm-nn-oof",
    "abdullahsafwan333/oof-dataset": "oof-dataset",
    "dmitriykolmikov/s6e10-catboost-all-categorical-oof-and-test": "s6e10-catboost-all-categorical-oof-and-test",
    "dariushafshar/s6e10-golem-oof-library": "s6e10-golem-oof-library",
    "sachith7/s6e10-stack-oof-predictions": "s6e10-stack-oof-predictions",
    "megayak/s6e10-oof-library": "s6e10-oof-library",
    "arhancanli12/s6e10-oof-test-predictions": "s6e10-oof-test-predictions",
    "najiama/s6e10-oof": "s6e10-oof",
}


def run(cmd, timeout=1800):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, encoding="utf-8", errors="replace")
    return (r.stdout or "") + (r.stderr or "")


def newest_remote_date(ref):
    """Newest creationDate among a dataset's files, or None on error."""
    out = run([KAGGLE, "datasets", "files", ref, "-v"], timeout=120)
    # newer CLI versions print a "Next Page Token = ..." line before the CSV header: keep only the table
    csv_part = "\n".join(line for line in out.splitlines() if "," in line and not line.startswith("Next Page Token"))
    try:
        df = pd.read_csv(io.StringIO(csv_part))
        return pd.to_datetime(df["creationDate"]).max().tz_localize(None)
    except Exception:  # noqa: BLE001
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force-stack", action="store_true")
    ap.add_argument("--no-submit", action="store_true")
    a = ap.parse_args()
    state = json.loads(STATE.read_text()) if STATE.exists() else {"best_cv": 0.962222, "last_import": "2026-10-13T12:00:00"}
    last_import = pd.Timestamp(state["last_import"])
    today = datetime.now().strftime("%Y%m%d")

    # 1. new versions of known sources
    changed = []
    for ref, folder in DATASETS.items():
        d = newest_remote_date(ref)
        if d is not None and d > last_import:
            print(f"NEW VERSION: {ref} ({d})")
            run([KAGGLE, "datasets", "download", "-d", ref, "-p", str(B / folder), "--unzip", "-q", "--force"])
            changed.append(ref)
    print(f"sources changed: {len(changed)}")

    # 2. newest notebooks mentioning OOF / stack (manual follow-up)
    out = run([KAGGLE, "kernels", "list", "--competition", COMPETITION, "--sort-by", "dateRun", "--page-size", "40", "-v"], timeout=120)
    try:
        df = pd.read_csv(io.StringIO(out))
        df = df[df["title"].str.contains("oof|stack|member|library", case=False, na=False)]
        print("recent OOF/stack notebooks:\n" + df[["ref", "title", "lastRunTime"]].to_string(index=False))
    except Exception as e:  # noqa: BLE001
        print("kernel list failed:", e)

    # 3. re-import + re-stack when something changed
    if changed or a.force_stack:
        print(run([PY, str(ROOT / "src" / "import_public_pool.py")])[-400:])
        out = run([PY, str(ROOT / "src" / "stack_oof.py"), "--members", "all",
                   "--exclude", "base_lr,base_knn,base_rf,base_lgbm_default,xgb_v1_pl,teacher5",
                   "--family-reg", "--quiet", "--out", f"stack_daily_{today}"], timeout=3600)
        line = [l for l in out.splitlines() if "nested CV" in l]
        print(line[-1] if line else out[-500:])
        cv = float(line[-1].split("nested CV   : ")[1].split()[0]) if line else 0.0
        # Kaggle's creationDate is UTC: store the watermark in UTC too, or a version uploaded in the
        # 7 hours after a local-time refresh would be skipped as "old"
        state["last_import"] = datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds")
        if cv > state["best_cv"] + 1e-6:
            print(f"IMPROVED: {cv:.6f} > {state['best_cv']:.6f}")
            if not a.no_submit:
                print(run([KAGGLE, "competitions", "submit", "-c", COMPETITION, "-f", str(SUB_DIR / f"stack_daily_{today}.csv"),
                           "-m", f"stack_daily_{today}: nested CV {cv:.6f}"], timeout=600)[-200:])
            state["best_cv"] = cv
            state["best_file"] = f"stack_daily_{today}.csv"
        else:
            print(f"no improvement ({cv:.6f} vs best {state['best_cv']:.6f})")
    STATE.write_text(json.dumps(state, indent=2))

    # 4. leaderboard position of our best public score
    run([KAGGLE, "competitions", "leaderboard", "-c", COMPETITION, "--download", "-p", "D:/tmp/lb_daily", "-q"], timeout=300)
    zips = sorted(Path("D:/tmp/lb_daily").glob("*.zip"), key=lambda p: p.stat().st_mtime)
    if zips:
        z = zipfile.ZipFile(zips[-1])
        lb = pd.read_csv(io.BytesIO(z.read(z.namelist()[0])))
        s = lb["Score"].astype(float).sort_values(ascending=False).values
        # the submissions endpoint times out now and then: retry, and never report a made-up score
        best = None
        for _ in range(3):
            try:
                subs = run([KAGGLE, "competitions", "submissions", "-c", COMPETITION, "-v"], timeout=120)
                best = pd.to_numeric(pd.read_csv(io.StringIO(subs))["publicScore"], errors="coerce").max()
                break
            except Exception:  # noqa: BLE001
                continue
        if best is None or pd.isna(best):
            print(f"LB: could not read our submissions (API timeout) | #1 = {s[0]:.5f}, "
                  f"top-10% cutoff = {s[int(len(s) * 0.1)]:.5f}, teams = {len(s)} -> run src/lb_rank.py")
        else:
            better, tied = int((s > best).sum()), int((s == best).sum())
            print(f"LB: our best public {best:.5f} -> rank {better + 1}..{better + tied}/{len(s)} "
                  f"(top {100 * (better + 1) / len(s):.1f}%..{100 * (better + tied) / len(s):.1f}%), #1 = {s[0]:.5f}, "
                  f"top-10% cutoff = {s[int(len(s) * 0.1)]:.5f}")


if __name__ == "__main__":
    main()
