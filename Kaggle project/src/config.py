"""Shared paths, column names and the fixed CV scheme for the S6E10 project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OOF_DIR = ROOT / "oof"
TEST_PRED_DIR = ROOT / "test_preds"
SUB_DIR = ROOT / "submissions"
LOG_DIR = ROOT / "logs"
for _d in (OOF_DIR, TEST_PRED_DIR, SUB_DIR, LOG_DIR):
    _d.mkdir(exist_ok=True)

COMPETITION = "playground-series-s6e10"
ID_COL = "id"
TARGET = "satisfaction"

# Same split the community shares OOFs on: StratifiedKFold(5, shuffle=True, random_state=42)
N_FOLDS = 5
FOLD_SEED = 42

CAT_COLS = ["Gender", "Customer Type", "Type of Travel", "Class"]
RATING_COLS = [
    "Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking",
    "Gate location", "Food and drink", "Online boarding", "Seat comfort",
    "Inflight entertainment", "On-board service", "Leg room service",
    "Baggage handling", "Checkin service", "Cleanliness",
]
DELAY_COLS = ["Departure Delay in Minutes", "Arrival Delay in Minutes"]
NUM_COLS = ["Age", "Flight Distance"] + DELAY_COLS
