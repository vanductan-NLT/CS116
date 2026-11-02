"""Shared paths, column names and the fixed CV scheme for the S6E11 project (Predicting Credit Card Fraud)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OOF_DIR = ROOT / "oof"
TEST_PRED_DIR = ROOT / "test_preds"
SUB_DIR = ROOT / "submissions"
LOG_DIR = ROOT / "logs"
for _d in (OOF_DIR, TEST_PRED_DIR, SUB_DIR, LOG_DIR):
    _d.mkdir(exist_ok=True)

COMPETITION = "playground-series-s6e11"
ID_COL = "id"
TARGET = "is_fraud"

# Same split convention the community shared OOFs on in S6E10: StratifiedKFold(5, shuffle=True, random_state=42)
N_FOLDS = 5
FOLD_SEED = 42

CAT_COLS = ["merchant_category", "card_type", "auth_method", "channel", "device_type"]
BOOL_COLS = ["is_foreign_transaction", "is_new_merchant", "used_vpn", "ip_country_mismatch",
             "billing_shipping_mismatch", "is_ai_generated_scam_attempt"]
# continuous-ish columns (many distinct values) vs small-integer columns
FLOAT_COLS = ["amount_usd", "hours_since_last_txn", "distance_from_home_km", "account_balance_usd",
              "velocity_score", "merchant_risk_score"]
INT_COLS = ["txn_count_last_24h", "card_age_months", "customer_age", "cvv_retry_count",
            "time_of_day_hour", "day_of_week", "prior_disputes"]
NUM_COLS = FLOAT_COLS + INT_COLS
