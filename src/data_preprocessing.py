import os
import re
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import train_test_split

# Sentiment
from nltk import download
from nltk.sentiment import SentimentIntensityAnalyzer
download("vader_lexicon")
sia = SentimentIntensityAnalyzer()

# ------------------- CONFIG -------------------
RAW_JSONL = "../datasets/Movies_and_TV.jsonl"

# Tokenization/cleaning
MIN_TOKENS = 10

# Capping + target transforms
CAP_Q = 0.99  # cap helpful votes at 99th percentile
BIN_POS_Q = 0.95  # positive threshold quantile computed on TRAIN ONLY
BIN_NEG_Q = 0.70  # negative threshold quantile computed on TRAIN ONLY
APPLY_GRAY_BAND = True
GRAY_BAND = 0.00  # set >0 (e.g., 0.02) to widen the drop zone around thresholds

# Optional age filter for binary only (to remove "not yet labeled" near-zero reviews)
APPLY_AGE_FILTER = True
DAYS_OLD_MIN = 14  # require at least 14 days old when helpful_vote <= 1

# Splits
RANDOM_STATE = 42
TRAIN_PCT = 0.7
VAL_PCT_OF_TEMP = 0.5  # of the remaining 30%

# ------------------------------------------------

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    return text.lower().strip()

def parse_timestamp(ts):
    """
    Tries to parse timestamp to pandas datetime.
    - If numeric-looking: first try seconds since epoch; if too small/large, try ms.
    - Else, let pandas infer.
    Returns pandas.Timestamp or NaT.
    """
    try:
        if pd.isna(ts):
            return pd.NaT
        # numeric?
        if isinstance(ts, (int, float, np.integer, np.floating)) or (isinstance(ts, str) and ts.isdigit()):
            ts = float(ts)
            # if looks like ms epoch
            if ts > 1e12:
                return pd.to_datetime(ts, unit="ms", utc=True)
            # if looks like s epoch
            if ts > 1e9:
                return pd.to_datetime(ts, unit="s", utc=True)
            # fallback: let pandas guess
            return pd.to_datetime(ts, utc=True, errors="coerce")
        # string datetime
        return pd.to_datetime(ts, utc=True, errors="coerce")
    except Exception:
        return pd.NaT

def load_reviews(path):
    rows = []
    with open(path, "r", encoding="utf-8") as fp:
        for line in tqdm(fp, desc="Loading reviews"):
            rows.append(json.loads(line.strip()))
    df = pd.DataFrame(rows)
    return df

print("➡️  Loading raw reviews...")
reviews_df = load_reviews(RAW_JSONL)

# ------------ Clean & basic features ------------
print("➡️  Cleaning text and building features...")
reviews_df["clean_text"] = reviews_df["text"].apply(clean_text)
reviews_df["clean_title"] = reviews_df["title"].apply(clean_text)
reviews_df["full_text"] = reviews_df["clean_title"].fillna('') + ". " + reviews_df["clean_text"].fillna('')

# types
reviews_df["helpful_vote"] = pd.to_numeric(reviews_df["helpful_vote"], errors="coerce").fillna(0).astype(int)
reviews_df["rating"] = pd.to_numeric(reviews_df.get("rating", np.nan), errors="coerce").fillna(0).astype(int)
reviews_df["is_verified"] = reviews_df.get("verified_purchase", False).astype(bool).astype(int)

# lengths & sentiment
reviews_df["review_length"] = reviews_df["clean_text"].apply(lambda x: len(x.split()))
reviews_df["sentiment_score"] = reviews_df["clean_text"].apply(lambda x: sia.polarity_scores(x)["compound"])

# timestamp -> age
print("➡️  Parsing timestamps...")
reviews_df["timestamp_parsed"] = reviews_df.get("timestamp", pd.NaT).apply(parse_timestamp)
now = pd.Timestamp.utcnow()
reviews_df["review_age_days"] = (now - reviews_df["timestamp_parsed"]).dt.days

# basic content filter (keep super-short out)
base = reviews_df[reviews_df["review_length"] >= MIN_TOKENS].copy()

# ---------------- Targets (shared transforms) ----------------
print("➡️  Building targets (cap + log)...")
cap_value = base["helpful_vote"].quantile(CAP_Q)
base["helpful_vote_capped"] = np.minimum(base["helpful_vote"], cap_value)
base["log_helpfulness_score"] = np.log1p(base["helpful_vote_capped"])
max_log = base["log_helpfulness_score"].max() if base["log_helpfulness_score"].max() > 0 else 1.0
base["normalized_helpfulness_score"] = (base["log_helpfulness_score"] / max_log) * 10.0

# ============================================================
#                 REGRESSION DATASET
#   (keep full distribution; no age/gray filtering)
# ============================================================
print("➡️  Preparing REGRESSION splits...")
reg_df = base.copy()
reg_df["regression_score"] = reg_df["log_helpfulness_score"]

# Required columns
reg_required = ["full_text", "regression_score", "review_length", "sentiment_score",
                "is_verified", "rating", "helpful_vote", "timestamp_parsed", "review_age_days"]
reg_df = reg_df.dropna(subset=["full_text", "regression_score"]).copy()

# Stratify on deciles of regression target for stable splits
reg_strat = pd.qcut(reg_df["regression_score"], q=10, labels=False, duplicates="drop")

reg_train, reg_temp = train_test_split(
    reg_df, test_size=1.0 - TRAIN_PCT, stratify=reg_strat, random_state=RANDOM_STATE
)
reg_temp_strat = pd.qcut(reg_temp["regression_score"], q=5, labels=False, duplicates="drop")
reg_val, reg_test = train_test_split(
    reg_temp, test_size=VAL_PCT_OF_TEMP, stratify=reg_temp_strat, random_state=RANDOM_STATE
)

os.makedirs("../datasets/preprocessed", exist_ok=True)
reg_train.to_csv("../datasets/preprocessed/regression_score_train.csv", index=False)
reg_val.to_csv("../datasets/preprocessed/regression_score_val.csv", index=False)
reg_test.to_csv("../datasets/preprocessed/regression_score_test.csv", index=False)
print("✅ Saved regression splits to ../datasets/preprocessed/")

# ============================================================
#                   BINARY DATASET (precision-oriented)
#   - compute thresholds from TRAIN ONLY
#   - drop gray zone
#   - optional age filter for (<=1 vote & too new)
# ============================================================
print("➡️  Preparing BINARY splits (precision‑oriented)...")
bin_df = base.copy()

# Optional age filter: drop likely-not-yet-labeled near-zero reviews
if APPLY_AGE_FILTER:
    before = len(bin_df)
    mask_drop = (bin_df["helpful_vote"] <= 1) & (bin_df["review_age_days"].notna()) & (bin_df["review_age_days"] < DAYS_OLD_MIN)
    bin_df = bin_df[~mask_drop].copy()
    print(f"   • Age filter removed {before - len(bin_df)} rows (<=1 vote & <{DAYS_OLD_MIN} days old).")

# Split first (using regression-like spectrum strat to keep distribution stable)
bin_strat = pd.qcut(bin_df["log_helpfulness_score"], q=10, labels=False, duplicates="drop")
bin_train, bin_temp = train_test_split(
    bin_df, test_size=1.0 - TRAIN_PCT, stratify=bin_strat, random_state=RANDOM_STATE
)
bin_temp_strat = pd.qcut(bin_temp["log_helpfulness_score"], q=5, labels=False, duplicates="drop")
bin_val, bin_test = train_test_split(
    bin_temp, test_size=VAL_PCT_OF_TEMP, stratify=bin_temp_strat, random_state=RANDOM_STATE
)

# Compute pos/neg thresholds from TRAIN ONLY (normalized space)
pos_thr = bin_train["normalized_helpfulness_score"].quantile(BIN_POS_Q)
neg_thr = bin_train["normalized_helpfulness_score"].quantile(BIN_NEG_Q)

# Optional gray band widening around thresholds
if APPLY_GRAY_BAND and GRAY_BAND > 0:
    pos_thr_eff = pos_thr + GRAY_BAND * (10.0)  # scale by range (0..10)
    neg_thr_eff = neg_thr - GRAY_BAND * (10.0)
else:
    pos_thr_eff = pos_thr
    neg_thr_eff = neg_thr

def label_and_filter(df, neg_thr, pos_thr):
    s = df["normalized_helpfulness_score"]
    labels = np.full(len(df), fill_value=-1, dtype=np.int64)
    labels[s >= pos_thr] = 1
    labels[s <= neg_thr] = 0
    keep = labels >= 0
    out = df.loc[keep].copy()
    out["binary_label"] = labels[keep]
    return out

bin_train_l = label_and_filter(bin_train, neg_thr_eff, pos_thr_eff)
bin_val_l   = label_and_filter(bin_val,   neg_thr_eff, pos_thr_eff)
bin_test_l  = label_and_filter(bin_test,  neg_thr_eff, pos_thr_eff)

print(f"   • Train sizes: before={len(bin_train)} after={len(bin_train_l)} | pos%={bin_train_l['binary_label'].mean():.3f}")
print(f"   •   Val sizes: before={len(bin_val)}   after={len(bin_val_l)}   | pos%={bin_val_l['binary_label'].mean():.3f}")
print(f"   •  Test sizes: before={len(bin_test)}  after={len(bin_test_l)}  | pos%={bin_test_l['binary_label'].mean():.3f}")
print(f"   • Thresholds (TRAIN): neg≤{neg_thr:.3f}  pos≥{pos_thr:.3f}  (effective: neg≤{neg_thr_eff:.3f} pos≥{pos_thr_eff:.3f})")

# Save
bin_train_l.to_csv("../datasets/preprocessed/binary_label_train.csv", index=False)
bin_val_l.to_csv("../datasets/preprocessed/binary_label_val.csv", index=False)
bin_test_l.to_csv("../datasets/preprocessed/binary_label_test.csv", index=False)
print("✅ Saved binary splits to ../datasets/preprocessed/")

print("Done.")
