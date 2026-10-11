"""
Task 3 & 4 -- Joining prediction files to review metadata and coverage analysis.

Task 3: Verify that prediction files join 1:1 to their respective test sets
        and show how to link predictions to parent_asin, rating, helpful_vote, review_length.

Task 4: Analyze the gray-band gap between count_stage1_test and binary_label_test,
        confirm ~217k gray-band reviews, and check binary prediction coverage.
"""

import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # ARA/
PRED = ROOT / "outputs" / "predictions"
DATA = ROOT / "datasets" / "preprocessed"

SEP = "=" * 72

# ── paths ──────────────────────────────────────────────────────────────
count_path = DATA / "count_stage1_test.csv"
binary_path = DATA / "binary_label_test.csv"

pred_files = {
    "xgb_binary":   PRED / "xgb_binary_test_predictions.csv",
    "bert_binary":  PRED / "bert_binary_test_predictions.csv",
    "xgb_hurdle":   PRED / "xgb_count_stage1_test_combined_count_stage2_log1p_predictions.csv",
    "bert_hurdle":  PRED / "bert_count_combined_count_stage2_log1p_test_predictions.csv",
}

feature_id_files = {
    "binary":     DATA / "X_test_feature_ids_binary.csv",
    "regression": DATA / "X_test_feature_ids_regression.csv",
}

# ── Load metadata (only the columns we need) ──────────────────────────
META_COLS = ["review_id", "parent_asin", "rating", "helpful_vote", "review_length"]

print(SEP)
print("LOADING METADATA")
print(SEP)

count_df = pd.read_csv(count_path, usecols=META_COLS)
print(f"count_stage1_test  : {len(count_df):>12,} rows")

binary_meta = pd.read_csv(binary_path, usecols=META_COLS + ["binary_label"])
print(f"binary_label_test  : {len(binary_meta):>12,} rows")

# ======================================================================
# TASK 3 -- Joining
# ======================================================================
print(f"\n{SEP}")
print("TASK 3 -- JOINING PREDICTION FILES TO METADATA")
print(SEP)

# 3.1  Headers & ID column detection
print("\n--- 3.1  Headers and ID column in each prediction file ---\n")
pred_dfs = {}
for name, path in pred_files.items():
    df = pd.read_csv(path)
    pred_dfs[name] = df
    cols = list(df.columns)
    has_rid = "review_id" in cols
    print(f"  {name}")
    print(f"    path   : {path.relative_to(ROOT)}")
    print(f"    rows   : {len(df):>12,}")
    print(f"    columns: {cols}")
    print(f"    has review_id: {has_rid}")
    if not has_rid:
        print("    ** WARNING: no review_id -- will check row-order match **")
    print()

# 3.2  Feature-ID mapping files
print("--- 3.2  Feature-ID mapping files ---\n")
for label, path in feature_id_files.items():
    if path.exists():
        fid = pd.read_csv(path)
        print(f"  {label}: {path.relative_to(ROOT)}  ({len(fid):,} rows, cols={list(fid.columns)})")
    else:
        print(f"  {label}: NOT FOUND at {path.relative_to(ROOT)}")
print()

# 3.3  1:1 join checks
print("--- 3.3  1:1 join verification ---\n")

def check_join(pred_name, pred_df, ref_name, ref_df):
    """Check if prediction review_ids join 1:1 to reference."""
    if "review_id" not in pred_df.columns:
        # Fall back to row-order check
        if len(pred_df) == len(ref_df):
            print(f"  {pred_name} -> {ref_name}: NO review_id, but row counts match ({len(pred_df):,}).")
            print(f"    Assuming implicit row-order alignment.")
        else:
            print(f"  {pred_name} -> {ref_name}: NO review_id, row counts DIFFER "
                  f"(pred={len(pred_df):,}, ref={len(ref_df):,}) -- CANNOT align.")
        return

    pred_ids = set(pred_df["review_id"])
    ref_ids = set(ref_df["review_id"])

    in_both = pred_ids & ref_ids
    pred_only = pred_ids - ref_ids
    ref_only = ref_ids - pred_ids

    dup_pred = pred_df["review_id"].duplicated().sum()
    dup_ref = ref_df["review_id"].duplicated().sum()

    is_1to1 = (len(pred_only) == 0 and len(ref_only) == 0 and dup_pred == 0 and dup_ref == 0)

    print(f"  {pred_name} ({len(pred_df):,}) <-> {ref_name} ({len(ref_df):,})")
    print(f"    matched     : {len(in_both):>12,}")
    print(f"    pred-only   : {len(pred_only):>12,}")
    print(f"    ref-only    : {len(ref_only):>12,}")
    print(f"    dup in pred : {dup_pred:>12,}")
    print(f"    dup in ref  : {dup_ref:>12,}")
    print(f"    1:1 join?   : {'YES' if is_1to1 else 'NO (see above for details)'}")
    print()

# Binary predictions -> binary_label_test
check_join("xgb_binary",  pred_dfs["xgb_binary"],  "binary_label_test", binary_meta)
check_join("bert_binary", pred_dfs["bert_binary"], "binary_label_test", binary_meta)

# Hurdle predictions -> count_stage1_test
check_join("xgb_hurdle",  pred_dfs["xgb_hurdle"],  "count_stage1_test", count_df)
check_join("bert_hurdle", pred_dfs["bert_hurdle"], "count_stage1_test", count_df)

# 3.4  Demonstrate linking predictions to metadata
print("--- 3.4  Linking predictions to metadata (example merge) ---\n")

for pred_name in ["xgb_binary", "bert_hurdle"]:
    pdf = pred_dfs[pred_name]
    if pred_name.endswith("binary"):
        ref = binary_meta
        ref_name = "binary_label_test"
    else:
        ref = count_df
        ref_name = "count_stage1_test"

    merged = pdf.merge(ref[META_COLS], on="review_id", how="inner")
    print(f"  {pred_name} merged with {ref_name}:")
    print(f"    merged rows: {len(merged):,}")
    print(f"    columns    : {list(merged.columns)}")
    print(f"    sample (first 3 rows):")
    sample_cols = ["review_id", "parent_asin", "rating", "helpful_vote", "review_length"]
    sample_cols = [c for c in sample_cols if c in merged.columns]
    print(merged[sample_cols].head(3).to_string(index=False))
    print()

# ======================================================================
# TASK 4 -- Coverage / Gray-band analysis
# ======================================================================
print(f"\n{SEP}")
print("TASK 4 -- COVERAGE / GRAY-BAND ANALYSIS")
print(SEP)

count_ids = set(count_df["review_id"])
binary_ids = set(binary_meta["review_id"])

gray_ids = count_ids - binary_ids
print(f"\n--- 4.1  Gray-band reviews (in count_stage1_test but NOT in binary_label_test) ---")
print(f"    count_stage1_test rows : {len(count_ids):>12,}")
print(f"    binary_label_test rows : {len(binary_ids):>12,}")
print(f"    gray-band count        : {len(gray_ids):>12,}")
print(f"    expected ~217k?        : {'YES' if 200_000 <= len(gray_ids) <= 230_000 else 'CHECK -- outside 200k-230k range'}")

# 4.2  Gray-band reviews with NO binary prediction
xgb_bin_ids = set(pred_dfs["xgb_binary"]["review_id"]) if "review_id" in pred_dfs["xgb_binary"].columns else set()
bert_bin_ids = set(pred_dfs["bert_binary"]["review_id"]) if "review_id" in pred_dfs["bert_binary"].columns else set()

gray_no_xgb  = gray_ids - xgb_bin_ids
gray_no_bert = gray_ids - bert_bin_ids
gray_no_any  = gray_ids - (xgb_bin_ids | bert_bin_ids)

print(f"\n--- 4.2  Gray-band reviews without binary predictions ---")
print(f"    gray-band without XGB binary  : {len(gray_no_xgb):>12,}")
print(f"    gray-band without BERT binary : {len(gray_no_bert):>12,}")
print(f"    gray-band without ANY binary  : {len(gray_no_any):>12,}")

# 4.3  Helpful_vote distribution of gray-band reviews
gray_df = count_df[count_df["review_id"].isin(gray_ids)].copy()

print(f"\n--- 4.3  helpful_vote distribution of gray-band reviews ---")
hv = gray_df["helpful_vote"]
print(f"    min    : {hv.min()}")
print(f"    max    : {hv.max()}")
print(f"    mean   : {hv.mean():.4f}")
print(f"    median : {hv.median():.1f}")
print(f"\n    value_counts for votes 0-10:")
vc = hv.value_counts().sort_index()
for v in range(11):
    cnt = vc.get(v, 0)
    pct = 100.0 * cnt / len(gray_df) if len(gray_df) > 0 else 0
    print(f"      vote={v:>2d} : {cnt:>10,}  ({pct:5.2f}%)")
above_10 = (hv > 10).sum()
print(f"      vote>10 : {above_10:>10,}  ({100.0 * above_10 / len(gray_df):5.2f}%)")

# 4.4  Do binary prediction files cover ONLY binary_label_test reviews, or also gray-band?
print(f"\n--- 4.5  Binary prediction coverage scope ---")
for name in ["xgb_binary", "bert_binary"]:
    pdf = pred_dfs[name]
    if "review_id" not in pdf.columns:
        print(f"  {name}: no review_id column, cannot check")
        continue
    pids = set(pdf["review_id"])
    in_binary = pids & binary_ids
    in_gray   = pids & gray_ids
    in_neither = pids - count_ids  # not even in count_stage1
    print(f"  {name}  ({len(pids):,} unique review_ids)")
    print(f"    in binary_label_test : {len(in_binary):>12,}")
    print(f"    in gray band         : {len(in_gray):>12,}")
    print(f"    in neither (unknown) : {len(in_neither):>12,}")
    if len(in_gray) == 0:
        print(f"    => covers ONLY binary_label_test reviews (no gray-band)")
    else:
        print(f"    => ALSO covers {len(in_gray):,} gray-band reviews!")
    print()

print(SEP)
print("DONE")
print(SEP)
