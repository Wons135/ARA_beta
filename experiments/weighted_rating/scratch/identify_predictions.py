"""
Identify which candidate prediction CSV files reproduce the reported thesis metrics.

Reported metrics:
  XGBoost binary (test):  AUROC 0.913050, AUPRC 0.543307, threshold 0.800070
  DistilBERT binary (test): AUROC 0.907987, AUPRC 0.492692, threshold 0.106897
  XGBoost hurdle log1p Stage 2 (test): RMSE 15.0280, MAE 1.7306, Spearman 0.5398
  DistilBERT hurdle log1p Stage 2 (test): RMSE 15.0149, MAE 1.9395
  Binary test set: 1,435,602 reviews
  Count test set:  1,653,016 reviews
"""

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import spearmanr
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # ARA/

# ── Reported targets ──────────────────────────────────────────────────────────
REPORTED_BINARY = {
    "xgb":  {"auroc": 0.913050, "auprc": 0.543307, "threshold": 0.800070},
    "bert": {"auroc": 0.907987, "auprc": 0.492692, "threshold": 0.106897},
}
REPORTED_COUNT = {
    "xgb_stage2":      {"rmse": 15.0280, "mae": 1.7306, "spearman": 0.5398},
    "bert_stage2":     {"rmse": 15.0149, "mae": 1.9395},
}
BINARY_N = 1_435_602
COUNT_N  = 1_653_016

TOL = 0.005  # tolerance for matching


def match_tag(computed, reported, tol=TOL):
    """Return True if computed value is within tol of reported."""
    return abs(computed - reported) < tol


# ── Binary candidates ─────────────────────────────────────────────────────────
BINARY_CANDIDATES = [
    ROOT / "outputs/predictions/xgb_binary_test_predictions.csv",
    ROOT / "outputs/predictions/bert_binary_test_predictions.csv",
    ROOT / "outputs/predictions/xgb_probs_binary.csv",
    ROOT / "outputs/predictions/bert_bin_predictions.csv",
    ROOT / "outputs/bert_binary_smoke_fix/predictions/bert_binary_test_predictions.csv",
]

BINARY_LABEL_FILE = ROOT / "datasets/preprocessed/binary_label_test.csv"

# ── Count / hurdle candidates ─────────────────────────────────────────────────
COUNT_CANDIDATES = [
    ROOT / "outputs/predictions/xgb_count_stage1_test_combined_count_stage2_log1p_predictions.csv",
    ROOT / "outputs/predictions/xgb_count_stage2_log1p_test_predictions.csv",
    ROOT / "outputs/predictions/bert_count_combined_count_stage2_log1p_test_predictions.csv",
    ROOT / "outputs/predictions/bert_count_stage2_log1p_test_predictions.csv",
]

COUNT_LABEL_FILE = ROOT / "datasets/preprocessed/count_stage1_test.csv"


def evaluate_binary(path: Path):
    print(f"\n{'='*80}")
    print(f"BINARY FILE: {path.relative_to(ROOT)}")
    print(f"{'='*80}")

    if not path.exists():
        print("  FILE NOT FOUND")
        return

    df = pd.read_csv(path)
    print(f"  Columns : {list(df.columns)}")
    print(f"  Rows    : {len(df):,}")

    # Identify prob and label columns
    prob_col = None
    label_col = None
    for c in df.columns:
        if "prob" in c.lower():
            prob_col = c
        if "true" in c.lower() or c == "binary_label":
            label_col = c

    if prob_col is None or label_col is None:
        print(f"  Could not identify prob_col={prob_col} / label_col={label_col}")
        # If label is missing, try joining with external labels
        if prob_col and label_col is None and "review_id" in df.columns:
            print("  Attempting join with binary_label_test.csv via review_id ...")
            labels = pd.read_csv(BINARY_LABEL_FILE, usecols=["review_id", "binary_label"])
            df = df.merge(labels, on="review_id", how="inner")
            label_col = "binary_label"
            print(f"  After join: {len(df):,} rows")
        else:
            return

    y_true = df[label_col].values
    y_prob = df[prob_col].values

    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    print(f"  AUROC   : {auroc:.6f}")
    print(f"  AUPRC   : {auprc:.6f}")

    # Check matches
    for tag, rep in REPORTED_BINARY.items():
        auroc_ok = match_tag(auroc, rep["auroc"])
        auprc_ok = match_tag(auprc, rep["auprc"])
        if auroc_ok and auprc_ok:
            print(f"  >>> MATCH: {tag} (AUROC diff={auroc-rep['auroc']:+.6f}, AUPRC diff={auprc-rep['auprc']:+.6f})")
        elif auroc_ok or auprc_ok:
            print(f"  ~~ partial match {tag}: AUROC diff={auroc-rep['auroc']:+.6f}, AUPRC diff={auprc-rep['auprc']:+.6f}")


def evaluate_count(path: Path):
    print(f"\n{'='*80}")
    print(f"COUNT FILE: {path.relative_to(ROOT)}")
    print(f"{'='*80}")

    if not path.exists():
        print("  FILE NOT FOUND")
        return

    df = pd.read_csv(path)
    print(f"  Columns : {list(df.columns)}")
    print(f"  Rows    : {len(df):,}")

    # Determine actual and predicted columns depending on file type
    actual_col = None
    pred_col = None

    # Combined hurdle files have true_count and expected_count / thresholded_count
    if "true_count" in df.columns:
        actual_col = "true_count"
        # Prefer thresholded_count, then expected_count
        if "thresholded_count" in df.columns:
            pred_col = "thresholded_count"
        elif "expected_count" in df.columns:
            pred_col = "expected_count"
    # Stage-2 only files: y_true / y_pred or true_count / pred_count
    if "y_true" in df.columns:
        actual_col = "y_true"
        pred_col = "y_pred"
    if "pred_count" in df.columns:
        actual_col = "true_count"
        pred_col = "pred_count"

    if actual_col is None or pred_col is None:
        print(f"  Could not identify actual_col={actual_col} / pred_col={pred_col}")
        return

    y_true = df[actual_col].values.astype(float)
    y_pred = df[pred_col].values.astype(float)

    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    sp, _ = spearmanr(y_true, y_pred)

    print(f"  Using   : actual='{actual_col}', pred='{pred_col}'")
    print(f"  RMSE    : {rmse:.4f}")
    print(f"  MAE     : {mae:.4f}")
    print(f"  Spearman: {sp:.4f}")

    # For combined files, also try expected_count if we used thresholded_count
    if pred_col == "thresholded_count" and "expected_count" in df.columns:
        y_pred2 = df["expected_count"].values.astype(float)
        rmse2 = np.sqrt(np.mean((y_true - y_pred2) ** 2))
        mae2 = np.mean(np.abs(y_true - y_pred2))
        sp2, _ = spearmanr(y_true, y_pred2)
        print(f"  --- also with expected_count ---")
        print(f"  RMSE    : {rmse2:.4f}")
        print(f"  MAE     : {mae2:.4f}")
        print(f"  Spearman: {sp2:.4f}")
        # Check matches for expected_count too
        for tag, rep in REPORTED_COUNT.items():
            rmse_ok = match_tag(rmse2, rep["rmse"])
            mae_ok = match_tag(mae2, rep["mae"])
            if rmse_ok and mae_ok:
                print(f"  >>> MATCH (expected_count): {tag} (RMSE diff={rmse2-rep['rmse']:+.4f}, MAE diff={mae2-rep['mae']:+.4f})")

    # Check matches
    for tag, rep in REPORTED_COUNT.items():
        rmse_ok = match_tag(rmse, rep["rmse"])
        mae_ok = match_tag(mae, rep["mae"])
        if rmse_ok and mae_ok:
            print(f"  >>> MATCH: {tag} (RMSE diff={rmse-rep['rmse']:+.4f}, MAE diff={mae-rep['mae']:+.4f})")
        elif rmse_ok or mae_ok:
            print(f"  ~~ partial match {tag}: RMSE diff={rmse-rep['rmse']:+.4f}, MAE diff={mae-rep['mae']:+.4f}")


def main():
    print("Reported binary test-set size :", f"{BINARY_N:,}")
    print("Reported count test-set size  :", f"{COUNT_N:,}")

    print("\n" + "#" * 80)
    print("# BINARY PREDICTION FILES")
    print("#" * 80)
    for p in BINARY_CANDIDATES:
        evaluate_binary(p)

    print("\n" + "#" * 80)
    print("# COUNT / HURDLE PREDICTION FILES")
    print("#" * 80)
    for p in COUNT_CANDIDATES:
        evaluate_count(p)

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
