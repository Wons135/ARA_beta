from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from experiments.weighted_rating.config import CFG
else:
    from .config import CFG


BASE_COLS = ["review_id", "parent_asin", "rating", "helpful_vote", "review_length"]


def validate_final_inputs(df: pd.DataFrame, expected_rows: int | None = None) -> None:
    if df["review_id"].duplicated().any():
        raise AssertionError("Duplicate review_id values in final inputs.")
    if df["p_bert"].isna().any():
        raise AssertionError("p_bert contains NaN.")
    if df["yhat_xgb"].isna().any():
        raise AssertionError("yhat_xgb contains NaN.")
    if not df["rating"].between(1, 5).all():
        raise AssertionError("rating values must be in [1, 5].")
    if not df["p_bert"].between(0, 1).all():
        raise AssertionError("p_bert values must be in [0, 1].")
    p_xgb_nonnull = df["p_xgb"].dropna()
    if not p_xgb_nonnull.empty and not p_xgb_nonnull.between(0, 1).all():
        raise AssertionError("Non-null p_xgb values must be in [0, 1].")
    if (df["yhat_xgb"] < 0).any() or (df["yhat_bert"] < 0).any():
        raise AssertionError("Count predictions must be non-negative.")
    if expected_rows is not None and len(df) != expected_rows:
        raise AssertionError(f"Final row count mismatch: expected {expected_rows}, actual {len(df)}.")


def _load_titles(parent_asins: set[str]) -> dict[str, str]:
    titles: dict[str, str] = {}
    with CFG.meta_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            parent = str(row.get("parent_asin", ""))
            if parent not in parent_asins:
                continue
            title = row.get("title")
            if isinstance(title, str) and title.strip():
                titles[parent] = title.strip()
            if len(titles) == len(parent_asins):
                break
    return titles


def build_inputs() -> pd.DataFrame:
    start = time.perf_counter()
    CFG.out_dir.mkdir(parents=True, exist_ok=True)
    print("Loading count-stage test base.")
    base = pd.read_csv(CFG.count_test_path, usecols=BASE_COLS)
    print("review_length source: raw token count from count_stage1_test.csv, not log-scaled.")

    xgb_count = pd.read_csv(CFG.xgb_count_predictions_path, usecols=["review_id", "expected_count"])
    xgb_count = xgb_count.rename(columns={"expected_count": "yhat_xgb"})
    missing_xgb = pd.Index(base["review_id"]).difference(pd.Index(xgb_count["review_id"]))
    pd.DataFrame({"review_id": missing_xgb}).to_csv(CFG.dropped_xgb_count_ids_path, index=False)
    print(f"Dropping {len(missing_xgb)} reviews without XGBoost count predictions.")
    print(f"Dropped review IDs -> {CFG.dropped_xgb_count_ids_path}")

    df = base.merge(xgb_count, on="review_id", how="inner", validate="one_to_one")

    bert_count = pd.read_csv(CFG.bert_count_predictions_path, usecols=["review_id", "expected_count"])
    bert_count = bert_count.rename(columns={"expected_count": "yhat_bert"})
    df = df.merge(bert_count, on="review_id", how="left", validate="one_to_one")

    saved_bert = pd.read_csv(CFG.bert_binary_predictions_path, usecols=["review_id", "prob_pos"])
    saved_bert = saved_bert.rename(columns={"prob_pos": "p_bert"})
    saved_bert["pred_source"] = "saved"
    inferred_bert = pd.read_csv(CFG.inferred_bert_path, usecols=["review_id", "prob_pos"])
    inferred_bert = inferred_bert.rename(columns={"prob_pos": "p_bert"})
    inferred_bert["pred_source"] = "inferred"
    bert_probs = pd.concat([saved_bert, inferred_bert], ignore_index=True)
    if bert_probs["review_id"].duplicated().any():
        raise AssertionError("Saved and inferred BERT probabilities overlap.")
    df = df.merge(bert_probs, on="review_id", how="left", validate="one_to_one")

    xgb_binary = pd.read_csv(CFG.xgb_binary_predictions_path, usecols=["review_id", "prob_pos"])
    xgb_binary = xgb_binary.rename(columns={"prob_pos": "p_xgb"})
    df = df.merge(xgb_binary, on="review_id", how="left", validate="one_to_one")

    parents = set(df["parent_asin"].astype(str))
    print(f"Loading titles for {len(parents)} parent_asin values.")
    title_map = _load_titles(parents)
    df["title"] = df["parent_asin"].astype(str).map(title_map)
    df["title"] = df["title"].where(df["title"].notna() & (df["title"].astype(str).str.strip() != ""), df["parent_asin"])

    ordered = [
        "review_id",
        "parent_asin",
        "rating",
        "helpful_vote",
        "review_length",
        "p_bert",
        "pred_source",
        "yhat_xgb",
        "yhat_bert",
        "p_xgb",
        "title",
    ]
    df = df.loc[:, ordered].sort_values("review_id").reset_index(drop=True)
    validate_final_inputs(df, expected_rows=CFG.expected_final_rows)
    df.to_csv(CFG.review_level_inputs_path, index=False)
    print(f"Expected rows: {CFG.expected_final_rows:,}; actual rows: {len(df):,}")
    print(f"Saved review-level inputs -> {CFG.review_level_inputs_path}")
    print(f"Build inputs time: {time.perf_counter() - start:.1f}s")
    return df


def main() -> None:
    build_inputs()


if __name__ == "__main__":
    main()
