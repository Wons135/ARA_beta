from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.feature_extraction import extract_features, setup_logger  # noqa: E402
from training.xgboost_common import _booster_expected_schema, load_booster  # noqa: E402


DATA = ROOT / "datasets" / "preprocessed"
OUT = ROOT / "outputs"
SCRATCH = ROOT / "experiments" / "weighted_rating" / "scratch"


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def load_rows_for_ids(path: Path, ids: set[int], usecols: list[str], chunksize: int = 100_000) -> pd.DataFrame:
    parts = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
        hit = chunk[chunk["review_id"].isin(ids)]
        if not hit.empty:
            parts.append(hit)
    if not parts:
        return pd.DataFrame(columns=usecols)
    return pd.concat(parts, ignore_index=True)


def print_artifact_summary() -> None:
    print("=== XGB_BINARY_PIPELINE_ARTIFACTS ===")
    meta = json.loads((DATA / "labels_meta_binary.json").read_text(encoding="utf-8"))
    print("labels_meta_binary:", json.dumps(meta, sort_keys=True))
    for name in ["tfidf_binary.pkl", "scaler_binary.pkl", "svd_binary.pkl", "selector_binary.pkl"]:
        path = OUT / "feature_models" / name
        print(f"artifact_exists:{rel(path)}={path.exists()}")
    tfidf = joblib.load(OUT / "feature_models" / "tfidf_binary.pkl")
    scaler = joblib.load(OUT / "feature_models" / "scaler_binary.pkl")
    svd = joblib.load(OUT / "feature_models" / "svd_binary.pkl")
    print(
        "tfidf:",
        json.dumps(
            {
                "class": type(tfidf).__name__,
                "max_features": tfidf.max_features,
                "ngram_range": tfidf.ngram_range,
                "min_df": tfidf.min_df,
                "sublinear_tf": tfidf.sublinear_tf,
                "vocab_size": len(tfidf.vocabulary_),
            },
            sort_keys=True,
        ),
    )
    print(
        "scaler:",
        json.dumps(
            {
                "class": type(scaler).__name__,
                "n_features_in": int(getattr(scaler, "n_features_in_", -1)),
                "mean": [float(x) for x in scaler.mean_],
                "scale": [float(x) for x in scaler.scale_],
            },
            sort_keys=True,
        ),
    )
    print(
        "svd:",
        json.dumps(
            {
                "class": type(svd).__name__,
                "n_components": int(svd.n_components),
                "n_features_in": int(getattr(svd, "n_features_in_", -1)),
                "random_state": int(svd.random_state),
            },
            sort_keys=True,
        ),
    )


def main() -> None:
    print_artifact_summary()
    prediction_path = OUT / "predictions" / "xgb_binary_test_predictions.csv"
    model_path = OUT / "checkpoints" / "xgb_binary.json"
    sample_n = 50_000
    seed = 42

    saved = pd.read_csv(prediction_path, usecols=["review_id", "prob_pos"])
    sample = saved.sample(n=sample_n, random_state=seed).reset_index(drop=True)
    ids = set(sample["review_id"].astype(int))
    usecols = [
        "review_id",
        "full_text",
        "sentiment_score",
        "review_length",
        "is_verified",
        "timestamp_recency_days",
    ]
    rows = load_rows_for_ids(DATA / "binary_label_test.csv", ids, usecols)
    if len(rows) != sample_n:
        raise RuntimeError(f"Expected {sample_n} source rows, found {len(rows)}")
    rows = sample[["review_id"]].merge(rows, on="review_id", how="left")
    if rows[usecols].isna().any().any():
        raise RuntimeError("Sample merge introduced missing feature inputs.")

    logger = setup_logger("INFO")
    t0 = time.time()
    feature_path = extract_features(
        rows,
        "full_text",
        "phase1_xgb_gate",
        False,
        SCRATCH,
        SCRATCH,
        OUT / "feature_models",
        "binary",
        False,
        logger,
    )
    transform_seconds = time.time() - t0
    X = np.load(feature_path, mmap_mode="r")
    booster = load_booster(model_path)
    expected_cols, feature_names = _booster_expected_schema(booster)
    dmatrix = xgb.DMatrix(X, feature_names=feature_names if feature_names else None)
    t1 = time.time()
    pred = booster.predict(dmatrix, validate_features=True).astype(float)
    predict_seconds = time.time() - t1

    compare = sample.copy()
    compare["recomputed_prob_pos"] = pred
    compare["abs_diff"] = (compare["recomputed_prob_pos"] - compare["prob_pos"]).abs()
    result = {
        "sample_n": sample_n,
        "seed": seed,
        "saved_prediction_path": rel(prediction_path),
        "model_path": rel(model_path),
        "feature_path": rel(Path(feature_path)),
        "feature_shape": [int(X.shape[0]), int(X.shape[1])],
        "model_expected_cols": int(expected_cols),
        "transform_seconds": transform_seconds,
        "predict_seconds": predict_seconds,
        "max_abs_diff": float(compare["abs_diff"].max()),
        "mean_abs_diff": float(compare["abs_diff"].mean()),
        "pass": bool(compare["abs_diff"].max() < 1e-5),
    }
    print("=== XGB_BINARY_GATE ===")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
