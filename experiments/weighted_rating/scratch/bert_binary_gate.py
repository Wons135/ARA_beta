from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.bert_common import setup_logger  # noqa: E402
from training.evaluator import BertInferenceSession  # noqa: E402


DATA = ROOT / "datasets" / "preprocessed"
OUT = ROOT / "outputs"


def rel(path: Path | str) -> str:
    return str(Path(path).resolve().relative_to(ROOT))


def load_rows_for_ids(path: Path, ids: set[int], usecols: list[str], chunksize: int = 100_000) -> pd.DataFrame:
    parts = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
        hit = chunk[chunk["review_id"].isin(ids)]
        if not hit.empty:
            parts.append(hit)
    if not parts:
        return pd.DataFrame(columns=usecols)
    return pd.concat(parts, ignore_index=True)


def checkpoint_summary(path: Path) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    keys = [
        "mode",
        "task",
        "checkpoint_role",
        "selection_metric",
        "selection_score",
        "decision_threshold",
        "epoch",
        "model_name",
        "tokenizer_name",
        "max_len",
        "use_aux_features",
        "aux_columns",
        "aux_means",
        "aux_stds",
    ]
    return {k: ckpt[k] for k in keys if k in ckpt}


def main() -> None:
    prediction_path = OUT / "predictions" / "bert_binary_test_predictions.csv"
    checkpoint_path = OUT / "checkpoints" / "bert_binary_best.pth"
    sample_n = 5_000
    seed = 42
    batch_size = 64

    print("=== BERT_BINARY_CHECKPOINT ===")
    summary = checkpoint_summary(checkpoint_path)
    print(json.dumps(summary, sort_keys=True))
    print(
        "gpu:",
        json.dumps(
            {
                "available": bool(torch.cuda.is_available()),
                "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
                "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            },
            sort_keys=True,
        ),
    )

    saved = pd.read_csv(prediction_path, usecols=["review_id", "prob_pos"])
    sample = saved.sample(n=sample_n, random_state=seed).reset_index(drop=True)
    ids = set(sample["review_id"].astype(int))
    usecols = ["review_id", "full_text", "binary_label", "review_length", "sentiment_score", "is_verified"]
    rows = load_rows_for_ids(DATA / "binary_label_test.csv", ids, usecols)
    if len(rows) != sample_n:
        raise RuntimeError(f"Expected {sample_n} source rows, found {len(rows)}")
    rows = sample[["review_id"]].merge(rows, on="review_id", how="left")
    if rows[usecols].isna().any().any():
        raise RuntimeError("Sample merge introduced missing BERT inputs.")

    logger = setup_logger("phase1_bert_binary_gate")
    session = BertInferenceSession(
        mode="binary",
        out_dir=str(OUT),
        checkpoint_path_override=str(checkpoint_path),
        batch_size=batch_size,
        logger=logger,
    ).load()
    t0 = time.time()
    outputs = session.predict_frame(rows)
    seconds = time.time() - t0
    pred = np.asarray(outputs["probabilities"], dtype=float)
    diff = np.abs(pred - sample["prob_pos"].to_numpy(dtype=float))
    rows_per_second = sample_n / seconds if seconds > 0 else float("inf")
    missing_n = 217_414
    result = {
        "sample_n": sample_n,
        "seed": seed,
        "batch_size": batch_size,
        "saved_prediction_path": rel(prediction_path),
        "checkpoint_path": rel(checkpoint_path),
        "device": str(session.device),
        "seconds": seconds,
        "rows_per_second": rows_per_second,
        "estimated_seconds_for_217414": missing_n / rows_per_second if rows_per_second > 0 else None,
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "pass": bool(diff.max() < 1e-4),
    }
    print("=== BERT_BINARY_GATE ===")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
