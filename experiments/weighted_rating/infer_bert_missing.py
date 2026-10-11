from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from experiments.weighted_rating.config import CFG
else:
    from .config import CFG

ROOT = CFG.root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.bert_common import setup_logger  # noqa: E402
from training.evaluator import BertInferenceSession  # noqa: E402


GATE_INPUT_COLS = ["review_id", "full_text", "binary_label", "review_length", "sentiment_score", "is_verified"]
MISSING_INPUT_COLS = ["review_id", "full_text", "review_length", "sentiment_score", "is_verified"]


def _load_rows_for_ids(path: Path, ids: set[int], usecols: list[str], chunksize: int = 100_000) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
        hit = chunk[chunk["review_id"].isin(ids)]
        if not hit.empty:
            parts.append(hit)
    if not parts:
        return pd.DataFrame(columns=usecols)
    return pd.concat(parts, ignore_index=True)


def _load_session(batch_size: int) -> BertInferenceSession:
    logger = setup_logger("weighted_rating_bert_infer", CFG.out_dir / "bert_binary_infer.log")
    return BertInferenceSession(
        mode="binary",
        out_dir=str(CFG.outputs_dir),
        checkpoint_path_override=str(CFG.bert_binary_checkpoint_path),
        batch_size=batch_size,
        logger=logger,
    ).load()


def run_gate(session: BertInferenceSession, sample_n: int = CFG.bert_gate_sample_n) -> dict[str, object]:
    saved = pd.read_csv(CFG.bert_binary_predictions_path, usecols=["review_id", "prob_pos"])
    sample = saved.sample(n=sample_n, random_state=CFG.seed).reset_index(drop=True)
    source = _load_rows_for_ids(CFG.data_dir / "binary_label_test.csv", set(sample["review_id"]), GATE_INPUT_COLS)
    source = sample[["review_id"]].merge(source, on="review_id", how="left", validate="one_to_one")
    if source[GATE_INPUT_COLS].isna().any().any():
        raise RuntimeError("BERT gate sample has missing source columns after merge.")

    start = time.perf_counter()
    with torch.no_grad():
        outputs = session.predict_frame(source)
    seconds = time.perf_counter() - start
    probs = np.asarray(outputs["probabilities"], dtype=float)
    diff = np.abs(probs - sample["prob_pos"].to_numpy(dtype=float))
    result = {
        "sample_n": int(sample_n),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "seconds": float(seconds),
        "rows_per_second": float(sample_n / seconds) if seconds > 0 else None,
        "pass": bool(diff.max() < CFG.bert_gate_max_abs_diff),
    }
    if not result["pass"]:
        raise RuntimeError(f"BERT gate failed: {result}")
    return result


def infer_missing(session: BertInferenceSession, batch_size: int) -> dict[str, object]:
    saved_ids = pd.read_csv(CFG.bert_binary_predictions_path, usecols=["review_id"])
    count_ids = pd.read_csv(CFG.count_test_path, usecols=["review_id"])
    missing_ids = pd.Index(count_ids["review_id"]).difference(pd.Index(saved_ids["review_id"]))
    source = _load_rows_for_ids(CFG.count_test_path, set(missing_ids.astype(int)), MISSING_INPUT_COLS)
    source = pd.DataFrame({"review_id": missing_ids}).merge(source, on="review_id", how="left", validate="one_to_one")
    if len(source) != len(missing_ids):
        raise RuntimeError(f"Expected {len(missing_ids)} missing rows, loaded {len(source)}.")
    if source[MISSING_INPUT_COLS].isna().any().any():
        raise RuntimeError("Missing BERT inference rows have null source columns.")
    source["binary_label"] = 0

    start = time.perf_counter()
    with torch.no_grad():
        outputs = session.predict_frame(source)
    seconds = time.perf_counter() - start
    out = pd.DataFrame({"review_id": source["review_id"].to_numpy(), "prob_pos": outputs["probabilities"]})
    out.to_csv(CFG.inferred_bert_path, index=False)
    meta = {
        "rows": int(len(out)),
        "batch_size": int(batch_size),
        "seconds": float(seconds),
        "rows_per_second": float(len(out) / seconds) if seconds > 0 else None,
        "device": str(session.device),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    (CFG.out_dir / "bert_binary_inferred_greyband_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Infer DistilBERT binary probabilities for grey-band count-test reviews.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing inferred probabilities.")
    parser.add_argument("--batch-size", type=int, default=CFG.bert_batch_size)
    args = parser.parse_args()

    CFG.out_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(CFG.seed)
    torch.manual_seed(CFG.seed)
    session = _load_session(args.batch_size)
    gate = run_gate(session)
    print("BERT gate:", json.dumps(gate, sort_keys=True))

    if CFG.inferred_bert_path.exists() and not args.force:
        print(f"Skipping inference because {CFG.inferred_bert_path} exists and gate passed.")
        return

    meta = infer_missing(session, args.batch_size)
    print("Inference:", json.dumps(meta, sort_keys=True))


if __name__ == "__main__":
    main()
