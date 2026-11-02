from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.bert_common import compute_binary_metrics, compute_count_metrics, load_mode_dataframe, safe_path, setup_logger
from training.evaluator import BertInferenceSession
from training.main_figure_plots import try_generate_requested_main_figures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the combined two-stage BERT count pipeline.")
    parser.add_argument("--stage2-mode", choices=["count_stage2_log1p", "count_stage2_poisson"], default="count_stage2_log1p")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--base-dir", default=str(ROOT / "datasets" / "preprocessed"))
    parser.add_argument("--out-dir", default=str(ROOT / "outputs"))
    parser.add_argument("--stage1-checkpoint-path", default=None)
    parser.add_argument("--stage2-checkpoint-path", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--target-precision", type=float, default=0.30)
    parser.add_argument("--min-recall", type=float, default=0.15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logger("evaluate_bert_count_combined", ROOT / "outputs" / "logs" / "evaluate_bert_count_combined.log")

    frame = load_mode_dataframe(args.base_dir, "count_stage1", args.split)

    stage1 = BertInferenceSession(
        mode="count_stage1",
        out_dir=args.out_dir,
        checkpoint_path_override=args.stage1_checkpoint_path,
        batch_size=args.batch_size,
        logger=logger,
    ).load()
    stage2 = BertInferenceSession(
        mode=args.stage2_mode,
        out_dir=args.out_dir,
        checkpoint_path_override=args.stage2_checkpoint_path,
        batch_size=args.batch_size,
        logger=logger,
    ).load()

    stage1_outputs = stage1.predict_frame(frame)
    stage2_outputs = stage2.predict_frame(frame)

    has_vote = pd.to_numeric(frame["has_helpful_vote"], errors="raise").to_numpy(dtype=int)
    true_count = pd.to_numeric(frame["helpful_vote"], errors="raise").to_numpy(dtype=float)
    prob_has_vote = stage1_outputs["probabilities"]
    pred_positive_count = stage2_outputs["predicted_count"]
    expected_count = prob_has_vote * pred_positive_count

    stage1_metrics = compute_binary_metrics(
        has_vote,
        prob_has_vote,
        target_precision=args.target_precision,
        min_recall=args.min_recall,
    )
    combined_metrics = compute_count_metrics(true_count, expected_count, positive_mask=(true_count > 0))

    pred_path = safe_path(args.out_dir, "predictions", f"bert_count_combined_{args.stage2_mode}_{args.split}_predictions.csv")
    metrics_path = safe_path(args.out_dir, "predictions", f"bert_count_combined_{args.stage2_mode}_{args.split}_metrics.json")

    pred_df = pd.DataFrame(
        {
            "review_id": frame["review_id"].to_numpy(),
            "true_count": true_count,
            "has_helpful_vote": has_vote,
            "prob_has_helpful_vote": prob_has_vote,
            "predicted_positive_count_if_positive": pred_positive_count,
            "expected_count": expected_count,
        }
    )
    pred_df.to_csv(pred_path, index=False)

    payload = {
        "split": args.split,
        "stage2_mode": args.stage2_mode,
        "stage1_checkpoint_path": stage1.checkpoint_path,
        "stage2_checkpoint_path": stage2.checkpoint_path,
        "stage1_metrics": stage1_metrics,
        "expected_count_metrics": combined_metrics,
        "prediction_path": pred_path,
    }
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    logger.info(
        "[COMBINED] RMSE=%.4f | MAE=%.4f | R2=%.4f | Spearman=%.4f | PosRMSE=%.4f | PosMAE=%.4f",
        combined_metrics["rmse"],
        combined_metrics["mae"],
        combined_metrics["r2"],
        combined_metrics["spearman_rho"],
        combined_metrics["positive_only_rmse"],
        combined_metrics["positive_only_mae"],
    )
    logger.info(
        "[STAGE1] AUPRC=%.4f | AUROC=%.4f | F1*=%.4f @ thr=%.4f",
        stage1_metrics["auprc"],
        stage1_metrics["auroc"],
        stage1_metrics["f1_opt"],
        stage1_metrics["thr_opt"],
    )
    logger.info("Saved combined predictions -> %s", pred_path)
    logger.info("Saved combined metrics -> %s", metrics_path)
    try_generate_requested_main_figures(
        args.out_dir,
        split=args.split,
        logger=logger,
        include_binary=False,
        include_count=True,
    )


if __name__ == "__main__":
    main()
