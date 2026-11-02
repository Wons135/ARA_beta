from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from xgboost_common import (
    COUNT_STAGE2_MODE_CHOICES,
    checkpoint_path,
    compute_binary_metrics,
    compute_count_metrics,
    get_mode_spec,
    load_booster,
    load_json,
    load_split_bundle,
    prediction_csv_path,
    safe_path,
    save_json,
    setup_logger,
    threshold_path,
    to_raw_count_predictions,
)
from main_figure_plots import try_generate_requested_main_figures


ROOT = Path(__file__).resolve().parents[1]


def _load_threshold(out_dir: str, fallback_threshold: float) -> float:
    path = threshold_path(out_dir, "count_stage1")
    threshold = fallback_threshold
    if Path(path).exists():
        threshold = float(load_json(path)["threshold"])
    return threshold


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the combined two-stage count XGBoost pipeline.")
    parser.add_argument("--stage2-mode", choices=COUNT_STAGE2_MODE_CHOICES, default="count_stage2_log1p")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--base-dir", default=str(ROOT / "datasets" / "preprocessed"))
    parser.add_argument("--out-dir", default=str(ROOT / "outputs"))
    parser.add_argument("--target-precision", type=float, default=0.30)
    parser.add_argument("--min-recall", type=float, default=0.15)
    parser.add_argument("--stage1-checkpoint-path", "--stage1-model-path", dest="stage1_checkpoint_path", default=None)
    parser.add_argument("--stage2-checkpoint-path", "--stage2-model-path", dest="stage2_checkpoint_path", default=None)
    args = parser.parse_args()

    logger = setup_logger("evaluate_xgb_count_combined", ROOT / "outputs" / "logs" / "evaluate_xgb_count_combined.log")

    stage1_bundle = load_split_bundle(
        args.base_dir,
        args.split,
        "count_stage1",
        logger,
        extra_columns=["helpful_vote"],
    )
    stage1_model_path = args.stage1_checkpoint_path or checkpoint_path(args.out_dir, "count_stage1")
    stage2_model_path = args.stage2_checkpoint_path or checkpoint_path(args.out_dir, args.stage2_mode)

    stage1_booster = load_booster(stage1_model_path)
    stage2_booster = load_booster(stage2_model_path)
    stage2_spec = get_mode_spec(args.stage2_mode)

    dsplit = xgb.DMatrix(stage1_bundle.X)
    stage1_probs = stage1_booster.predict(dsplit, validate_features=True).astype(float)
    stage2_raw = stage2_booster.predict(dsplit, validate_features=True).astype(float)
    stage2_expected_pos = to_raw_count_predictions(stage2_raw, stage2_spec)

    expected_count = stage1_probs * stage2_expected_pos
    actual_count = stage1_bundle.frame["helpful_vote"].to_numpy(dtype=float)
    positive_mask = actual_count > 0

    stage1_metrics = compute_binary_metrics(
        stage1_bundle.y.astype(int),
        stage1_probs,
        target_precision=args.target_precision,
        min_recall=args.min_recall,
    )
    expected_metrics = compute_count_metrics(actual_count, expected_count, positive_mask=positive_mask)

    opt_threshold = _load_threshold(args.out_dir, fallback_threshold=stage1_metrics["thr_opt"])
    thresholded_count = np.where(stage1_probs >= opt_threshold, stage2_expected_pos, 0.0)
    thresholded_metrics = compute_count_metrics(actual_count, thresholded_count, positive_mask=positive_mask)

    pred_df = pd.DataFrame(
        {
            "review_id": stage1_bundle.frame["review_id"].to_numpy(),
            "true_count": actual_count,
            "has_helpful_vote": stage1_bundle.y.astype(int),
            "prob_has_helpful_vote": stage1_probs,
            "predicted_positive_count_if_positive": stage2_expected_pos,
            "expected_count": expected_count,
            "thresholded_count": thresholded_count,
        }
    )
    pred_path = prediction_csv_path(args.out_dir, "count_stage1", f"{args.split}_combined_{args.stage2_mode}")
    pred_df.to_csv(pred_path, index=False)

    payload = {
        "split": args.split,
        "stage1_model_path": stage1_model_path,
        "stage2_model_path": stage2_model_path,
        "stage2_mode": args.stage2_mode,
        "stage1_metrics": stage1_metrics,
        "expected_count_metrics": expected_metrics,
        "thresholded_count_metrics": thresholded_metrics,
        "threshold_used": float(opt_threshold),
        "prediction_path": pred_path,
    }
    metrics_path = safe_path(args.out_dir, "predictions", f"xgb_count_combined_{args.stage2_mode}_{args.split}_metrics.json")
    save_json(metrics_path, payload)

    logger.info(
        "[COMBINED] expected-count RMSE=%.4f | MAE=%.4f | R2=%.4f | Spearman=%.4f | PosRMSE=%.4f | PosMAE=%.4f",
        expected_metrics["rmse"],
        expected_metrics["mae"],
        expected_metrics["r2"],
        expected_metrics["spearman_rho"],
        expected_metrics["positive_only_rmse"],
        expected_metrics["positive_only_mae"],
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
