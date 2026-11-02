from __future__ import annotations

import argparse
from pathlib import Path

from baseline_common import (
    ModeT,
    checkpoint_path,
    evaluate_binary,
    evaluate_regression,
    load_model,
    load_split_frame,
    metrics_json_path,
    prediction_csv_path,
    save_json,
    setup_logger,
)


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate baseline models for regression and binary classification.")
    parser.add_argument("--mode", choices=["regression", "binary"], default="binary")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--base-dir", default=str(ROOT / "datasets" / "preprocessed"))
    parser.add_argument("--out-dir", default=str(ROOT / "outputs"))
    parser.add_argument("--checkpoint-path", "--model-path", dest="checkpoint_path", default=None)
    parser.add_argument("--regression-target-column", default="regression_score")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mode: ModeT = args.mode
    logger = setup_logger("evaluate_baseline", ROOT / "outputs" / "logs" / "evaluate_baseline.log")

    model_path = args.checkpoint_path or checkpoint_path(args.out_dir, mode)
    model = load_model(model_path)
    logger.info("Loaded baseline mode=%s from %s", mode, model_path)

    split_df = load_split_frame(
        args.base_dir,
        args.split,
        mode,
        regression_target_column=args.regression_target_column,
    )
    logger.info("Loaded %d %s rows", int(split_df.shape[0]), args.split)

    if mode == "regression":
        prediction_df, metrics = evaluate_regression(split_df, model)
    else:
        prediction_df, metrics = evaluate_binary(split_df, model)

    prediction_path = prediction_csv_path(args.out_dir, mode, args.split)
    prediction_df.to_csv(prediction_path, index=False)

    payload = {
        "summary": {
            "mode": mode,
            "split": args.split,
            "rows": int(split_df.shape[0]),
            "model_path": model_path,
            "baseline_name": model.baseline_name,
            "target_column": model.target_column,
        },
        "metrics": metrics,
    }
    save_json(metrics_json_path(args.out_dir, mode, args.split), payload)

    for key, value in metrics.items():
        logger.info("%s=%s", key, f"{value:.6f}" if isinstance(value, float) else value)
    logger.info("Saved predictions -> %s", prediction_path)


if __name__ == "__main__":
    main()
