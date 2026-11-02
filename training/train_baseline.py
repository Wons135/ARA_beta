from __future__ import annotations

import argparse
from pathlib import Path

from baseline_common import (
    ModeT,
    checkpoint_path,
    get_spec,
    load_split_frame,
    save_model,
    setup_logger,
    train_binary_baseline,
    train_regression_baseline,
)


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline models for regression and binary classification.")
    parser.add_argument("--mode", choices=["regression", "binary"], default="binary")
    parser.add_argument("--base-dir", default=str(ROOT / "datasets" / "preprocessed"))
    parser.add_argument("--out-dir", default=str(ROOT / "outputs"))
    parser.add_argument("--regression-target-column", default="regression_score")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mode: ModeT = args.mode
    spec = get_spec(mode)

    logger = setup_logger("train_baseline", ROOT / "outputs" / "logs" / "train_baseline.log")
    logger.info("Training baseline mode=%s", mode)

    train_df = load_split_frame(
        args.base_dir,
        "train",
        mode,
        regression_target_column=args.regression_target_column,
    )
    logger.info(
        "Loaded %d train rows from %s_train.csv using target_column=%s",
        int(train_df.shape[0]),
        spec.dataset_prefix,
        args.regression_target_column if mode == "regression" else spec.target_column,
    )

    if mode == "regression":
        model = train_regression_baseline(train_df, target_column=args.regression_target_column)
        logger.info("Mean baseline value=%.6f", model.constant_value)
    else:
        model = train_binary_baseline(train_df, target_column=spec.target_column)
        logger.info(
            "Majority class=%d | positive_rate=%.6f",
            int(model.majority_class or 0),
            float(model.positive_rate or 0.0),
        )

    model_path = checkpoint_path(args.out_dir, mode)
    save_model(model, model_path)
    logger.info("Saved baseline artifact -> %s", model_path)


if __name__ == "__main__":
    main()
