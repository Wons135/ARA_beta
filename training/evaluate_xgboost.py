from __future__ import annotations

import argparse
from pathlib import Path

from xgboost_common import evaluate_mode, setup_logger


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compatibility wrapper for XGBoost evaluation. Prefer the split evaluators for new work."
    )
    parser.add_argument(
        "--mode",
        choices=["binary", "count_stage1", "count_stage2_poisson", "count_stage2_log1p"],
        default="binary",
    )
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--base-dir", default=str(ROOT / "datasets" / "preprocessed"))
    parser.add_argument("--out-dir", default=str(ROOT / "outputs"))
    parser.add_argument("--target-precision", type=float, default=0.30)
    parser.add_argument("--min-recall", type=float, default=0.15)
    parser.add_argument("--model-path", default=None)
    args = parser.parse_args()

    logger = setup_logger("evaluate_xgboost")
    logger.info(
        "Compatibility wrapper in use. Prefer evaluate_xgb_binary.py, evaluate_xgb_count_stage1.py, "
        "evaluate_xgb_count_stage2.py, or evaluate_xgb_count_combined.py."
    )
    evaluate_mode(
        mode=args.mode,
        split=args.split,
        base_dir=args.base_dir,
        out_dir=args.out_dir,
        target_precision=args.target_precision,
        min_recall=args.min_recall,
        logger=logger,
        model_path_override=args.model_path,
    )


if __name__ == "__main__":
    main()
