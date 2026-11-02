from __future__ import annotations

import argparse
from pathlib import Path

from xgboost_common import evaluate_mode, setup_logger


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the binary XGBoost baseline.")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--base-dir", default=str(ROOT / "datasets" / "preprocessed"))
    parser.add_argument("--out-dir", default=str(ROOT / "outputs"))
    parser.add_argument("--target-precision", type=float, default=0.30)
    parser.add_argument("--min-recall", type=float, default=0.15)
    parser.add_argument("--checkpoint-path", "--model-path", dest="checkpoint_path", default=None)
    args = parser.parse_args()

    logger = setup_logger("evaluate_xgb_binary", ROOT / "outputs" / "logs" / "evaluate_xgb_binary.log")
    evaluate_mode(
        mode="binary",
        split=args.split,
        base_dir=args.base_dir,
        out_dir=args.out_dir,
        target_precision=args.target_precision,
        min_recall=args.min_recall,
        logger=logger,
        model_path_override=args.checkpoint_path,
    )


if __name__ == "__main__":
    main()
