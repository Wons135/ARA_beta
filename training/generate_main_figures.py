from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.bert_common import setup_logger
from training.main_figure_plots import try_generate_requested_main_figures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the main comparison figures from saved evaluation artifacts.")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--out-dir", default=str(ROOT / "outputs"))
    parser.add_argument("--target-precision", type=float, default=0.30)
    parser.add_argument("--min-recall", type=float, default=0.15)
    parser.add_argument("--skip-binary", action="store_true")
    parser.add_argument("--skip-count", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logger("generate_main_figures", ROOT / "outputs" / "logs" / "generate_main_figures.log")
    generated = try_generate_requested_main_figures(
        args.out_dir,
        split=args.split,
        logger=logger,
        include_binary=not args.skip_binary,
        include_count=not args.skip_count,
        target_precision=args.target_precision,
        min_recall=args.min_recall,
    )
    if not generated:
        logger.info("No main figures were generated. Check whether the prerequisite evaluation artifacts exist.")


if __name__ == "__main__":
    main()
