from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.bert_common import setup_logger
from training.evaluator import EvalConfig, Evaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a BERT baseline on binary or count-helpfulness tasks.")
    parser.add_argument(
        "--mode",
        choices=["binary", "count_stage1", "count_stage2_log1p", "count_stage2_poisson"],
        default="binary",
    )
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--base-dir", default=str(ROOT / "datasets" / "preprocessed"))
    parser.add_argument("--out-dir", default=str(ROOT / "outputs"))
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--tokenizer-name", default=None)
    parser.add_argument("--max-len", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--disable-aux", action="store_true")
    parser.add_argument("--target-precision", type=float, default=0.30)
    parser.add_argument("--min-recall", type=float, default=0.15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logger("evaluate_bert", ROOT / "outputs" / "logs" / "evaluate_bert.log")
    cfg = EvalConfig(
        mode=args.mode,
        split=args.split,
        base_dir=args.base_dir,
        out_dir=args.out_dir,
        checkpoint_path=args.checkpoint_path,
        tokenizer_name=args.tokenizer_name,
        max_len=args.max_len,
        batch_size=args.batch_size,
        use_aux=(False if args.disable_aux else None),
        target_precision=args.target_precision,
        min_recall=args.min_recall,
    )
    Evaluator(cfg, logger=logger).run()


if __name__ == "__main__":
    main()
