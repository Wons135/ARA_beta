from __future__ import annotations

import importlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from experiments.weighted_rating import build_inputs, export_latex, infer_bert_missing, weighted_rating
    from experiments.weighted_rating.config import CFG
else:
    from . import build_inputs, export_latex, infer_bert_missing, weighted_rating
    from .config import CFG


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=CFG.root, text=True).strip()
    except Exception:
        return None


def _versions() -> dict[str, str]:
    modules = ["numpy", "pandas", "scipy", "xgboost", "torch", "transformers", "sklearn"]
    versions = {"python": sys.version}
    for name in modules:
        try:
            module = importlib.import_module(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except Exception as exc:
            versions[name] = f"IMPORT_FAILED: {exc!r}"
    return versions


def run_all() -> None:
    CFG.out_dir.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}

    start = time.perf_counter()
    session = infer_bert_missing._load_session(CFG.bert_batch_size)
    gate = infer_bert_missing.run_gate(session)
    if CFG.inferred_bert_path.exists():
        print(f"Skipping BERT inference because {CFG.inferred_bert_path} exists and gate passed.")
        infer_meta = {"skipped": True, "existing_path": str(CFG.inferred_bert_path)}
    else:
        infer_meta = infer_bert_missing.infer_missing(session, CFG.bert_batch_size)
    timings["infer"] = time.perf_counter() - start

    start = time.perf_counter()
    inputs = build_inputs.build_inputs()
    timings["build_inputs"] = time.perf_counter() - start

    start = time.perf_counter()
    diagnostics = weighted_rating.run_analysis(
        path=CFG.review_level_inputs_path,
        out_dir=CFG.out_dir,
        n_min=CFG.n_min,
        tau=CFG.bert_validation_threshold(),
        n_perm=CFG.n_perm,
        seed=CFG.seed,
    )
    timings["analyse"] = time.perf_counter() - start

    start = time.perf_counter()
    export_latex.export_tables(CFG.out_dir)
    timings["export_latex"] = time.perf_counter() - start

    pred_source_counts = inputs["pred_source"].value_counts().to_dict()
    run_info = {
        "git_commit": _git_commit(),
        "versions": _versions(),
        "config": CFG.as_serializable_dict(),
        "timings_seconds": timings,
        "bert_gate": gate,
        "bert_inference": infer_meta,
        "review_level_rows": int(len(inputs)),
        "pred_source_counts": {str(k): int(v) for k, v in pred_source_counts.items()},
        "diagnostics": diagnostics,
    }
    CFG.run_info_path.write_text(json.dumps(run_info, indent=2), encoding="utf-8")
    print(f"Run info -> {CFG.run_info_path}")


def main() -> None:
    np.random.seed(CFG.seed)
    run_all()


if __name__ == "__main__":
    main()
