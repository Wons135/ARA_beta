from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class WeightedRatingConfig:
    root: Path = ROOT
    data_dir: Path = ROOT / "datasets" / "preprocessed"
    outputs_dir: Path = ROOT / "outputs"
    out_dir: Path = ROOT / "outputs" / "weighted_rating"
    count_test_path: Path = ROOT / "datasets" / "preprocessed" / "count_stage1_test.csv"
    xgb_count_predictions_path: Path = (
        ROOT
        / "outputs"
        / "predictions"
        / "xgb_count_stage1_test_combined_count_stage2_log1p_predictions.csv"
    )
    bert_count_predictions_path: Path = (
        ROOT
        / "outputs"
        / "predictions"
        / "bert_count_combined_count_stage2_log1p_test_predictions.csv"
    )
    xgb_binary_predictions_path: Path = ROOT / "outputs" / "predictions" / "xgb_binary_test_predictions.csv"
    bert_binary_predictions_path: Path = ROOT / "outputs" / "predictions" / "bert_binary_test_predictions.csv"
    bert_binary_checkpoint_path: Path = ROOT / "outputs" / "checkpoints" / "bert_binary_best.pth"
    xgb_binary_threshold_path: Path = ROOT / "outputs" / "checkpoints" / "xgb_binary_thresholds.json"
    meta_path: Path = ROOT / "datasets" / "meta_Movies_and_TV.jsonl"
    inferred_bert_path: Path = ROOT / "outputs" / "weighted_rating" / "bert_binary_inferred_greyband.csv"
    review_level_inputs_path: Path = ROOT / "outputs" / "weighted_rating" / "review_level_inputs.csv"
    dropped_xgb_count_ids_path: Path = ROOT / "outputs" / "weighted_rating" / "dropped_missing_xgb_count_review_ids.csv"
    threshold_audit_path: Path = ROOT / "outputs" / "weighted_rating" / "threshold_audit.csv"
    run_info_path: Path = ROOT / "outputs" / "weighted_rating" / "run_info.json"
    tables_dir: Path = ROOT / "outputs" / "weighted_rating" / "tables"
    n_min: int = 20
    n_perm: int = 100
    seed: int = 42
    bert_gate_sample_n: int = 5_000
    bert_gate_max_abs_diff: float = 1e-4
    bert_batch_size: int = 64
    expected_count_test_rows: int = 1_653_016
    expected_xgb_count_missing: int = 26
    main_weight: str = "p_bert"

    @property
    def expected_final_rows(self) -> int:
        return self.expected_count_test_rows - self.expected_xgb_count_missing

    def bert_validation_threshold(self) -> float:
        checkpoint = torch.load(self.bert_binary_checkpoint_path, map_location="cpu", weights_only=True)
        return float(checkpoint["decision_threshold"])

    def as_serializable_dict(self) -> dict[str, object]:
        payload = asdict(self)
        for key, value in list(payload.items()):
            if isinstance(value, Path):
                payload[key] = str(value)
        payload["bert_validation_threshold"] = self.bert_validation_threshold()
        payload["expected_final_rows"] = self.expected_final_rows
        return payload


CFG = WeightedRatingConfig()
