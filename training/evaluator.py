from __future__ import annotations

import json
import logging
import os
import pickle
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.bert_model import BERTClassifier, BERTRegressor
from src.dataset import ReviewDataset, make_collate
from training.bert_common import (
    BertModeT,
    apply_target_transform,
    checkpoint_path,
    compute_binary_metrics,
    compute_count_metrics,
    get_mode_spec,
    invert_predictions,
    load_mode_dataframe,
    metrics_path,
    predictions_path,
)


def _json_ready(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_ready(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def device_aware_loader(
    ds: ReviewDataset,
    *,
    batch_size: int,
    device: torch.device,
    shuffle: bool = False,
    collate_fn=None,
) -> DataLoader:
    is_cuda = device.type == "cuda"
    is_windows = os.name == "nt"
    num_workers = 0 if is_windows else 2
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=is_cuda,
        persistent_workers=(num_workers > 0),
        drop_last=False,
        collate_fn=collate_fn,
    )


@dataclass(frozen=True)
class EvalConfig:
    mode: BertModeT = "binary"
    split: str = "test"
    base_dir: str = str(ROOT / "datasets" / "preprocessed")
    out_dir: str = str(ROOT / "outputs")
    checkpoint_path: Optional[str] = None
    tokenizer_name: Optional[str] = None
    max_len: Optional[int] = None
    batch_size: int = 32
    use_aux: Optional[bool] = None
    aux_columns: tuple[str, ...] = ("review_length", "sentiment_score", "is_verified")
    target_precision: float = 0.30
    min_recall: float = 0.15


class BertInferenceSession:
    def __init__(
        self,
        *,
        mode: BertModeT,
        out_dir: str,
        checkpoint_path_override: Optional[str] = None,
        tokenizer_name_override: Optional[str] = None,
        max_len_override: Optional[int] = None,
        use_aux_override: Optional[bool] = None,
        aux_columns_override: Optional[tuple[str, ...]] = None,
        batch_size: int = 32,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.mode = mode
        self.spec = get_mode_spec(mode)
        self.out_dir = out_dir
        self.checkpoint_path_override = checkpoint_path_override
        self.tokenizer_name_override = tokenizer_name_override
        self.max_len_override = max_len_override
        self.use_aux_override = use_aux_override
        self.aux_columns_override = aux_columns_override
        self.batch_size = batch_size
        self.logger = logger or logging.getLogger(f"bert_eval_{mode}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.checkpoint_path: Optional[str] = None
        self.checkpoint_payload: dict[str, Any] | None = None
        self.transform_meta: dict[str, Any] | None = None
        self.tokenizer_name: str = "distilbert-base-uncased"
        self.max_len: int = 128
        self.use_aux: bool = True
        self.aux_columns: tuple[str, ...] = ("review_length", "sentiment_score", "is_verified")
        self.aux_means: dict[str, float] = {}
        self.aux_stds: dict[str, float] = {}
        self.model: nn.Module | None = None
        self.criterion: nn.Module | None = None

    def _load_checkpoint(self, path: str) -> dict[str, Any]:
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
        except (TypeError, pickle.UnpicklingError):
            ckpt = torch.load(path, map_location="cpu")
        except Exception as exc:
            if "Weights only load failed" not in str(exc):
                raise
            ckpt = torch.load(path, map_location="cpu")
        if not isinstance(ckpt, dict):
            raise ValueError(f"Unexpected checkpoint payload at {path}.")
        return ckpt

    def _normalize_state_dict_keys(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        remapped = dict(state_dict)
        legacy_to_current = {
            "norm.weight": "text_norm.weight",
            "norm.bias": "text_norm.bias",
        }
        for legacy_key, current_key in legacy_to_current.items():
            if legacy_key in remapped and current_key not in remapped:
                remapped[current_key] = remapped.pop(legacy_key)
                self.logger.info("Mapped legacy checkpoint key '%s' -> '%s'.", legacy_key, current_key)
        return remapped

    def _build_model(self) -> tuple[nn.Module, nn.Module]:
        assert self.checkpoint_payload is not None
        aux_dim = len(self.aux_columns) if self.use_aux else 0
        common_kwargs = dict(
            model_name=self.checkpoint_payload.get("model_name", self.tokenizer_name),
            dropout_rate=float(self.checkpoint_payload.get("dropout_rate", 0.5)),
            use_mean_pooling=bool(self.checkpoint_payload.get("use_mean_pooling", True)),
            aux_feat_dim=aux_dim,
        )
        if self.spec.head_task == "binary":
            model = BERTClassifier(**common_kwargs)
            criterion: nn.Module = nn.CrossEntropyLoss()
        else:
            model = BERTRegressor(**common_kwargs)
            loss_name = self.checkpoint_payload.get("loss_name")
            if loss_name == "poisson_nll" or self.mode == "count_stage2_poisson":
                criterion = nn.PoissonNLLLoss(log_input=True, full=False)
            elif loss_name == "smooth_l1":
                criterion = nn.SmoothL1Loss()
            else:
                criterion = nn.MSELoss()
        return model.to(self.device).eval(), criterion

    def load(self) -> "BertInferenceSession":
        self.checkpoint_path = self.checkpoint_path_override or checkpoint_path(self.out_dir, self.mode, "best")
        self.checkpoint_payload = self._load_checkpoint(self.checkpoint_path)
        payload_mode = self.checkpoint_payload.get("mode")
        if payload_mode is not None and payload_mode != self.mode:
            raise ValueError(
                f"Checkpoint mode mismatch: evaluator requested '{self.mode}' but checkpoint stores '{payload_mode}'."
            )

        self.transform_meta = self.checkpoint_payload.get("target_transform")
        if self.spec.head_task == "regression":
            if not isinstance(self.transform_meta, dict):
                raise ValueError(f"Checkpoint {self.checkpoint_path} is missing target_transform metadata.")
            if self.mode == "count_stage2_log1p":
                if self.transform_meta.get("y_mean") is None or self.transform_meta.get("y_std") is None:
                    raise ValueError(
                        f"Checkpoint {self.checkpoint_path} is missing y_mean/y_std for log1p target inversion."
                    )
        else:
            self.transform_meta = {"name": "identity", "source_column": self.spec.label_column}

        self.tokenizer_name = self.tokenizer_name_override or self.checkpoint_payload.get(
            "tokenizer_name",
            "distilbert-base-uncased",
        )
        self.max_len = int(self.max_len_override or self.checkpoint_payload.get("max_len", 128))
        if self.use_aux_override is None:
            self.use_aux = bool(self.checkpoint_payload.get("use_aux_features", True))
        else:
            self.use_aux = bool(self.use_aux_override)
        if self.aux_columns_override is not None:
            self.aux_columns = tuple(self.aux_columns_override)
        else:
            self.aux_columns = tuple(self.checkpoint_payload.get("aux_columns", self.aux_columns))
        self.aux_means = dict(self.checkpoint_payload.get("aux_means", {}))
        self.aux_stds = dict(self.checkpoint_payload.get("aux_stds", {}))

        self.model, self.criterion = self._build_model()
        state_dict = self._normalize_state_dict_keys(self.checkpoint_payload["model_state_dict"])
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            self.logger.warning("State_dict mismatch. Missing=%s | Unexpected=%s", missing, unexpected)
        return self

    def _ensure_label_columns(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        if self.spec.head_task == "binary":
            if self.spec.label_column not in out.columns:
                raise ValueError(f"Missing label column '{self.spec.label_column}' for mode '{self.mode}'.")
            return out

        raw_target_column = self.spec.raw_target_column
        if raw_target_column not in out.columns:
            if self.mode.startswith("count_stage2") and "helpful_vote" in out.columns:
                out[raw_target_column] = pd.to_numeric(out["helpful_vote"], errors="raise")
            else:
                raise ValueError(f"Missing raw target column '{raw_target_column}' for mode '{self.mode}'.")
        if self.mode == "count_stage2_log1p" and self.spec.label_column not in out.columns:
            out[self.spec.label_column] = np.log1p(pd.to_numeric(out[raw_target_column], errors="raise"))
        if self.mode == "count_stage2_poisson" and self.spec.label_column not in out.columns:
            out[self.spec.label_column] = pd.to_numeric(out[raw_target_column], errors="raise")
        return out

    def predict_frame(self, frame: pd.DataFrame) -> dict[str, Any]:
        if self.model is None or self.criterion is None or self.transform_meta is None:
            self.load()
        assert self.model is not None
        assert self.criterion is not None
        assert self.transform_meta is not None

        prepared = self._ensure_label_columns(frame)
        transformed = apply_target_transform(prepared, self.spec, self.transform_meta)
        ds = ReviewDataset(
            transformed,
            task=self.spec.head_task,
            label_col=self.spec.label_column,
            use_aux_features=self.use_aux,
            aux_cols=self.aux_columns,
        )
        if self.use_aux:
            ds.set_aux_stats(self.aux_means, self.aux_stds)

        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, use_fast=True)
        collate = make_collate(
            tokenizer,
            max_len=self.max_len,
            task=self.spec.head_task,
            pad_to_max_length=False,
            return_aux=self.use_aux,
        )
        loader = device_aware_loader(
            ds,
            batch_size=self.batch_size,
            device=self.device,
            shuffle=False,
            collate_fn=collate,
        )

        losses: list[float] = []
        logits_all: list[np.ndarray] = []
        targets_all: list[np.ndarray] = []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
                targets = batch["target"].to(self.device, non_blocking=True)
                aux = batch.get("aux_features")
                if aux is not None:
                    aux = aux.to(self.device, non_blocking=True)

                logits = self.model(input_ids=input_ids, attention_mask=attention_mask, aux_features=aux)
                loss = self.criterion(logits.float(), targets.long() if self.spec.head_task == "binary" else targets.float())
                losses.append(float(loss.item()))
                logits_all.append(logits.detach().cpu().float().numpy())
                targets_all.append(targets.detach().cpu().float().numpy())

        logits_np = np.concatenate(logits_all, axis=0)
        targets_np = np.concatenate(targets_all, axis=0)
        result: dict[str, Any] = {
            "loss": float(np.mean(losses)) if losses else float("nan"),
            "logits": logits_np,
            "targets": targets_np,
            "frame": prepared,
        }
        if self.spec.head_task == "binary":
            probs = torch.softmax(torch.tensor(logits_np), dim=1).numpy()[:, 1]
            result["probabilities"] = probs
        else:
            raw_pred = invert_predictions(logits_np.reshape(-1), self.transform_meta)
            result["predicted_count"] = raw_pred
        return result


class Evaluator:
    def __init__(self, cfg: EvalConfig, logger: Optional[logging.Logger] = None):
        self.cfg = cfg
        self.logger = logger or logging.getLogger("evaluate_bert")
        self.session = BertInferenceSession(
            mode=cfg.mode,
            out_dir=cfg.out_dir,
            checkpoint_path_override=cfg.checkpoint_path,
            tokenizer_name_override=cfg.tokenizer_name,
            max_len_override=cfg.max_len,
            use_aux_override=cfg.use_aux,
            aux_columns_override=cfg.aux_columns,
            batch_size=cfg.batch_size,
            logger=self.logger,
        )

    def run(self) -> dict[str, Any]:
        frame = load_mode_dataframe(self.cfg.base_dir, self.cfg.mode, self.cfg.split)
        outputs = self.session.predict_frame(frame)
        eval_frame = outputs["frame"]

        if self.session.spec.head_task == "binary":
            labels = pd.to_numeric(eval_frame[self.session.spec.raw_target_column], errors="raise").to_numpy(dtype=int)
            probs = outputs["probabilities"]
            metrics = compute_binary_metrics(
                labels,
                probs,
                target_precision=self.cfg.target_precision,
                min_recall=self.cfg.min_recall,
            )
            pred_df = pd.DataFrame(
                {
                    "review_id": eval_frame["review_id"].to_numpy(),
                    "true_label": labels,
                    "prob_pos": probs,
                    "pred_label_05": (probs >= 0.5).astype(int),
                    "pred_label_opt": (probs >= metrics["thr_opt"]).astype(int),
                }
            )
        else:
            true_count = pd.to_numeric(eval_frame[self.session.spec.raw_target_column], errors="raise").to_numpy(dtype=float)
            pred_count = outputs["predicted_count"]
            metrics = compute_count_metrics(true_count, pred_count, positive_mask=(true_count > 0))
            pred_df = pd.DataFrame(
                {
                    "review_id": eval_frame["review_id"].to_numpy(),
                    "true_count": true_count,
                    "pred_count": pred_count,
                    "model_output": outputs["logits"].reshape(-1),
                }
            )

        metrics["eval_loss"] = float(outputs["loss"])
        pred_path = predictions_path(self.cfg.out_dir, self.cfg.mode, self.cfg.split)
        metrics_out_path = metrics_path(self.cfg.out_dir, self.cfg.mode, self.cfg.split)
        pred_df.to_csv(pred_path, index=False)

        payload = {
            "config": asdict(self.cfg),
            "checkpoint_path": self.session.checkpoint_path,
            "target_transform": self.session.transform_meta,
            "metrics": metrics,
            "prediction_path": pred_path,
        }
        with open(metrics_out_path, "w", encoding="utf-8") as fh:
            json.dump(_json_ready(payload), fh, indent=2)

        self.logger.info("Saved predictions -> %s", pred_path)
        self.logger.info("Saved metrics -> %s", metrics_out_path)
        for key, value in metrics.items():
            if isinstance(value, float):
                self.logger.info("%s=%.6f", key, value)
            else:
                self.logger.info("%s=%s", key, value)
        return payload
