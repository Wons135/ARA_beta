from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


BertModeT = Literal["binary", "count_stage1", "count_stage2_log1p", "count_stage2_poisson"]
HeadTaskT = Literal["binary", "regression"]


@dataclass(frozen=True)
class BertModeSpec:
    mode: BertModeT
    head_task: HeadTaskT
    dataset_prefix: str
    label_column: str
    raw_target_column: str
    checkpoint_stem: str


MODE_SPECS: dict[BertModeT, BertModeSpec] = {
    "binary": BertModeSpec(
        mode="binary",
        head_task="binary",
        dataset_prefix="binary_label",
        label_column="binary_label",
        raw_target_column="binary_label",
        checkpoint_stem="bert_binary",
    ),
    "count_stage1": BertModeSpec(
        mode="count_stage1",
        head_task="binary",
        dataset_prefix="count_stage1",
        label_column="has_helpful_vote",
        raw_target_column="has_helpful_vote",
        checkpoint_stem="bert_count_stage1",
    ),
    "count_stage2_log1p": BertModeSpec(
        mode="count_stage2_log1p",
        head_task="regression",
        dataset_prefix="count_stage2",
        label_column="log1p_helpful_vote",
        raw_target_column="helpful_vote_pos",
        checkpoint_stem="bert_count_stage2_log1p",
    ),
    "count_stage2_poisson": BertModeSpec(
        mode="count_stage2_poisson",
        head_task="regression",
        dataset_prefix="count_stage2",
        label_column="helpful_vote_pos",
        raw_target_column="helpful_vote_pos",
        checkpoint_stem="bert_count_stage2_poisson",
    ),
}


def get_mode_spec(mode: BertModeT) -> BertModeSpec:
    try:
        return MODE_SPECS[mode]
    except KeyError as exc:
        raise ValueError(f"Unsupported BERT mode: {mode}") from exc


def setup_logger(name: str, log_path: Optional[str | Path] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in logger.handlers):
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(stream_handler)
    if log_path is not None:
        resolved = Path(log_path).expanduser().resolve()
        if not any(isinstance(h, logging.FileHandler) and Path(h.baseFilename).resolve() == resolved for h in logger.handlers):
            resolved.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(resolved, mode="a", encoding="utf-8")
            file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
            logger.addHandler(file_handler)
    return logger


def safe_path(base_dir: str | Path, *parts: str) -> str:
    base = Path(base_dir).expanduser().resolve()
    out = base.joinpath(*parts).resolve()
    try:
        out.relative_to(base)
    except ValueError as exc:
        raise ValueError(f"Unsafe path outside base dir: {out}") from exc
    out.parent.mkdir(parents=True, exist_ok=True)
    return str(out)


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def checkpoint_path(out_dir: str | Path, mode: BertModeT, kind: Literal["best", "last"] = "best") -> str:
    spec = get_mode_spec(mode)
    return safe_path(out_dir, "checkpoints", f"{spec.checkpoint_stem}_{kind}.pth")


def metrics_path(out_dir: str | Path, mode: BertModeT, split: str) -> str:
    spec = get_mode_spec(mode)
    return safe_path(out_dir, "predictions", f"{spec.checkpoint_stem}_{split}_metrics.json")


def predictions_path(out_dir: str | Path, mode: BertModeT, split: str) -> str:
    spec = get_mode_spec(mode)
    return safe_path(out_dir, "predictions", f"{spec.checkpoint_stem}_{split}_predictions.csv")


def load_mode_dataframe(base_dir: str | Path, mode: BertModeT, split: str) -> pd.DataFrame:
    spec = get_mode_spec(mode)
    path = Path(base_dir) / f"{spec.dataset_prefix}_{split}.csv"
    return pd.read_csv(path)


def fit_target_transform(df: pd.DataFrame, spec: BertModeSpec) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = df.copy()
    transform_meta: dict[str, Any] = {
        "name": "identity",
        "source_column": spec.label_column,
        "invert_prediction": "identity",
        "y_mean": None,
        "y_std": None,
    }

    if spec.mode == "count_stage2_log1p":
        y = pd.to_numeric(out[spec.label_column], errors="raise").astype(float)
        y_mean = float(y.mean())
        y_std = float(y.std())
        if y_std < 1e-12:
            y_std = 1.0
        out[spec.label_column] = ((y - y_mean) / (y_std + 1e-12)).astype(np.float32)
        transform_meta = {
            "name": "standardized_log1p_helpful_vote",
            "source_column": spec.label_column,
            "invert_prediction": "expm1((pred * y_std) + y_mean)",
            "y_mean": y_mean,
            "y_std": y_std,
        }
    elif spec.mode == "count_stage2_poisson":
        transform_meta = {
            "name": "poisson_raw_positive_count",
            "source_column": spec.label_column,
            "invert_prediction": "exp(pred)",
            "y_mean": None,
            "y_std": None,
        }

    metadata = {
        "mode": spec.mode,
        "head_task": spec.head_task,
        "label_column": spec.label_column,
        "raw_target_column": spec.raw_target_column,
        "dataset_prefix": spec.dataset_prefix,
        "target_transform": transform_meta,
    }
    return out, metadata


def apply_target_transform(df: pd.DataFrame, spec: BertModeSpec, transform_meta: dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    transform_name = transform_meta.get("name")
    if transform_name == "standardized_log1p_helpful_vote":
        y_mean = transform_meta.get("y_mean")
        y_std = transform_meta.get("y_std")
        if y_mean is None or y_std is None:
            raise ValueError("Missing y_mean/y_std for standardized log1p target transform.")
        y = pd.to_numeric(out[spec.label_column], errors="raise").astype(float)
        out[spec.label_column] = ((y - float(y_mean)) / (float(y_std) + 1e-12)).astype(np.float32)
        return out
    if transform_name in {"identity", "poisson_raw_positive_count"}:
        return out
    raise ValueError(f"Unsupported target transform metadata: {transform_name}")


def invert_predictions(preds: np.ndarray, transform_meta: dict[str, Any]) -> np.ndarray:
    preds = np.asarray(preds, dtype=float)
    transform_name = transform_meta.get("name")
    if transform_name == "standardized_log1p_helpful_vote":
        y_mean = transform_meta.get("y_mean")
        y_std = transform_meta.get("y_std")
        if y_mean is None or y_std is None:
            raise ValueError("Missing y_mean/y_std for standardized log1p prediction inversion.")
        return np.clip(np.expm1(preds * float(y_std) + float(y_mean)), a_min=0.0, a_max=None)
    if transform_name == "poisson_raw_positive_count":
        return np.clip(np.exp(preds), a_min=0.0, a_max=None)
    if transform_name == "identity":
        return preds
    raise ValueError(f"Unsupported target transform metadata: {transform_name}")


def pick_precision_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    target_precision: float = 0.30,
    min_recall: float = 0.15,
) -> tuple[float, float, float, float]:
    precs, recs, thrs = precision_recall_curve(labels, probs)
    precs = precs[:-1]
    recs = recs[:-1]
    if thrs.size == 0:
        preds = (probs >= 0.5).astype(int)
        return (
            0.5,
            float(precision_score(labels, preds, zero_division=0)),
            float(recall_score(labels, preds, zero_division=0)),
            float(f1_score(labels, preds, zero_division=0)),
        )

    mask = (precs >= target_precision) & (recs >= min_recall)
    if np.any(mask):
        idxs = np.flatnonzero(mask)
        f1s = 2 * precs[idxs] * recs[idxs] / np.maximum(precs[idxs] + recs[idxs], 1e-12)
        local_idx = int(np.argmax(f1s))
        idx = int(idxs[local_idx])
        return float(thrs[idx]), float(precs[idx]), float(recs[idx]), float(f1s[local_idx])

    f1s = 2 * precs * recs / np.maximum(precs + recs, 1e-12)
    idx = int(np.argmax(f1s))
    return float(thrs[idx]), float(precs[idx]), float(recs[idx]), float(f1s[idx])


def compute_binary_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
    target_precision: float,
    min_recall: float,
) -> dict[str, float]:
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs).astype(float)
    preds_05 = (probs >= 0.5).astype(int)
    thr_opt, prec_opt, rec_opt, f1_opt = pick_precision_threshold(
        probs,
        labels,
        target_precision=target_precision,
        min_recall=min_recall,
    )
    return {
        "auroc": float(roc_auc_score(labels, probs)) if np.unique(labels).size > 1 else float("nan"),
        "auprc": float(average_precision_score(labels, probs)) if np.unique(labels).size > 1 else float("nan"),
        "acc_05": float(accuracy_score(labels, preds_05)),
        "prec_05": float(precision_score(labels, preds_05, zero_division=0)),
        "rec_05": float(recall_score(labels, preds_05, zero_division=0)),
        "f1_05": float(f1_score(labels, preds_05, zero_division=0)),
        "thr_opt": float(thr_opt),
        "prec_opt": float(prec_opt),
        "rec_opt": float(rec_opt),
        "f1_opt": float(f1_opt),
    }


def binary_probabilities_from_logits(logits: np.ndarray) -> np.ndarray:
    arr = np.asarray(logits, dtype=np.float32)
    if arr.ndim == 1:
        clipped = np.clip(arr, -50.0, 50.0)
        return 1.0 / (1.0 + np.exp(-clipped))
    if arr.ndim == 2 and arr.shape[1] == 1:
        clipped = np.clip(arr[:, 0], -50.0, 50.0)
        return 1.0 / (1.0 + np.exp(-clipped))
    if arr.ndim == 2 and arr.shape[1] == 2:
        shifted = arr - np.max(arr, axis=1, keepdims=True)
        exp = np.exp(np.clip(shifted, -50.0, 50.0))
        denom = np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)
        return (exp / denom)[:, 1]
    raise ValueError(f"Unsupported binary logits shape: {arr.shape}")


def binary_confusion_payload(labels: np.ndarray, probs: np.ndarray, threshold: float) -> dict[str, float]:
    preds = (np.asarray(probs) >= float(threshold)).astype(int)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "predicted_positive_rate": float(preds.mean()),
    }


def _plot_confusion_matrix(cm: np.ndarray, path: str | Path, title: str) -> None:
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(2)
    plt.xticks(ticks, [0, 1])
    plt.yticks(ticks, [0, 1])
    thresh = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = int(cm[i, j])
            plt.text(j, i, str(value), ha="center", va="center", color="white" if value > thresh else "black")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_binary_diagnostics(
    out_dir: str | Path,
    stem: str,
    labels: np.ndarray,
    probs: np.ndarray,
    *,
    threshold_opt: float,
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=int)
    probs = np.asarray(probs, dtype=float)
    pos_probs = probs[labels == 1]
    neg_probs = probs[labels == 0]

    hist_path = safe_path(out_dir, "plots", f"{stem}_score_hist.png")
    pr_path = safe_path(out_dir, "plots", f"{stem}_pr.png")
    cm05_path = safe_path(out_dir, "plots", f"{stem}_cm_05.png")
    cmopt_path = safe_path(out_dir, "plots", f"{stem}_cm_opt.png")

    bins = np.linspace(0.0, 1.0, 51)
    plt.figure(figsize=(7, 5))
    plt.hist(neg_probs, bins=bins, alpha=0.65, label=f"negative (n={neg_probs.size:,})", color="#355C7D")
    plt.hist(pos_probs, bins=bins, alpha=0.65, label=f"positive (n={pos_probs.size:,})", color="#C06C84")
    plt.axvline(0.5, color="black", linestyle="--", linewidth=1, label="threshold 0.5")
    plt.axvline(float(threshold_opt), color="#F67280", linestyle="--", linewidth=1, label=f"threshold opt={threshold_opt:.4f}")
    plt.xlabel("Predicted probability")
    plt.ylabel("Count")
    plt.title("Positive vs Negative Score Histogram")
    plt.legend()
    plt.tight_layout()
    plt.savefig(hist_path)
    plt.close()

    precisions, recalls, thresholds = precision_recall_curve(labels, probs)
    plt.figure(figsize=(6, 5))
    plt.plot(recalls, precisions, lw=2, color="#355C7D")
    preds_05 = (probs >= 0.5).astype(int)
    prec_05 = precision_score(labels, preds_05, zero_division=0)
    rec_05 = recall_score(labels, preds_05, zero_division=0)
    preds_opt = (probs >= threshold_opt).astype(int)
    prec_opt = precision_score(labels, preds_opt, zero_division=0)
    rec_opt = recall_score(labels, preds_opt, zero_division=0)
    plt.scatter([rec_05], [prec_05], color="black", label="@0.5")
    plt.scatter([rec_opt], [prec_opt], color="#F67280", label="@thr_opt")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(pr_path)
    plt.close()

    cm05 = confusion_matrix(labels, preds_05, labels=[0, 1])
    cmopt = confusion_matrix(labels, preds_opt, labels=[0, 1])
    _plot_confusion_matrix(cm05, cm05_path, "Confusion Matrix @ 0.5")
    _plot_confusion_matrix(cmopt, cmopt_path, f"Confusion Matrix @ {threshold_opt:.4f}")

    quantiles = [0.0, 0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 1.0]

    def _distribution_summary(values: np.ndarray) -> dict[str, float]:
        if values.size == 0:
            return {"count": 0}
        qs = np.quantile(values, quantiles)
        summary = {"count": int(values.size), "mean": float(values.mean())}
        for q, value in zip(quantiles, qs):
            summary[f"q{int(round(q * 100)):02d}"] = float(value)
        return summary

    return {
        "plots": {
            "score_histogram": hist_path,
            "pr_curve": pr_path,
            "cm_05": cm05_path,
            "cm_opt": cmopt_path,
        },
        "summary": {
            "pct_prob_ge_05": float((probs >= 0.5).mean()),
            "pct_prob_ge_thr_opt": float((probs >= threshold_opt).mean()),
            "threshold_05": binary_confusion_payload(labels, probs, 0.5),
            "threshold_opt": binary_confusion_payload(labels, probs, threshold_opt),
            "positive_scores": _distribution_summary(pos_probs),
            "negative_scores": _distribution_summary(neg_probs),
            "num_pr_thresholds": int(thresholds.size),
        },
    }


def compute_count_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    positive_mask: Optional[np.ndarray] = None,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    positive_mask = (y_true > 0) if positive_mask is None else np.asarray(positive_mask, dtype=bool)
    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)) if y_true.size > 1 else float("nan"),
    }
    if y_true.size > 1 and np.unique(y_true).size > 1 and np.unique(y_pred).size > 1:
        spearman = spearmanr(y_true, y_pred).statistic
        metrics["spearman_rho"] = float(spearman) if spearman is not None else float("nan")
    else:
        metrics["spearman_rho"] = float("nan")
    if positive_mask.any():
        metrics["positive_only_rmse"] = float(np.sqrt(mean_squared_error(y_true[positive_mask], y_pred[positive_mask])))
        metrics["positive_only_mae"] = float(mean_absolute_error(y_true[positive_mask], y_pred[positive_mask]))
    else:
        metrics["positive_only_rmse"] = float("nan")
        metrics["positive_only_mae"] = float("nan")
    return metrics
