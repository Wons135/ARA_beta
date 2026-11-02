from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class BinaryArtifactSpec:
    name: str
    metrics_path: str
    predictions_path: str
    color: str


@dataclass(frozen=True)
class CountArtifactSpec:
    name: str
    metrics_path: str
    color: str


@dataclass(frozen=True)
class BinaryArtifact:
    name: str
    labels: np.ndarray
    probs: np.ndarray
    metrics: dict[str, float]
    threshold_opt: float
    color: str


@dataclass(frozen=True)
class CountArtifact:
    name: str
    y_true: np.ndarray
    y_pred: np.ndarray
    metrics: dict[str, float]
    color: str


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _pick_precision_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    *,
    target_precision: float,
    min_recall: float,
) -> tuple[float, float, float, float]:
    precisions, recalls, thresholds = precision_recall_curve(labels, probs)
    precisions = precisions[:-1]
    recalls = recalls[:-1]

    if thresholds.size == 0:
        preds = (probs >= 0.5).astype(int)
        return (
            0.5,
            float(precision_score(labels, preds, zero_division=0)),
            float(recall_score(labels, preds, zero_division=0)),
            float(f1_score(labels, preds, zero_division=0)),
        )

    mask = (precisions >= target_precision) & (recalls >= min_recall)
    if np.any(mask):
        candidate_idx = np.flatnonzero(mask)
        f1s = 2 * precisions[candidate_idx] * recalls[candidate_idx] / np.maximum(
            precisions[candidate_idx] + recalls[candidate_idx],
            1e-12,
        )
        best_idx = int(candidate_idx[np.argmax(f1s)])
        return (
            float(thresholds[best_idx]),
            float(precisions[best_idx]),
            float(recalls[best_idx]),
            float(np.max(f1s)),
        )

    f1s = 2 * precisions * recalls / np.maximum(precisions + recalls, 1e-12)
    best_idx = int(np.argmax(f1s))
    return (
        float(thresholds[best_idx]),
        float(precisions[best_idx]),
        float(recalls[best_idx]),
        float(f1s[best_idx]),
    )


def _binary_specs(split: str, out_dir: Path) -> list[BinaryArtifactSpec]:
    pred_dir = out_dir / "predictions"
    return [
        BinaryArtifactSpec(
            name="Baseline",
            metrics_path=str(pred_dir / f"baseline_binary_{split}_metrics.json"),
            predictions_path=str(pred_dir / f"baseline_binary_{split}_predictions.csv"),
            color="#6C757D",
        ),
        BinaryArtifactSpec(
            name="BERT",
            metrics_path=str(pred_dir / f"bert_binary_{split}_metrics.json"),
            predictions_path=str(pred_dir / f"bert_binary_{split}_predictions.csv"),
            color="#355C7D",
        ),
        BinaryArtifactSpec(
            name="XGBoost",
            metrics_path=str(pred_dir / f"xgb_binary_{split}_metrics.json"),
            predictions_path=str(pred_dir / f"xgb_binary_{split}_predictions.csv"),
            color="#C06C84",
        ),
    ]


def _count_specs(split: str, out_dir: Path) -> list[CountArtifactSpec]:
    pred_dir = out_dir / "predictions"
    return [
        CountArtifactSpec(
            name="BERT log1p",
            metrics_path=str(pred_dir / f"bert_count_combined_count_stage2_log1p_{split}_metrics.json"),
            color="#355C7D",
        ),
        CountArtifactSpec(
            name="BERT Poisson",
            metrics_path=str(pred_dir / f"bert_count_combined_count_stage2_poisson_{split}_metrics.json"),
            color="#6C5B7B",
        ),
        CountArtifactSpec(
            name="XGBoost log1p",
            metrics_path=str(pred_dir / f"xgb_count_combined_count_stage2_log1p_{split}_metrics.json"),
            color="#C06C84",
        ),
        CountArtifactSpec(
            name="XGBoost Poisson",
            metrics_path=str(pred_dir / f"xgb_count_combined_count_stage2_poisson_{split}_metrics.json"),
            color="#F67280",
        ),
    ]


def _load_binary_artifacts(
    split: str,
    out_dir: Path,
    *,
    target_precision: float,
    min_recall: float,
    logger: logging.Logger,
) -> list[BinaryArtifact]:
    artifacts: list[BinaryArtifact] = []
    for spec in _binary_specs(split, out_dir):
        metrics_path = Path(spec.metrics_path)
        predictions_path = Path(spec.predictions_path)
        if not metrics_path.exists() or not predictions_path.exists():
            logger.info("Skipping binary figure source '%s' because artifacts are missing.", spec.name)
            continue

        payload = _load_json(metrics_path)
        pred_df = pd.read_csv(predictions_path)
        if "true_label" not in pred_df.columns or "prob_pos" not in pred_df.columns:
            logger.warning("Skipping binary figure source '%s' due to unexpected prediction columns.", spec.name)
            continue

        labels = pd.to_numeric(pred_df["true_label"], errors="raise").to_numpy(dtype=int)
        probs = pd.to_numeric(pred_df["prob_pos"], errors="raise").to_numpy(dtype=float)
        payload_metrics = payload.get("metrics", payload)

        if all(key in payload_metrics for key in ("auroc", "auprc", "f1_opt", "thr_opt")):
            metrics = {
                "auroc": float(payload_metrics["auroc"]),
                "auprc": float(payload_metrics["auprc"]),
                "f1_opt": float(payload_metrics["f1_opt"]),
            }
            threshold_opt = float(payload_metrics["thr_opt"])
        else:
            threshold_opt, _, _, f1_opt = _pick_precision_threshold(
                probs,
                labels,
                target_precision=target_precision,
                min_recall=min_recall,
            )
            metrics = {
                "auroc": float(roc_auc_score(labels, probs)) if np.unique(labels).size > 1 else float("nan"),
                "auprc": float(average_precision_score(labels, probs)) if np.unique(labels).size > 1 else float("nan"),
                "f1_opt": float(f1_opt),
            }

        artifacts.append(
            BinaryArtifact(
                name=spec.name,
                labels=labels,
                probs=probs,
                metrics=metrics,
                threshold_opt=threshold_opt,
                color=spec.color,
            )
        )
    return artifacts


def _load_count_artifacts(split: str, out_dir: Path, logger: logging.Logger) -> list[CountArtifact]:
    artifacts: list[CountArtifact] = []
    for spec in _count_specs(split, out_dir):
        metrics_path = Path(spec.metrics_path)
        if not metrics_path.exists():
            logger.info("Skipping count figure source '%s' because metrics are missing.", spec.name)
            continue

        payload = _load_json(metrics_path)
        payload_metrics = payload.get("expected_count_metrics")
        prediction_path = payload.get("prediction_path")
        if not isinstance(payload_metrics, dict) or not prediction_path:
            logger.warning("Skipping count figure source '%s' due to incomplete combined payload.", spec.name)
            continue

        pred_df = pd.read_csv(prediction_path)
        if "true_count" not in pred_df.columns:
            logger.warning("Skipping count figure source '%s' due to missing true_count column.", spec.name)
            continue

        pred_col = "expected_count" if "expected_count" in pred_df.columns else "y_pred"
        if pred_col not in pred_df.columns:
            logger.warning("Skipping count figure source '%s' due to missing predicted-count column.", spec.name)
            continue

        artifacts.append(
            CountArtifact(
                name=spec.name,
                y_true=pd.to_numeric(pred_df["true_count"], errors="raise").to_numpy(dtype=float),
                y_pred=pd.to_numeric(pred_df[pred_col], errors="raise").to_numpy(dtype=float),
                metrics={
                    "rmse": float(payload_metrics["rmse"]),
                    "mae": float(payload_metrics["mae"]),
                    "spearman_rho": float(payload_metrics["spearman_rho"]),
                },
                color=spec.color,
            )
        )
    return artifacts


def _annotate_bar_values(ax: plt.Axes, values: np.ndarray) -> None:
    ymin, ymax = ax.get_ylim()
    offset = max((ymax - ymin) * 0.02, 0.01)
    for idx, value in enumerate(values):
        if not np.isfinite(value):
            continue
        ax.text(idx, value + offset, f"{value:.3f}", ha="center", va="bottom", fontsize=8)


def _plot_binary_metric_comparison(artifacts: list[BinaryArtifact], plot_dir: Path, split: str) -> str:
    metric_names = ("auroc", "auprc", "f1_opt")
    metric_labels = ("AUROC", "AUPRC", "F1*")
    x = np.arange(len(metric_names))
    width = 0.8 / max(len(artifacts), 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    for idx, artifact in enumerate(artifacts):
        values = np.array([artifact.metrics[name] for name in metric_names], dtype=float)
        positions = x + (idx - (len(artifacts) - 1) / 2.0) * width
        ax.bar(positions, values, width=width, label=artifact.name, color=artifact.color)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title(f"Binary Metrics Comparison ({split})")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()

    out_path = plot_dir / f"binary_metric_comparison_{split}.png"
    fig.savefig(out_path)
    plt.close(fig)
    return str(out_path)


def _plot_binary_pr_curve(artifacts: list[BinaryArtifact], plot_dir: Path, split: str) -> str:
    fig, ax = plt.subplots(figsize=(7, 6))
    for artifact in artifacts:
        precisions, recalls, _ = precision_recall_curve(artifact.labels, artifact.probs)
        ax.plot(
            recalls,
            precisions,
            lw=2,
            color=artifact.color,
            label=f"{artifact.name} (AUPRC={artifact.metrics['auprc']:.3f})",
        )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Binary Precision-Recall Curve ({split})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    out_path = plot_dir / f"binary_pr_curve_{split}.png"
    fig.savefig(out_path)
    plt.close(fig)
    return str(out_path)


def _draw_confusion_matrix(ax: plt.Axes, cm: np.ndarray, title: str) -> None:
    ax.imshow(cm, interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    threshold = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = int(cm[i, j])
            ax.text(j, i, str(value), ha="center", va="center", color="white" if value > threshold else "black")


def _plot_binary_confusion_matrices(artifacts: list[BinaryArtifact], plot_dir: Path, split: str) -> str:
    fig, axes = plt.subplots(1, len(artifacts), figsize=(5 * len(artifacts), 4.5), squeeze=False)
    for ax, artifact in zip(axes[0], artifacts):
        preds = (artifact.probs >= artifact.threshold_opt).astype(int)
        cm = confusion_matrix(artifact.labels, preds, labels=[0, 1])
        _draw_confusion_matrix(ax, cm, f"{artifact.name}\n@thr={artifact.threshold_opt:.3f}")

    fig.suptitle(f"Binary Confusion Matrices at Optimal Threshold ({split})")
    fig.tight_layout()
    out_path = plot_dir / f"binary_confusion_matrices_opt_{split}.png"
    fig.savefig(out_path)
    plt.close(fig)
    return str(out_path)


def _plot_count_metric_comparison(artifacts: list[CountArtifact], plot_dir: Path, split: str) -> str:
    metric_names = ("rmse", "mae", "spearman_rho")
    metric_labels = ("RMSE", "MAE", "Spearman")
    fig, axes = plt.subplots(1, len(metric_names), figsize=(13, 4.5))

    for ax, metric_name, metric_label in zip(axes, metric_names, metric_labels):
        values = np.array([artifact.metrics[metric_name] for artifact in artifacts], dtype=float)
        colors = [artifact.color for artifact in artifacts]
        ax.bar(range(len(artifacts)), values, color=colors)
        ax.set_xticks(range(len(artifacts)))
        ax.set_xticklabels([artifact.name for artifact in artifacts], rotation=20, ha="right")
        ax.set_title(metric_label)
        ax.grid(axis="y", alpha=0.25)
        _annotate_bar_values(ax, values)

    fig.suptitle(f"Combined Count-Task Metrics Comparison ({split})")
    fig.tight_layout()
    out_path = plot_dir / f"count_metric_comparison_{split}.png"
    fig.savefig(out_path)
    plt.close(fig)
    return str(out_path)


def _plot_count_actual_vs_predicted(artifacts: list[CountArtifact], plot_dir: Path, split: str) -> str:
    n = len(artifacts)
    ncols = 2 if n > 1 else 1
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 5.2 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for ax, artifact in zip(axes_flat, artifacts):
        true_log = np.log1p(np.clip(artifact.y_true, a_min=0.0, a_max=None))
        pred_log = np.log1p(np.clip(artifact.y_pred, a_min=0.0, a_max=None))
        ax.scatter(true_log, pred_log, alpha=0.25, s=8, color=artifact.color)
        lo = float(min(np.min(true_log), np.min(pred_log))) if true_log.size and pred_log.size else 0.0
        hi = float(max(np.max(true_log), np.max(pred_log))) if true_log.size and pred_log.size else 1.0
        ax.plot([lo, hi], [lo, hi], linestyle="--", color="black", linewidth=1)
        ax.set_title(artifact.name)
        ax.set_xlabel("log1p(True count)")
        ax.set_ylabel("log1p(Predicted count)")
        ax.grid(True, alpha=0.25)

    for ax in axes_flat[n:]:
        ax.axis("off")

    fig.suptitle(f"Actual vs Predicted Log-Count ({split})")
    fig.tight_layout()
    out_path = plot_dir / f"count_actual_vs_predicted_log_{split}.png"
    fig.savefig(out_path)
    plt.close(fig)
    return str(out_path)


def generate_binary_main_figures(
    out_dir: str | Path,
    *,
    split: str,
    target_precision: float,
    min_recall: float,
    logger: logging.Logger,
) -> dict[str, str]:
    out_root = Path(out_dir).expanduser().resolve()
    plot_dir = out_root / "plots" / "main_figures"
    plot_dir.mkdir(parents=True, exist_ok=True)

    artifacts = _load_binary_artifacts(
        split,
        out_root,
        target_precision=target_precision,
        min_recall=min_recall,
        logger=logger,
    )
    if not artifacts:
        return {}

    return {
        "binary_metric_comparison": _plot_binary_metric_comparison(artifacts, plot_dir, split),
        "binary_pr_curve": _plot_binary_pr_curve(artifacts, plot_dir, split),
        "binary_confusion_matrices_opt": _plot_binary_confusion_matrices(artifacts, plot_dir, split),
    }


def generate_count_main_figures(
    out_dir: str | Path,
    *,
    split: str,
    logger: logging.Logger,
) -> dict[str, str]:
    out_root = Path(out_dir).expanduser().resolve()
    plot_dir = out_root / "plots" / "main_figures"
    plot_dir.mkdir(parents=True, exist_ok=True)

    artifacts = _load_count_artifacts(split, out_root, logger)
    if not artifacts:
        return {}

    return {
        "count_metric_comparison": _plot_count_metric_comparison(artifacts, plot_dir, split),
        "count_actual_vs_predicted_log": _plot_count_actual_vs_predicted(artifacts, plot_dir, split),
    }


def try_generate_requested_main_figures(
    out_dir: str | Path,
    *,
    split: str,
    logger: logging.Logger,
    include_binary: bool = True,
    include_count: bool = True,
    target_precision: float = 0.30,
    min_recall: float = 0.15,
) -> dict[str, str]:
    generated: dict[str, str] = {}
    try:
        if include_binary:
            generated.update(
                generate_binary_main_figures(
                    out_dir,
                    split=split,
                    target_precision=target_precision,
                    min_recall=min_recall,
                    logger=logger,
                )
            )
        if include_count:
            generated.update(generate_count_main_figures(out_dir, split=split, logger=logger))
    except Exception:
        logger.exception("Main figure generation failed.")
        return generated

    for key, value in generated.items():
        logger.info("Generated main figure %s -> %s", key, value)
    return generated
