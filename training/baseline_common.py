from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from xgboost_common import safe_path, save_json, setup_logger


ModeT = Literal["regression", "binary"]


@dataclass(frozen=True)
class BaselineSpec:
    mode: ModeT
    dataset_prefix: str
    target_column: str
    checkpoint_stem: str
    baseline_name: str


@dataclass
class BaselineModel:
    mode: ModeT
    baseline_name: str
    target_column: str
    train_rows: int
    constant_value: float
    positive_rate: float | None = None
    majority_class: int | None = None
    metadata: dict[str, Any] | None = None


BASELINE_SPECS: dict[ModeT, BaselineSpec] = {
    "regression": BaselineSpec(
        mode="regression",
        dataset_prefix="regression_score",
        target_column="regression_score",
        checkpoint_stem="baseline_regression",
        baseline_name="mean_regression_score",
    ),
    "binary": BaselineSpec(
        mode="binary",
        dataset_prefix="binary_label",
        target_column="binary_label",
        checkpoint_stem="baseline_binary",
        baseline_name="majority_class",
    ),
}


def get_spec(mode: ModeT) -> BaselineSpec:
    try:
        return BASELINE_SPECS[mode]
    except KeyError as exc:
        raise ValueError(f"Unsupported baseline mode: {mode}") from exc


def checkpoint_path(out_dir: str | Path, mode: ModeT) -> str:
    spec = get_spec(mode)
    return safe_path(out_dir, "checkpoints", f"{spec.checkpoint_stem}.json")


def metrics_json_path(out_dir: str | Path, mode: ModeT, split: str) -> str:
    spec = get_spec(mode)
    return safe_path(out_dir, "predictions", f"{spec.checkpoint_stem}_{split}_metrics.json")


def prediction_csv_path(out_dir: str | Path, mode: ModeT, split: str) -> str:
    spec = get_spec(mode)
    return safe_path(out_dir, "predictions", f"{spec.checkpoint_stem}_{split}_predictions.csv")


def load_split_frame(
    base_dir: str | Path,
    split: str,
    mode: ModeT,
    *,
    regression_target_column: str | None = None,
) -> pd.DataFrame:
    spec = get_spec(mode)
    target_column = regression_target_column if mode == "regression" and regression_target_column else spec.target_column
    path = Path(base_dir) / f"{spec.dataset_prefix}_{split}.csv"
    usecols = ["review_id", target_column]
    return pd.read_csv(path, usecols=usecols)


def train_regression_baseline(frame: pd.DataFrame, target_column: str) -> BaselineModel:
    target = pd.to_numeric(frame[target_column], errors="raise").to_numpy(dtype=float)
    return BaselineModel(
        mode="regression",
        baseline_name="mean_regression_score",
        target_column=target_column,
        train_rows=int(target.shape[0]),
        constant_value=float(target.mean()),
        metadata={
            "target_mean": float(target.mean()),
            "target_std": float(target.std(ddof=0)),
            "target_min": float(target.min()),
            "target_max": float(target.max()),
        },
    )


def train_binary_baseline(frame: pd.DataFrame, target_column: str) -> BaselineModel:
    labels = pd.to_numeric(frame[target_column], errors="raise").to_numpy(dtype=int)
    positive_rate = float((labels == 1).mean())
    positive_count = int((labels == 1).sum())
    negative_count = int((labels == 0).sum())
    majority_class = 1 if positive_count >= negative_count else 0
    return BaselineModel(
        mode="binary",
        baseline_name="majority_class",
        target_column=target_column,
        train_rows=int(labels.shape[0]),
        constant_value=float(majority_class),
        positive_rate=positive_rate,
        majority_class=majority_class,
        metadata={
            "positive_count": positive_count,
            "negative_count": negative_count,
            "positive_rate": positive_rate,
        },
    )


def save_model(model: BaselineModel, path: str | Path) -> None:
    payload = asdict(model)
    save_json(path, payload)


def load_model(path: str | Path) -> BaselineModel:
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return BaselineModel(**payload)


def evaluate_regression(frame: pd.DataFrame, model: BaselineModel) -> tuple[pd.DataFrame, dict[str, Any]]:
    y_true = pd.to_numeric(frame[model.target_column], errors="raise").to_numpy(dtype=float)
    y_pred = np.full(shape=y_true.shape[0], fill_value=float(model.constant_value), dtype=float)
    if np.unique(y_true).size <= 1 or np.unique(y_pred).size <= 1:
        spearman_rho = float("nan")
    else:
        spearman_rho = float(pd.Series(y_true).corr(pd.Series(y_pred), method="spearman"))

    prediction_df = frame.loc[:, ["review_id"]].copy()
    prediction_df["y_true"] = y_true
    prediction_df["y_pred"] = y_pred

    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "baseline_value": float(model.constant_value),
        "baseline_constant_value": float(model.constant_value),
        "spearman_rho": spearman_rho,
    }
    return prediction_df, metrics


def evaluate_binary(frame: pd.DataFrame, model: BaselineModel) -> tuple[pd.DataFrame, dict[str, Any]]:
    labels = pd.to_numeric(frame[model.target_column], errors="raise").to_numpy(dtype=int)
    pred_label = int(model.majority_class if model.majority_class is not None else round(model.constant_value))
    prob_pos = float(model.positive_rate if model.positive_rate is not None else pred_label)
    positive_prevalence = float((labels == 1).mean())

    preds = np.full(shape=labels.shape[0], fill_value=pred_label, dtype=int)
    probs = np.full(shape=labels.shape[0], fill_value=prob_pos, dtype=float)

    prediction_df = frame.loc[:, ["review_id"]].copy()
    prediction_df["true_label"] = labels
    prediction_df["pred_label"] = preds
    prediction_df["prob_pos"] = probs

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "majority_class": pred_label,
        "positive_prevalence": positive_prevalence,
        "prob_pos": prob_pos,
    }
    metrics["auroc"] = float(roc_auc_score(labels, probs)) if np.unique(labels).size > 1 else float("nan")
    metrics["auprc"] = float(average_precision_score(labels, probs))
    return prediction_df, metrics
